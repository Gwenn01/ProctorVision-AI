import asyncio, time, traceback, os, threading, base64, cv2, numpy as np, mediapipe as mp, requests
from collections import defaultdict, deque
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole
from flask import Blueprint, request, jsonify

# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------
webrtc_bp = Blueprint("webrtc", __name__)

# Base URL of your main (Railway) backend
RAILWAY_API = os.getenv("RAILWAY_API", "").rstrip("/")
if not RAILWAY_API:
    print("⚠️ WARNING: RAILWAY_API not set — backend communication may fail.")

SUMMARY_EVERY_S = float(os.getenv("PROCTOR_SUMMARY_EVERY_S", "1.0"))
RECV_TIMEOUT_S  = float(os.getenv("PROCTOR_RECV_TIMEOUT_S", "5.0"))
HEARTBEAT_S     = float(os.getenv("PROCTOR_HEARTBEAT_S", "10.0"))

# ----------------------------------------------------------------------
# LOGGING UTIL
# ----------------------------------------------------------------------
def log(event, sid="-", eid="-", **kv):
    tail = " ".join(f"{k}={v}" for k, v in kv.items())
    print(f"[{event}] sid={sid} eid={eid} {tail}".strip(), flush=True)

# ----------------------------------------------------------------------
# HELPER: send background POST to Railway backend
# ----------------------------------------------------------------------
def _send_to_railway(endpoint, payload, sid, eid):
    """Send POST requests asynchronously to Railway backend."""
    def _worker():
        try:
            url = f"{RAILWAY_API}{endpoint}"
            r = requests.post(url, json=payload, timeout=10)
            if r.status_code != 200:
                log("RAILWAY_POST_FAIL", sid, eid, code=r.status_code, msg=r.text)
        except Exception as e:
            log("RAILWAY_POST_ERR", sid, eid, err=str(e))
    threading.Thread(target=_worker, daemon=True).start()

# ----------------------------------------------------------------------
# GLOBAL STATE
# ----------------------------------------------------------------------
_loop = asyncio.new_event_loop()
threading.Thread(target=_loop.run_forever, daemon=True).start()
pcs = set()
last_warning = defaultdict(lambda: {"warning": "Looking Forward", "at": 0})
last_capture = defaultdict(lambda: {"label": None, "at": 0})
last_metrics = defaultdict(lambda: {"yaw": None, "pitch": None, "dx": None, "dy": None,
                                    "fps": None, "label": "n/a", "at": 0})

# ----------------------------------------------------------------------
# MEDIAPIPE SETUP
# ----------------------------------------------------------------------
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=1, refine_landmarks=True,
    min_detection_confidence=0.6, min_tracking_confidence=0.6
)
hands = mp_hands.Hands(
    static_image_mode=False, max_num_hands=2,
    min_detection_confidence=0.6, min_tracking_confidence=0.6
)

# ----------------------------------------------------------------------
# DETECTOR CLASS
# ----------------------------------------------------------------------
IDX_NOSE, IDX_CHIN, IDX_LE, IDX_RE, IDX_LM, IDX_RM = 1, 152, 263, 33, 291, 61
MODEL_3D = np.array([
    [0.0,   0.0,   0.0],
    [0.0, -63.6, -12.5],
    [-43.3, 32.7, -26.0],
    [43.3,  32.7, -26.0],
    [-28.9, -28.9, -24.1],
    [28.9,  -28.9, -24.1],
], dtype=np.float32)

def _landmarks_to_pts(lms, w, h):
    ids = [IDX_NOSE, IDX_CHIN, IDX_LE, IDX_RE, IDX_LM, IDX_RM]
    return np.array([[lms[i].x * w, lms[i].y * h] for i in ids], dtype=np.float32)

def _bbox_from_landmarks(lms, w, h, pad=0.03):
    xs = [p.x for p in lms]; ys = [p.y for p in lms]
    x1n, y1n = max(0.0, min(xs) - pad), max(0.0, min(ys) - pad)
    x2n, y2n = min(1.0, max(xs) + pad), min(1.0, max(ys) + pad)
    return (int(x1n*w), int(y1n*h), int(x2n*w), int(y2n*h))

# Thresholds
YAW_DEG_TRIG, PITCH_UP, PITCH_DOWN = 12, 10, 16
DX_TRIG, DY_UP, DY_DOWN = 0.06, 0.08, 0.10
SMOOTH_N, CAPTURE_MIN_MS = 5, 1200
HOLD_FRAMES_HEAD, HOLD_FRAMES_NOFACE, HOLD_FRAMES_HAND = 3, 3, 5

class ProctorDetector:
    def __init__(self):
        self.yaw_hist, self.pitch_hist, self.dx_hist, self.dy_hist = deque(maxlen=SMOOTH_N), deque(maxlen=SMOOTH_N), deque(maxlen=SMOOTH_N), deque(maxlen=SMOOTH_N)
        self.base_yaw = self.base_pitch = None
        self.last_capture_ms, self.noface_streak, self.hand_streak = 0, 0, 0
        self.last_print = 0.0

    def _pose_angles(self, lms, w, h):
        try:
            pts2d = _landmarks_to_pts(lms, w, h)
            cam = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]], dtype=np.float32)
            ok, rvec, _ = cv2.solvePnP(MODEL_3D, pts2d, cam, np.zeros((4,1)))
            if not ok: return None, None
            R, _ = cv2.Rodrigues(rvec)
            _, _, euler = cv2.RQDecomp3x3(R)
            pitch, yaw, _ = map(float, euler)
            return yaw, pitch
        except Exception:
            return None, None

    def detect(self, bgr, sid="-", eid="-"):
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)
        if not res.multi_face_landmarks:
            log("FRAME", sid, eid, note="no_face")
            self.noface_streak += 1
            return "No Face", None, rgb
        self.noface_streak = 0
        lms = res.multi_face_landmarks[0].landmark
        yaw, pitch = self._pose_angles(lms, w, h)
        label = "Looking Forward"
        if yaw and abs(yaw) > YAW_DEG_TRIG: label = "Looking Left" if yaw < 0 else "Looking Right"
        if pitch and pitch > PITCH_DOWN: label = "Looking Down"
        if pitch and -pitch > PITCH_UP: label = "Looking Up"
        return label, _bbox_from_landmarks(lms, w, h), rgb

    def detect_hands_anywhere(self, rgb):
        res = hands.process(rgb)
        if not res.multi_hand_landmarks:
            self.hand_streak = 0
            return None
        self.hand_streak += 1
        return "Hand Detected"

    def _throttle_ok(self):
        return int(time.time()*1000) - self.last_capture_ms >= CAPTURE_MIN_MS
    def _mark_captured(self): self.last_capture_ms = int(time.time()*1000)

detectors = defaultdict(ProctorDetector)

# ----------------------------------------------------------------------
# CAPTURE HANDLER — NOW CALLS RAILWAY API
# ----------------------------------------------------------------------
def _maybe_capture(student_id: str, exam_id: str, bgr, label: str):
    ok, buf = cv2.imencode(".jpg", bgr)
    if not ok:
        log("CAPTURE_SKIP", student_id, exam_id, reason="encode_failed")
        return

    img_b64 = base64.b64encode(buf).decode("utf-8")
    log("CAPTURE_ENQUEUE", student_id, exam_id, label=label, bytes=len(buf))

    # 👉 send to Railway backend instead of local DB
    _send_to_railway("/api/save_behavior_log", {
        "user_id": int(student_id),
        "exam_id": int(exam_id),
        "image_base64": img_b64,
        "warning_type": label
    }, student_id, exam_id)

    _send_to_railway("/api/increment_suspicious", {
        "student_id": int(student_id)
    }, student_id, exam_id)

    ts = int(time.time() * 1000)
    last_capture[(student_id, exam_id)] = {"label": label, "at": ts}
    log("LAST_CAPTURE_SET", student_id, exam_id, label=label, at=ts)

# ----------------------------------------------------------------------
# WEBRTC OFFER HANDLER
# ----------------------------------------------------------------------
async def _wait_ice_complete(pc):
    if pc.iceGatheringState == "complete": return
    done = asyncio.Event()
    @pc.on("icegatheringstatechange")
    def _(_ev=None):
        if pc.iceGatheringState == "complete": done.set()
    await asyncio.wait_for(done.wait(), timeout=5.0)

async def handle_offer(data):
    sid, eid = str(data.get("student_id", "0")), str(data.get("exam_id", "0"))
    log("OFFER_HANDLE", sid, eid)
    offer = RTCSessionDescription(sdp=data["sdp"], type=data["type"])
    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def _():
        if pc.connectionState in ("failed", "closed", "disconnected"):
            await pc.close()
            pcs.discard(pc)
            for d in (detectors, last_warning, last_metrics, last_capture):
                d.pop((sid, eid), None)
            log("PC_CLOSED", sid, eid)

    @pc.on("track")
    def on_track(track):
        log("TRACK", sid, eid, kind=track.kind)
        if track.kind != "video":
            MediaBlackhole().addTrack(track)
            return
        async def reader():
            det = detectors[(sid, eid)]
            while True:
                try:
                    frame = await asyncio.wait_for(track.recv(), timeout=RECV_TIMEOUT_S)
                except Exception as e:
                    log("TRACK_RECV_ERR", sid, eid, err=str(e))
                    break
                try:
                    bgr = frame.to_ndarray(format="bgr24")
                    head_label, _, rgb = det.detect(bgr, sid, eid)
                    hand_label = det.detect_hands_anywhere(rgb)
                    warn = hand_label or head_label
                    ts = int(time.time() * 1000)
                    last_warning[(sid, eid)] = {"warning": warn, "at": ts}
                    if det._throttle_ok() and warn not in ("Looking Forward", None):
                        _maybe_capture(sid, eid, bgr, warn)
                        det._mark_captured()
                except Exception as e:
                    log("DETECT_ERR", sid, eid, err=str(e))
                    continue
        asyncio.ensure_future(reader(), loop=_loop)

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    await _wait_ice_complete(pc)
    return pc.localDescription

# ----------------------------------------------------------------------
# ROUTES
# ----------------------------------------------------------------------
@webrtc_bp.route("/webrtc/offer", methods=["POST"])
def webrtc_offer():
    try:
        data = request.get_json(force=True)
        desc = asyncio.run_coroutine_threadsafe(handle_offer(data), _loop).result()
        return jsonify({"sdp": desc.sdp, "type": desc.type})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@webrtc_bp.route("/webrtc/cleanup", methods=["POST"])
def webrtc_cleanup():
    async def _close_all():
        for pc in list(pcs):
            await pc.close()
            pcs.discard(pc)
    asyncio.run_coroutine_threadsafe(_close_all(), _loop)
    return jsonify({"ok": True})

@webrtc_bp.route("/proctor/last_warning")
def proctor_last_warning():
    sid, eid = request.args.get("student_id"), request.args.get("exam_id")
    if not sid or not eid:
        return jsonify(error="missing student_id or exam_id"), 400
    return jsonify(last_warning.get((sid, eid), {"warning": "Looking Forward", "at": 0}))

@webrtc_bp.route("/proctor/last_capture")
def proctor_last_capture():
    sid, eid = request.args.get("student_id"), request.args.get("exam_id")
    if not sid or not eid:
        return jsonify(error="missing student_id or exam_id"), 400
    return jsonify(last_capture.get((sid, eid), {"label": None, "at": 0}))
