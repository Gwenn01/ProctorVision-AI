import os, io, base64, requests
from pathlib import Path
from flask import Blueprint, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image

try:
    from tensorflow.keras.applications import mobilenet_v2 as _mv2
except Exception:
    from keras.applications import mobilenet_v2 as _mv2

preprocess_input = _mv2.preprocess_input
classification_bp = Blueprint('classification_bp', __name__)

# ------------------------------------------------------------
# Model setup and auto-download
# ------------------------------------------------------------
MODEL_DIR = Path(os.getenv("MODEL_DIR", "/tmp/model"))
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_URLS = {
    "model": "https://huggingface.co/Gwen01/ProctorVision-Models/resolve/main/cheating_mobilenetv2_final.keras",
    "threshold": "https://huggingface.co/Gwen01/ProctorVision-Models/resolve/main/best_threshold.npy"
}

MODEL_PATHS = {}
for key, url in MODEL_URLS.items():
    local_path = MODEL_DIR / Path(url).name
    MODEL_PATHS[key] = str(local_path)
    if not local_path.exists():
        print(f"📥 Downloading {key} from Hugging Face…")
        r = requests.get(url)
        r.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(r.content)
        print(f"✅ Saved {key} → {local_path}")

# Candidate filenames for compatibility
CANDIDATES = [
    "cheating_mobilenetv2_final.keras",
    "mnv2_clean_best.keras",
    "mnv2_continue.keras",
    "mnv2_finetune_best.keras",
]


model_path = next((MODEL_DIR / f for f in CANDIDATES if (MODEL_DIR / f).exists()), None)
if model_path and model_path.exists():
    model = tf.keras.models.load_model(model_path, compile=False)
    print(f"✅ Model loaded: {model_path}")
else:
    model = None
    print(f"⚠️ No model found in {MODEL_DIR}. Put one of: {CANDIDATES}")

# --- Load threshold ---
thr_file = MODEL_DIR / "best_threshold.npy"
THRESHOLD = float(np.load(thr_file)[0]) if thr_file.exists() else 0.555
print(f"📊 Using decision threshold: {THRESHOLD:.3f}")

# --- Input shape ---
if model is not None:
    H, W = model.input_shape[1:3]
else:
    H, W = 224, 224  # fallback

LABELS = ["Cheating", "Not Cheating"]

# ------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------
def preprocess_pil(pil_img: Image.Image) -> np.ndarray:
    img = pil_img.convert("RGB")
    if img.size != (W, H):
        img = img.resize((W, H), Image.BILINEAR)
    x = np.asarray(img, dtype=np.float32)
    x = preprocess_input(x)
    return np.expand_dims(x, 0)

def predict_batch(batch_np: np.ndarray) -> np.ndarray:
    probs = model.predict(batch_np, verbose=0).ravel()
    if probs.ndim == 0:
        probs = np.array([probs])
    if len(probs) != batch_np.shape[0]:
        raw = model.predict(batch_np, verbose=0)
        if raw.ndim == 2 and raw.shape[1] == 2:
            probs = raw[:, 1]  # probability of "Not Cheating"
        else:
            probs = raw.ravel()
    return probs

def label_from_prob(prob_non_cheating: float) -> str:
    return LABELS[int(prob_non_cheating >= THRESHOLD)]

# ------------------------------------------------------------
# Environment Variables
# ------------------------------------------------------------
RAILWAY_API = os.getenv("RAILWAY_API", "").rstrip("/")
if not RAILWAY_API:
    print("⚠️ WARNING: RAILWAY_API not set — backend sync will fail.")

# ------------------------------------------------------------
# Route 1 — Classify uploaded multiple files (manual)
# ------------------------------------------------------------
@classification_bp.route('/classify_multiple', methods=['POST'])
def classify_multiple():
    if model is None:
        return jsonify({"error": "Model not loaded."}), 500

    files = request.files.getlist('files') if 'files' in request.files else []
    if not files:
        return jsonify({"error": "No files uploaded"}), 400

    batch = []
    for f in files:
        try:
            pil = Image.open(io.BytesIO(f.read()))
            batch.append(preprocess_pil(pil)[0])
        except Exception as e:
            return jsonify({"error": f"Error reading image: {str(e)}"}), 400

    batch_np = np.stack(batch, axis=0)
    probs = predict_batch(batch_np)
    labels = [label_from_prob(p) for p in probs]

    return jsonify({
        "threshold": THRESHOLD,
        "results": [{"label": lbl, "prob_non_cheating": float(p)} for lbl, p in zip(labels, probs)]
    })

# ------------------------------------------------------------
# Route 2 — Auto-classify Behavior Logs (Backend-to-Backend)
# ------------------------------------------------------------
@classification_bp.route('/classify_behavior_logs', methods=['POST'])
def classify_behavior_logs():
    if model is None:
        return jsonify({"error": "Model not loaded."}), 500

    data = request.get_json(silent=True) or {}
    user_id = data.get('user_id')
    exam_id = data.get('exam_id')
    if not user_id or not exam_id:
        return jsonify({"error": "Missing user_id or exam_id"}), 400

    # --- Fetch behavior logs from Railway ---
    try:
        fetch_url = f"{RAILWAY_API}/api/fetch_behavior_logs"
        response = requests.get(fetch_url, params={"user_id": user_id, "exam_id": exam_id})
        if response.status_code != 200:
            return jsonify({"error": f"Failed to fetch logs: {response.text}"}), 500

        logs = response.json().get("logs", [])
        if not logs:
            return jsonify({"message": "No logs to classify."}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to reach Railway API: {str(e)}"}), 500

    # --- Process & Predict ---
    updates = []
    CHUNK = 64
    for i in range(0, len(logs), CHUNK):
        chunk = logs[i:i+CHUNK]
        batch = []
        ids = []

        for log in chunk:
            try:
                img_data = base64.b64decode(log["image_base64"])
                pil = Image.open(io.BytesIO(img_data))
                batch.append(preprocess_pil(pil)[0])
                ids.append(log["id"])
            except Exception as e:
                print(f"⚠️ Failed to read image ID {log['id']}: {e}")

        if not batch:
            continue

        batch_np = np.stack(batch, axis=0)
        probs = predict_batch(batch_np)
        labels = [label_from_prob(p) for p in probs]

        for log_id, lbl in zip(ids, labels):
            updates.append({"id": log_id, "label": lbl})

    # --- Send predictions back to Railway ---
    try:
        update_url = f"{RAILWAY_API}/api/update_classifications"
        post_res = requests.post(update_url, json={"updates": updates})
        if post_res.status_code != 200:
            return jsonify({"error": f"Failed to update classifications: {post_res.text}"}), 500
    except Exception as e:
        return jsonify({"error": f"Failed to push updates: {str(e)}"}), 500

    return jsonify({
        "message": f"Classification complete for {len(updates)} logs.",
        "threshold": THRESHOLD
    }), 200
