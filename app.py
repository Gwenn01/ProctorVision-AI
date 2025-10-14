import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Hide TensorFlow logs
os.environ["GLOG_minloglevel"] = "2"

from flask import Flask, jsonify
from flask_cors import CORS

# ----------------------------------------------
# Flask Initialization
# ----------------------------------------------
app = Flask(__name__)

# Allow requests from both local and deployed clients
CORS(
    app,
    resources={r"/api/*": {
        "origins": [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "https://proctorvision-client.vercel.app"  # <-- your frontend URL
        ]
    }},
    supports_credentials=True,
)

# ----------------------------------------------
# Import Blueprints (Routes)
# ----------------------------------------------
from routes.classification_routes import classification_bp
from routes.video_routes import video_bp
from routes.webrtc_routes import webrtc_bp

# Register them under /api
app.register_blueprint(classification_bp, url_prefix="/api")
app.register_blueprint(video_bp, url_prefix="/api")
app.register_blueprint(webrtc_bp, url_prefix="/api")

# ----------------------------------------------
# Root & Health Check
# ----------------------------------------------
@app.route("/")
def home():
    return jsonify({
        "status": "ok",
        "message": "✅ ProctorVision AI Backend Running",
        "routes": ["/api/classify", "/api/video_feed", "/api/webrtc"]
    })


# ----------------------------------------------
# Run the App
# ----------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))  # Railway will set PORT env
    app.run(host="0.0.0.0", port=port, debug=False)
