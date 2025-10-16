import os

# -------------------------------------------------------------
# Environment Configuration
# -------------------------------------------------------------
# Use writable directories for model/data/temp
os.environ["MODEL_DIR"] = "/tmp/model"
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # Suppress TensorFlow INFO/WARN
os.environ["GLOG_minloglevel"] = "2"       # Suppress Mediapipe logs

# Create /tmp directories (Hugging Face allows writes only under /tmp)
os.makedirs(os.environ["MODEL_DIR"], exist_ok=True)
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

from flask import Flask, jsonify
from flask_cors import CORS

# -------------------------------------------------------------
# Flask Initialization
# -------------------------------------------------------------
app = Flask(__name__)

# CORS Configuration
CORS(
    app,
    resources={
        r"/api/*": {
            "origins": [
                "http://localhost:3000",
                "http://127.0.0.1:3000",
                "https://proctorvision-client.vercel.app",
                "https://proctorvision-server-production.up.railway.app",
            ]
        }
    },
    supports_credentials=True,
)

# -------------------------------------------------------------
# Import Blueprints (with diagnostics)
# -------------------------------------------------------------
try:
    print("🔍 Attempting to import blueprints...")

    from routes.classification_routes import classification_bp
    from routes.webrtc_routes import webrtc_bp

    print("✅ Successfully imported blueprints!")

    app.register_blueprint(classification_bp, url_prefix="/api")
    app.register_blueprint(webrtc_bp, url_prefix="/api")

    print("✅ Blueprints registered successfully.")

except Exception as e:
    # Log permission or import errors clearly
    print(f"⚠️ Failed to import or register blueprints: {e}")

# -------------------------------------------------------------
# Log Registered Routes
# -------------------------------------------------------------
print("Registered routes:")
for rule in app.url_map.iter_rules():
    print(" ", rule)

# -------------------------------------------------------------
# Root & Health Check Route
# -------------------------------------------------------------
@app.route("/")
def home():
    routes = [str(rule) for rule in app.url_map.iter_rules()]
    return jsonify({
        "status": "ok",
        "message": "✅ ProctorVision AI Backend Running",
        "available_routes": routes
    })

# -------------------------------------------------------------
# Main Entrypoint
# -------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))  # Hugging Face default port
    debug = os.environ.get("DEBUG", "False").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
