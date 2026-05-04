from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
import time
import math
import os

# ─── App ──────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
CORS(app)

# ─── Class definitions ────────────────────────────────────────────────────────
CLASS_NAMES = [
    "Built-Up", "Vegetation", "Barren",
    "Water", "Impervious Surfaces", "Informal Settlements"
]

CLASS_COLORS = np.array([
    [200, 200, 200],
    [80, 140, 50],
    [200, 160, 40],
    [40, 120, 240],
    [100, 100, 150],
    [250, 235, 185],
], dtype=np.uint8)

PATCH_SIZE = 64
model = None

# ─── Model loading ────────────────────────────────────────────────────────────
def load_model():
    global model
    print("🔄 Loading VGG19-UNet model…")

    model_path = os.path.join(os.getcwd(), "models", "vgg19_unet_final.keras")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    if not os.path.exists(model_path):
        print("⬇️ Downloading model from Google Drive...")
        import gdown
        url = "https://drive.google.com/uc?id=1o7fu5j8ubWm6OW06lZbG1mYwDe3eG-A_"
        gdown.download(url, model_path, quiet=False)

    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        print("✅ Model loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Model load error: {e}")
        return False


# 🔥 IMPORTANT FIX: LOAD MODEL WHEN SERVER STARTS (for Render/Gunicorn)
load_model()


# ─── Helpers ──────────────────────────────────────────────────────────────────
def open_image(file_bytes: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(file_bytes))
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img

def create_colored_mask(predictions: np.ndarray) -> np.ndarray:
    return CLASS_COLORS[predictions]

def calculate_gos(predictions: np.ndarray) -> dict:
    total = predictions.size
    pcts = {i: float(np.sum(predictions == i)) / total * 100 for i in range(6)}
    return {
        "builtUp": f"{pcts[0]:.2f}",
        "vegetation": f"{pcts[1]:.2f}",
        "barren": f"{pcts[2]:.2f}",
        "water": f"{pcts[3]:.2f}",
        "impervious": f"{pcts[4]:.2f}",
        "informal": f"{pcts[5]:.2f}",
        "gos": f"{pcts[1] + pcts[2]:.2f}",
        "gosVegOnly": f"{pcts[1]:.2f}",
        "isSlum": bool(pcts[5] > 10),
    }

def img_to_b64_png(pil_img: Image.Image) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model": model is not None,
        "modelName": "VGG19-UNet",
        "accuracy": "93.72%",
        "version": "2.0"
    })

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        img = open_image(request.files["image"].read())
        arr = np.array(img.resize((PATCH_SIZE, PATCH_SIZE))) / 255.0
        arr = np.expand_dims(arr, 0)

        preds = np.argmax(model.predict(arr)[0], axis=-1)
        mask = create_colored_mask(preds)

        return jsonify({
            "success": True,
            "mask": img_to_b64_png(Image.fromarray(mask)),
            "gos": calculate_gos(preds)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─── Entry point (only for local run) ──────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)