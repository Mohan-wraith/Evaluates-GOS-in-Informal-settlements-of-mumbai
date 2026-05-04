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

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB max upload
CORS(app)

# ─── Class definitions ────────────────────────────────────────────────────────
CLASS_NAMES = [
    "Built-Up", "Vegetation", "Barren",
    "Water", "Impervious Surfaces", "Informal Settlements"
]

CLASS_COLORS = np.array([
    [200, 200, 200],   # 0 Built-Up        – gray
    [80,  140,  50],   # 1 Vegetation      – green
    [200, 160,  40],   # 2 Barren          – amber
    [40,  120, 240],   # 3 Water           – blue
    [100, 100, 150],   # 4 Impervious      – slate
    [250, 235, 185],   # 5 Informal        – cream
], dtype=np.uint8)

PATCH_SIZE = 64
model = None

# ─── Model loading ────────────────────────────────────────────────────────────
def load_model():
    global model
    print("🔄 Loading VGG19-UNet model…")

    model_path = os.path.join(os.getcwd(), "models", "vgg19_unet_final.keras")

    # ensure folder exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # download if not present
    if not os.path.exists(model_path):
        print("⬇️ Downloading model from Google Drive...")
        import gdown

        url = "https://drive.google.com/uc?id=1o7fu5j8ubWm6OW06lZbG1mYwDe3eG-A_"
        gdown.download(url, model_path, quiet=False)

    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        print(f"✅ Model loaded ({len(model.layers)} layers)")
        return True
    except Exception as e:
        print(f"❌ Load error: {e}")
        return False
# ─── Helpers ──────────────────────────────────────────────────────────────────
def open_image(file_bytes: bytes) -> Image.Image:
    """Open any uploaded image (TIFF, PNG, JPG …) and return RGB PIL Image."""
    img = Image.open(io.BytesIO(file_bytes))
    if img.mode not in ("RGB",):
        img = img.convert("RGB")
    return img

def create_colored_mask(predictions: np.ndarray) -> np.ndarray:
    return CLASS_COLORS[predictions]

def calculate_gos(predictions: np.ndarray) -> dict:
    total = predictions.size
    pcts = {i: float(np.sum(predictions == i)) / total * 100 for i in range(6)}
    return {
        "builtUp":      f"{pcts[0]:.2f}",
        "vegetation":   f"{pcts[1]:.2f}",
        "barren":       f"{pcts[2]:.2f}",
        "water":        f"{pcts[3]:.2f}",
        "impervious":   f"{pcts[4]:.2f}",
        "informal":     f"{pcts[5]:.2f}",
        # GOS = Vegetation + Barren (open but usable land)
        "gos":          f"{pcts[1] + pcts[2]:.2f}",
        # Strict GOS = vegetation only
        "gosVegOnly":   f"{pcts[1]:.2f}",
        "isSlum":       bool(pcts[5] > 10),
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

@app.route("/model-info", methods=["GET"])
def model_info():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503
    return jsonify({
        "name": "VGG19-UNet",
        "accuracy": 93.72,
        "f1Score": 94.79,
        "classes": CLASS_NAMES,
        "inputSize": [64, 64, 3],
        "perClassF1": {
            "Built-Up": 95.1,
            "Vegetation": 91.3,
            "Barren": 88.7,
            "Water": 97.2,
            "Impervious": 89.4,
            "Informal Settlements": 94.79
        }
    })

# ── Single-patch prediction (original behaviour) ──────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    t0 = time.time()
    try:
        img = open_image(request.files["image"].read())
        orig_w, orig_h = img.size

        arr = np.array(img.resize((PATCH_SIZE, PATCH_SIZE), Image.BILINEAR),
                       dtype=np.float32) / 255.0
        arr = np.expand_dims(arr, 0)

        preds = np.argmax(model.predict(arr, verbose=0)[0], axis=-1)
        colored = create_colored_mask(preds)

        display_w = max(256, orig_w)
        display_h = max(256, orig_h)
        mask_img = Image.fromarray(colored, "RGB").resize(
            (display_w, display_h), Image.NEAREST
        )

        ms = int((time.time() - t0) * 1000)
        print(f"  [patch] {orig_w}×{orig_h}px → {ms} ms")

        return jsonify({
            "success": True,
            "mask": img_to_b64_png(mask_img),
            "gos": calculate_gos(preds),
            "processingTime": ms,
            "model": "VGG19-UNet",
            "mode": "patch",
            "patchCount": 1,
            "dimensions": {"width": orig_w, "height": orig_h},
        })

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


# ── Full-scene tile-stitching prediction (NEW) ────────────────────────────────
@app.route("/predict-full", methods=["POST"])
def predict_full():
    """
    Accepts any-size image.
    Slices it into 64×64 patches, batch-predicts all patches,
    stitches predictions back into a full-resolution segmentation map.
    """
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    t0 = time.time()
    try:
        img = open_image(request.files["image"].read())
        orig_w, orig_h = img.size

        # Cap at 2048 to avoid OOM
        MAX_DIM = 2048
        if orig_w > MAX_DIM or orig_h > MAX_DIM:
            scale   = MAX_DIM / max(orig_w, orig_h)
            img     = img.resize((int(orig_w * scale), int(orig_h * scale)), Image.BILINEAR)
            orig_w, orig_h = img.size

        P       = PATCH_SIZE
        n_cols  = math.ceil(orig_w / P)
        n_rows  = math.ceil(orig_h / P)
        pad_w   = n_cols * P
        pad_h   = n_rows * P
        total   = n_rows * n_cols

        print(f"  [full] {orig_w}×{orig_h}px → {n_rows}×{n_cols} grid = {total} patches")

        # Pad
        padded = Image.new("RGB", (pad_w, pad_h), (0, 0, 0))
        padded.paste(img, (0, 0))
        padded_arr = np.array(padded, dtype=np.float32) / 255.0

        # Collect patches
        patches = np.stack([
            padded_arr[r * P:(r + 1) * P, c * P:(c + 1) * P]
            for r in range(n_rows) for c in range(n_cols)
        ])  # (N, 64, 64, 3)

        # Batch predict
        BATCH = 64
        all_preds = np.concatenate([
            np.argmax(model.predict(patches[i:i + BATCH], verbose=0), axis=-1)
            for i in range(0, len(patches), BATCH)
        ])  # (N, 64, 64)

        # Stitch
        full_pred = np.zeros((pad_h, pad_w), dtype=np.uint8)
        for idx, pred in enumerate(all_preds):
            r, c = divmod(idx, n_cols)
            full_pred[r * P:(r + 1) * P, c * P:(c + 1) * P] = pred

        full_pred = full_pred[:orig_h, :orig_w]   # crop padding

        colored   = create_colored_mask(full_pred)
        mask_img  = Image.fromarray(colored, "RGB")

        ms = int((time.time() - t0) * 1000)
        print(f"  [full] done in {ms} ms")

        return jsonify({
            "success": True,
            "mask": img_to_b64_png(mask_img),
            "gos": calculate_gos(full_pred),
            "processingTime": ms,
            "model": "VGG19-UNet",
            "mode": "full",
            "patchCount": total,
            "dimensions": {"width": orig_w, "height": orig_h},
        })

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n╔══════════════════════════════════════════════════╗")
    print("║  Mumbai GOS Analyzer  –  Backend  v2.0          ║")
    print("╚══════════════════════════════════════════════════╝\n")

    if load_model():
        print("\n🚀  http://localhost:5000")
        print("    GET  /health")
        print("    GET  /model-info")
        print("    POST /predict          single patch")
        print("    POST /predict-full     full scene tile-stitch\n")
        app.run(host="0.0.0.0", port=5000, debug=False)
    else:
        print("\n❌  Could not load model — check path in server.py\n")