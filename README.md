bash

cat << 'HEREDOC' > /mnt/user-data/outputs/README.md
# 🌿 Evaluating Green and Open Spaces in Informal Settlements of Mumbai Using Deep Learning

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16.1-orange?style=for-the-badge&logo=tensorflow)
![Flask](https://img.shields.io/badge/Flask-2.3.3-black?style=for-the-badge&logo=flask)
![Next.js](https://img.shields.io/badge/Next.js-15-white?style=for-the-badge&logo=next.js)
![Keras](https://img.shields.io/badge/Keras-3.x-red?style=for-the-badge&logo=keras)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**M.Sc. Data Science Thesis Project — GITAM University, Visakhapatnam**

[Live Demo](https://mohangodofnewworld-gos-analyzer.hf.space/health) • [Backend API](https://mohangodofnewworld-gos-analyzer.hf.space) • [Report an Issue](https://github.com/Mohan-wraith/Evaluates-GOS-in-Informal-settlements-of-mumbai/issues)

</div>

---

## 📌 Overview

Mumbai is home to **10–12 million people** living in informal settlements — nearly half the city's population — crammed into just **8% of the total land area**. Within these settlements, access to green and open spaces (GOS) is critically limited, with direct consequences for physical health, mental wellbeing, thermal comfort, and air quality.

This project presents a **complete end-to-end deep learning framework** that:

- Automatically classifies every pixel in a **Pleiades-1A satellite image** (0.5 m resolution) into 6 urban land cover categories
- Computes a **Green and Open Space (GOS) Index** for any uploaded satellite image
- Quantifies the **spatial inequity in GOS access** between informal settlements and planned residential areas
- Delivers all of this through an **interactive web application** accessible to non-specialist urban planners

### 🔑 Key Finding

> Informal settlements in Mumbai average **9.02% GOS** coverage compared to **37.95%** in planned residential areas — a **fourfold disparity** that is now backed by satellite-derived, deep-learning-validated data.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER BROWSER                             │
│   Next.js 15 + React 18 + Tailwind CSS + Recharts + geotiff.js  │
└──────────────────────────┬──────────────────────────────────────┘
                           │ HTTP POST multipart/form-data
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FLASK REST API                              │
│         /predict  •  /predict-full  •  /health  •  /model-info  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                   VGG19-UNet Model                              │
│     TensorFlow 2.16.1 + Keras 3 — 43.4M Parameters             │
│     Tile-Stitch Pipeline: n_cols=ceil(W/64) × n_rows=ceil(H/64) │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              6-Class Segmentation Output                        │
│  Built-Up • Vegetation • Barren • Water • Impervious • Informal │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🧠 Models

| Model | Accuracy | Precision | Recall | Weighted F1 |
|-------|----------|-----------|--------|-------------|
| **VGG19-UNet (Ours)** | **93.72%** | **94.12%** | **93.41%** | **94.79%** |
| ResNet50-UNet (Ours) | 84.73% | 83.61% | 84.41% | 84.09% |
| DeepLab v3+ (Literature) | 89.14% | 88.72% | 88.61% | 88.90% |
| SegNet (Literature) | 85.23% | 84.91% | 85.07% | 85.10% |
| FCN-8s (Literature) | 81.47% | 80.98% | 81.31% | 81.20% |
| Random Forest (Literature) | 74.82% | 73.50% | 74.30% | 74.10% |

### Per-Class F1 Score — VGG19-UNet

| Class | F1 Score |
|-------|----------|
| Built-Up | 95.1% |
| Vegetation | 91.3% |
| Barren | 88.7% |
| Water | 97.2% |
| Impervious Surfaces | 89.4% |
| Informal Settlements | 94.8% |

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| Satellite | Pleiades-1A |
| Resolution | 0.5 m/px |
| Patch Size | 64 × 64 pixels (32 × 32 m ground footprint) |
| Total Patches | 8,910 |
| Training | 7,128 (80%) |
| Validation | 891 (10%) |
| Test | 891 (10%) |
| Classes | 6 |
| Study Area | Dharavi + Bandra-Kurla Complex, Mumbai |

### Land Cover Classes

| ID | Class | Color |
|----|-------|-------|
| 0 | Built-Up | Gray `[200, 200, 200]` |
| 1 | Vegetation | Green `[80, 140, 50]` |
| 2 | Barren | Amber `[200, 160, 40]` |
| 3 | Water | Blue `[40, 120, 240]` |
| 4 | Impervious Surfaces | Slate `[100, 100, 150]` |
| 5 | Informal Settlements | Cream `[250, 235, 185]` |

---

## 🚀 Features

- **Full Scene Tile-Stitching** — Upload any-resolution Pleiades-1A image. The pipeline tiles it into 64×64 patches, predicts each independently, and stitches them back into a full-resolution segmentation map
- **Before/After Drag Slider** — Interactively compare the original satellite image and the segmentation mask using a draggable divider
- **GOS Ring Gauge** — Visual circular indicator displaying the GOS index with colour coding (red/amber/green) by threshold
- **Mumbai Benchmark Panel** — Compares the analysed image against city-wide informal (9.02%) and planned (37.95%) GOS averages
- **Compare Mode** — Side-by-side analysis of two different satellite images with GOS difference computation
- **Configurable GOS Formula** — Toggle between Strict (vegetation only, WHO-aligned) and Extended (vegetation + barren) formulas in real time
- **Client-side TIF Decoding** — Native Pleiades GeoTIFF upload with 2% percentile stretch rendering via geotiff.js — no server-side preprocessing required
- **Multi-format Export** — Download the segmentation mask as PNG, or export results as CSV or JSON

---

## 🛠️ Tech Stack

### Backend
```
Python 3.10
Flask 3.x + Flask-CORS
TensorFlow 2.16.1 + Keras 3
NumPy + Pillow + gdown
Gunicorn (production server)
Deployed on: Hugging Face Spaces (Docker)
```

### Frontend
```
Next.js 15 (App Router)
React 18
Tailwind CSS
Recharts (BarChart, PieChart)
geotiff.js (client-side GeoTIFF decoding)
lucide-react (icons)
Deployed on: Vercel
```

### Training Environment
```
Google Colab (NVIDIA T4 GPU)
TensorFlow 2.16.1 + Keras 3
Focal Loss (γ=2.0, α=0.25)
Adam Optimizer
Two-phase transfer learning from ImageNet
```

---

## 📡 API Reference

**Base URL:** `https://mohangodofnewworld-gos-analyzer.hf.space`

### GET `/health`
Returns server and model status.

**Response:**
```json
{
  "status": "healthy",
  "model": true,
  "modelName": "VGG19-UNet",
  "accuracy": "93.72%",
  "version": "2.0"
}
```

---

### GET `/model-info`
Returns full model metadata and per-class F1 scores.

**Response:**
```json
{
  "name": "VGG19-UNet",
  "accuracy": 93.72,
  "f1Score": 94.79,
  "classes": ["Built-Up", "Vegetation", "Barren", "Water", "Impervious Surfaces", "Informal Settlements"],
  "inputSize": [64, 64, 3],
  "perClassF1": {
    "Built-Up": 95.1,
    "Vegetation": 91.3,
    "Barren": 88.7,
    "Water": 97.2,
    "Impervious": 89.4,
    "Informal Settlements": 94.79
  }
}
```

---

### POST `/predict`
Single-patch prediction. Resizes the input image to 64×64 and runs one forward pass.

**Request:** `multipart/form-data` with field `image` (PNG, JPG, or TIF)

**Response:**
```json
{
  "success": true,
  "mask": "data:image/png;base64,...",
  "gos": {
    "builtUp": "27.42",
    "vegetation": "38.09",
    "barren": "8.02",
    "water": "1.05",
    "impervious": "22.50",
    "informal": "0.16",
    "gos": "46.11",
    "gosVegOnly": "38.09",
    "isSlum": false
  },
  "processingTime": 342,
  "model": "VGG19-UNet",
  "mode": "patch",
  "patchCount": 1,
  "dimensions": {"width": 600, "height": 600}
}
```

---

### POST `/predict-full`
Full-scene tile-stitching prediction. Accepts any-resolution image, tiles into 64×64 patches, batch-predicts all patches, stitches back into full-resolution segmentation map.

**Request:** `multipart/form-data` with field `image`

**Response:** Same structure as `/predict` with `"mode": "full"` and `patchCount` reflecting the number of patches processed.

---

## 🔧 Local Development

### Prerequisites
- Python 3.10+
- Node.js 18+
- Git

### Backend Setup

```bash
# Clone the repository
git clone https://github.com/Mohan-wraith/Evaluates-GOS-in-Informal-settlements-of-mumbai.git
cd Evaluates-GOS-in-Informal-settlements-of-mumbai

# Install dependencies
pip install -r requirements.txt

# Run the server
python server.py
```

The backend will start at `http://localhost:5000`. On first run, the VGG19-UNet model (360 MB) will be automatically downloaded from Google Drive.

### Frontend Setup

```bash
# Navigate to frontend folder
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

The frontend will start at `http://localhost:3000`.

> Make sure the backend is running on port 5000 before starting the frontend.

---

## 📁 Project Structure

```
├── server.py                  # Flask REST API backend
├── requirements.txt           # Python dependencies
├── Procfile                   # Gunicorn start command
├── Dockerfile                 # Docker config for Hugging Face Spaces
├── frontend/
│   ├── app/
│   │   └── page.js            # Main Next.js application
│   ├── public/                # Static assets
│   ├── package.json
│   └── next.config.mjs
└── README.md
```

---

## 🧪 Training Details

| Parameter | Value |
|-----------|-------|
| Patch Size | 64 × 64 pixels |
| Batch Size | 32 |
| Phase 1 LR | 1 × 10⁻⁴ (frozen encoder) |
| Phase 2 LR | 1 × 10⁻⁵ (full fine-tuning) |
| Loss Function | Focal Loss (γ=2.0, α=0.25) |
| Optimizer | Adam (β₁=0.9, β₂=0.999) |
| LR Scheduler | ReduceLROnPlateau (factor=0.5, patience=3) |
| Early Stopping | patience=5, restore_best_weights=True |
| Environment | Google Colab T4 GPU |
| Class Weights | Informal Settlements: 2.0 · Vegetation: 1.5 · Barren: 1.2 |

### Data Augmentation
- Horizontal and vertical flip (p=0.5 each)
- Rotation in 90° multiples
- Brightness adjustment ×[0.8, 1.2]
- All augmentations applied identically to image and mask

---

## 📈 Results

### GOS Index — Key Finding

```
Informal Settlements (Dharavi):     9.02%  ████░░░░░░░░░░░░░░░░░░░░░░░░░░
Planned Areas (Bandra-Kurla):      37.95%  ████████████████████████░░░░░░
WHO Recommendation:                ~43.00%  ████████████████████████████░░
```

**The fourfold GOS disparity (9.02% vs 37.95%) provides the first satellite-derived, deep-learning-validated measurement of green space inequity in Mumbai at neighbourhood scale.**

---

## 🙏 Acknowledgements

- **Guide:** Mr. B. Ravi Kumar, Associate Professor, Department of Computer Science, GITAM School of Science
- **Head of Department:** Prof. T. Uma Devi
- **Dean (Principal):** Prof. S. Anantha Ramakrishna
- **Project Coordinators:** Dr. K. Vanitha Kakollu and Mr. Shaik Shahid
- **Institution:** GITAM (Deemed to be University), Visakhapatnam

---

## 📚 References

1. Ronneberger, O., Fischer, P., and Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. *MICCAI*
2. Simonyan, K., and Zisserman, A. (2015). Very Deep Convolutional Networks for Large-Scale Image Recognition. *ICLR*
3. He, K., Zhang, X., Ren, S., and Sun, J. (2016). Deep Residual Learning for Image Recognition. *CVPR*
4. Lin, T.-Y., et al. (2017). Focal Loss for Dense Object Detection. *ICCV*
5. Yuan, Q., et al. (2020). Deep Learning in Environmental Remote Sensing. *Remote Sensing of Environment*
6. Li, Y., Xu, L., and Chen, Z. (2020). Deep Learning-Based Urban Green Space Mapping for 15 Chinese Cities. *Remote Sensing*
7. World Health Organization. (2016). Urban Green Spaces and Health. *WHO Regional Office for Europe*

---

## 📄 License

This project is licensed under the MIT License.

---

<div align="center">

**M. Mohan Devendra** · Regd. No: 2024119686  
M.Sc. Data Science · GITAM University Visakhapatnam · 2024–2025

*"A result sitting in a Jupyter notebook that nobody can access isn't really a result."*

</div>
HEREDOC
echo "Done: $(wc -l < /mnt/user-data/outputs/README.md) lines"
