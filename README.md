# Deepfake & AI Media Detection System 🛡️🔍

An advanced forensic and deep learning application designed to detect AI-generated and manipulated facial media (GANs, Diffusion models, FaceSwap, Grok, Midjourney, StyleGAN). Features **Explainable AI (XAI)** through Error Level Analysis (ELA) heatmaps and a multi-modal ensemble classifier.

---

## 🌟 Key Features

- **Multi-Modal Forensic Architecture**:
  - **Error Level Analysis (ELA)**: Quantifies compression rate inconsistencies and manipulation boundaries.
  - **2D Fast Fourier Transform (FFT)**: Identifies spectral lattice grid patterns characteristic of generative AI.
  - **Micro-Texture & Denoising Analysis**: Detects unnatural skin smoothing vs. high-contrast edge gradients.
  - **AI Generator Badge Scanner**: Detects known watermarks and signatures in corner regions.
- **Stacked Ensemble Model**: Combines **Random Forest**, **Extra Trees**, **Gradient Boosting**, and **Multi-Layer Perceptron (MLP)** over a 48-dimensional multi-scale forensic feature vector.
- **Explainable AI Heatmap**: Side-by-side visual comparison between original images and ELA artifact heatmaps.
- **Modern Responsive Web UI**: Clean glassmorphism dark theme with live image upload previews, calibrated confidence meters, and dual real vs. fake distribution bars.
- **RESTful API**: `/api/predict` endpoint with CORS enabled for automated integrations.

---

## 📊 Dataset Benchmark Evaluation

Evaluated across genuine portraits (FFHQ / CelebA-HQ) and synthetic faces (StyleGAN / Diffusion):

| Metric | Score |
| :--- | :--- |
| **Accuracy** | **91.38%** |
| **Precision** | **92.65%** |
| **Recall** | **93.84%** |
| **F1-Score** | **93.24%** |

---

## 📁 Repository Structure

```text
├── app.py                  # Flask web server & REST API
├── model.py                # Multi-modal inference pipeline & ensemble loader
├── utils.py                # Image preprocessing, ELA heatmap generator, watermark scanner
├── train.py                # Model training, dataset pooling & calibration script
├── fetch_dataset.py        # Automated real/AI web image downloader
├── requirements.txt        # Pinned Python package dependencies
├── model_weights/
│   ├── forensic_ensemble.pkl  # Trained 4-model ensemble weights
│   └── metrics.json           # Evaluation metrics & confusion matrix
├── static/
│   ├── css/style.css       # Modern dark dashboard stylesheet
│   └── uploads/            # Uploaded images & generated heatmaps
└── templates/
    ├── index.html          # Upload page with live preview & benchmark badge
    ├── result.html         # Forensic report & Explainable AI comparison
    └── error.html          # Graceful error handler
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/raghunath485/Deepfake-Detector.git
cd Deepfake-Detector
```

### 2. Set Up Virtual Environment
```bash
python -m venv .venv

# On Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# On Linux / macOS
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
python app.py
```
Open [http://localhost:5000](http://localhost:5000) in your browser.

---

## 🧠 Model Training

To retrain the ensemble classifier on your own dataset or downloaded images:
```bash
# Optional: Download additional open-access real and AI portraits
python fetch_dataset.py

# Train the 4-way balanced ensemble
python train.py
```

---

## 🔌 REST API Usage

```bash
curl -X POST -F "file=@sample.jpg" http://localhost:5000/api/predict
```

**Sample Response**:
```json
{
  "filename": "sample.jpg",
  "label": "Deepfake / AI-Generated",
  "confidence": 88.5,
  "fake_probability": 88.5,
  "real_probability": 11.5,
  "findings": [
    "Trained forensic model identified deepfake characteristics (88.5% confidence).",
    "High compression variance detected via Error Level Analysis (ELA std: 6.2)."
  ],
  "ela_heatmap": "/static/uploads/ela_sample.jpg"
}
```

---

## 📄 License
This project is open-source under the MIT License for educational and research purposes.
