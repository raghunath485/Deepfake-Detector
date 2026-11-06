# Deepfake Image Detection

## Overview
A lightweight Flask web app that detects whether an uploaded face image is **real** or a **deepfake** using a TensorFlow/Keras model.

- **Web UI** – upload an image, see a friendly loading overlay, and view the prediction.
- **REST API** – `/api/predict` (POST) for programmatic usage, CORS enabled.
- **Health endpoint** – `/health` to check if the model is loaded.
- **Docker support** – build and run the app in a container.
- **Training script** – `train.py` provides a quick baseline using EfficientNetB0.

## Project Structure
```
deepfake_detection/
├─ app.py               # Flask server
├─ model.py             # Model loader / inference
├─ utils.py             # Image preprocessing
├─ train.py             # Simple training script
├─ requirements.txt     # Python dependencies
├─ Dockerfile           # Container build
├─ README.md            # This file
├─ .gitignore           # Ignored files
├─ static/
│   ├─ css/style.css
│   └─ uploads/          # Uploaded images (auto‑created)
└─ templates/
    ├─ index.html        # Upload page (with loading overlay)
    ├─ result.html       # Result page
    └─ error.html        # Error page
```

## Setup (local)
```bash
# Clone / copy the project directory
cd deepfake_detection

# Create a virtual environment
python -m venv venv
source venv/bin/activate   # on Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Model
- Place a trained `deepfake_model.h5` inside `model_weights/`.
- Or train a quick prototype:
```bash
python train.py
```
  This will create `model_weights/deepfake_model.h5`.

### Run the app
```bash
python app.py
```
Open <http://127.0.0.1:5000> in a browser.

## Docker
```bash
# Build
docker build -t deepfake-detector .
# Run (mount model_weights if you have a custom model)
docker run -p 5000:5000 -v $(pwd)/model_weights:/app/model_weights deepfake-detector
```
The container will start the Flask app on port 5000.

## API Usage
```bash
curl -X POST -F "file=@path/to/image.jpg" http://localhost:5000/api/predict
```
Response (JSON):
```json
{
  "filename": "image.jpg",
  "label": "Deepfake",
  "confidence": 87.34
}
```

## Health Check
`GET /health` returns `{ "status": "ok" }` when the model is loaded, otherwise `{ "status": "model_not_loaded" }`.

## License
Feel free to modify and use this project for educational purposes.
