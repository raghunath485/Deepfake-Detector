import os
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify

from werkzeug.utils import secure_filename
import numpy as np
import cv2

from model import load_model, predict_image
from utils import preprocess_image, generate_ela_heatmap

# Configuration
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'replace-with-a-secure-secret-key')

# Enable CORS for API usage
from flask_cors import CORS
CORS(app)

# Logging setup
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model once at startup
try:
    deepfake_model = load_model()
    logger.info("Deepfake model loaded successfully.")
except Exception as e:
    deepfake_model = None
    logger.error(f"Failed to load model: {e}")


import json

def get_metrics():
    metrics_path = os.path.join(os.path.dirname(__file__), 'model_weights', 'metrics.json')
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r') as f:
                return json.load(f)
        except Exception:
            return None
    return None

# API endpoint for programmatic use
@app.route('/api/predict', methods=['POST'])
def api_predict():
    if deepfake_model is None:
        logger.error('Model not loaded – API request rejected')
        return jsonify({'error': 'Model not loaded'}), 503
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid or missing file'}), 400
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    try:
        ela_filename = f"ela_{filename}"
        ela_filepath = os.path.join(app.config['UPLOAD_FOLDER'], ela_filename)
        generate_ela_heatmap(filepath, ela_filepath)
        label, confidence, findings, fake_prob, real_prob, pillars = deepfake_model.predict_image_path(filepath)
    except Exception as exc:
        logger.exception('Prediction error')
        return jsonify({'error': str(exc)}), 500
    return jsonify({
        'filename': filename,
        'label': label,
        'confidence': confidence,
        'fake_probability': fake_prob,
        'real_probability': real_prob,
        'findings': findings,
        'pillars': pillars,
        'ela_heatmap': url_for('static', filename='uploads/' + ela_filename)
    })

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if deepfake_model is None:
            flash('Model not loaded. Please ensure the model file is placed correctly.')
            return render_template('error.html')
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If user does not select file, browser may submit an empty part
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Generate Explainable AI ELA Heatmap
            ela_filename = f"ela_{filename}"
            ela_filepath = os.path.join(app.config['UPLOAD_FOLDER'], ela_filename)
            generate_ela_heatmap(filepath, ela_filepath)
            
            # Run comprehensive forensic analysis
            label, confidence, findings, fake_prob, real_prob, pillars = deepfake_model.predict_image_path(filepath)
            return render_template(
                'result.html',
                filename=filename,
                ela_filename=ela_filename,
                label=label,
                confidence=confidence,
                fake_prob=fake_prob,
                real_prob=real_prob,
                findings=findings,
                pillars=pillars,
                metrics=get_metrics()
            )
        else:
            flash('Unsupported file type')
            return redirect(request.url)
    return render_template('index.html', metrics=get_metrics())

# Health check endpoint
@app.route('/health', methods=['GET'])
def health():
    status = 'ok' if deepfake_model is not None else 'model_not_loaded'
    return jsonify({'status': status})

def uploaded_file(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == '__main__':
    # Run on all interfaces so it can be reached from outside the sandbox
    app.run(host='0.0.0.0', port=5000, debug=True)
