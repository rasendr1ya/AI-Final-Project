from flask import Flask, request, jsonify, render_template, url_for
import os
import cv2
import numpy as np
from deepface import DeepFace
from werkzeug.utils import secure_filename
import base64
from PIL import Image
import io

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Available models and their characteristics
AVAILABLE_MODELS = {
    'VGG-Face': {'accuracy': 'High', 'speed': 'Medium', 'description': 'Oxford VGG model, balanced accuracy and speed'},
    'Facenet': {'accuracy': 'Very High', 'speed': 'Medium', 'description': 'Google model, excellent accuracy'},
    'Facenet512': {'accuracy': 'Very High', 'speed': 'Slow', 'description': 'Facenet 512-dim, highest accuracy'},
    'OpenFace': {'accuracy': 'Medium', 'speed': 'Fast', 'description': 'Lightweight model, fast processing'},
    'DeepFace': {'accuracy': 'High', 'speed': 'Medium', 'description': 'Facebook original model'},
    'DeepID': {'accuracy': 'High', 'speed': 'Medium', 'description': 'DeepID model from CUHK'},
    'ArcFace': {'accuracy': 'Very High', 'speed': 'Medium', 'description': 'State-of-the-art accuracy'},
    'Dlib': {'accuracy': 'Medium', 'speed': 'Fast', 'description': 'Traditional computer vision'},
    'SFace': {'accuracy': 'High', 'speed': 'Fast', 'description': 'Recent OpenCV model'}
}

DISTANCE_METRICS = {
    'cosine': 'Cosine similarity (most common)',
    'euclidean': 'Euclidean distance (geometric)',
    'euclidean_l2': 'Normalized Euclidean distance'
}

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    """Preprocess image for face detection"""
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            return None
            
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img_rgb
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def image_to_base64(image_path):
    """Convert image to base64 for display"""
    try:
        with open(image_path, "rb") as img_file:
            img_data = img_file.read()
            base64_string = base64.b64encode(img_data).decode('utf-8')
            return base64_string
    except Exception as e:
        print(f"Error converting image to base64: {e}")
        return None

def calculate_similarity_percentage(distance, threshold, verified):
    """Calculate similarity percentage based on distance and threshold"""
    if verified:
        # If verified, similarity is high
        similarity = max(0, (1 - (distance / threshold)) * 100)
    else:
        # If not verified, calculate based on how close to threshold
        if distance <= threshold:
            similarity = 100
        else:
            # Scale down based on how far from threshold
            similarity = max(0, (1 - (distance - threshold) / threshold) * 50)
    
    return min(100, max(0, similarity))

@app.route('/')
def index():
    return render_template('ai.html')

@app.route('/models', methods=['GET'])
def get_models():
    """Get available models and distance metrics"""
    return jsonify({
        'models': AVAILABLE_MODELS,
        'distance_metrics': DISTANCE_METRICS
    })

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file selected'}), 400
            
        file = request.files['file']
        image_type = request.form.get('type')  # 'image1' or 'image2'
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
            
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # Add timestamp to avoid conflicts
            import time
            timestamp = str(int(time.time()))
            filename = f"{image_type}_{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Convert to base64 for preview
            base64_img = image_to_base64(filepath)
            
            return jsonify({
                'success': True,
                'filename': filename,
                'filepath': filepath,
                'image_data': f"data:image/jpeg;base64,{base64_img}"
            })
        else:
            return jsonify({'error': 'Invalid file type'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict_similarity():
    try:
        data = request.get_json()
        image1_path = data.get('image1_path')
        image2_path = data.get('image2_path')
        model_name = data.get('model_name', 'VGG-Face')  # Default to VGG-Face
        distance_metric = data.get('distance_metric', 'cosine')  # Default to cosine
        
        if not image1_path or not image2_path:
            return jsonify({'error': 'Both images are required'}), 400
            
        # Validate model and distance metric
        if model_name not in AVAILABLE_MODELS:
            return jsonify({'error': f'Invalid model: {model_name}'}), 400
            
        if distance_metric not in DISTANCE_METRICS:
            return jsonify({'error': f'Invalid distance metric: {distance_metric}'}), 400
            
        # Check if files exist
        if not os.path.exists(image1_path) or not os.path.exists(image2_path):
            return jsonify({'error': 'One or both image files not found'}), 400
            
        # Use DeepFace to verify similarity
        result = DeepFace.verify(
            img1_path=image1_path,
            img2_path=image2_path,
            model_name=model_name,
            distance_metric=distance_metric
        )
        
        # Calculate similarity percentage
        distance = result['distance']
        threshold = result['threshold']
        verified = result['verified']
        
        # Calculate similarity percentage with improved algorithm
        similarity_percentage = calculate_similarity_percentage(distance, threshold, verified)
        
        # Create detailed explanation
        explanation = {
            'distance_meaning': f'Distance: {distance:.4f} (lower = more similar)',
            'threshold_meaning': f'Threshold: {threshold:.4f} (boundary for same person)',
            'verification_meaning': f'Verified: {"YES - Same person" if verified else "NO - Different person"}',
            'similarity_meaning': f'Similarity: {similarity_percentage:.2f}% confident match',
            'model_info': AVAILABLE_MODELS[model_name],
            'metric_info': DISTANCE_METRICS[distance_metric]
        }
        
        return jsonify({
            'success': True,
            'verified': verified,
            'distance': distance,
            'threshold': threshold,
            'similarity_percentage': round(similarity_percentage, 2),
            'model': result['model'],
            'detector_backend': result['detector_backend'],
            'time_taken': result['time'],
            'explanation': explanation,
            'selected_model': model_name,
            'selected_metric': distance_metric
        })
        
    except Exception as e:
        error_msg = str(e)
        if "Face could not be detected" in error_msg:
            return jsonify({'error': 'Face could not be detected in one or both images. Please use clear face images.'}), 400
        else:
            return jsonify({'error': f'Prediction error: {error_msg}'}), 500

@app.route('/clear', methods=['POST'])
def clear_uploads():
    """Clear uploaded files"""
    try:
        for filename in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            if os.path.isfile(file_path):
                os.unlink(file_path)
        return jsonify({'success': True, 'message': 'All files cleared'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)