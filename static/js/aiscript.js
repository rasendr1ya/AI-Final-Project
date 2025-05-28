let uploadedFiles = {
    image1: null,
    image2: null
};

let modelsData = {};

document.addEventListener('DOMContentLoaded', function() {
    // Load available models and metrics
    loadModelsData();
    
    // Button event listeners
    document.getElementById('load-btn1').addEventListener('click', () => document.getElementById('file1').click());
    document.getElementById('load-btn2').addEventListener('click', () => document.getElementById('file2').click());
    document.getElementById('predict-btn').addEventListener('click', predictSimilarity);
    document.getElementById('clear-btn').addEventListener('click', clearAll);

    // File input event listeners
    document.getElementById('file1').addEventListener('change', (e) => handleFileUpload(e, 'image1'));
    document.getElementById('file2').addEventListener('change', (e) => handleFileUpload(e, 'image2'));

    // Model selection event listeners
    document.getElementById('model-select').addEventListener('change', updateModelDescription);
    document.getElementById('metric-select').addEventListener('change', updateMetricDescription);

    // Drag and drop functionality
    setupDragAndDrop('upload1', 'image1');
    setupDragAndDrop('upload2', 'image2');
});

async function loadModelsData() {
    try {
        const response = await fetch('/models');
        modelsData = await response.json();
    } catch (error) {
        console.error('Failed to load models data:', error);
    }
}

function updateModelDescription() {
    const selectedModel = document.getElementById('model-select').value;
    const descriptionElement = document.getElementById('model-description');
    
    if (modelsData.models && modelsData.models[selectedModel]) {
        const modelInfo = modelsData.models[selectedModel];
        descriptionElement.textContent = `${modelInfo.description} (Accuracy: ${modelInfo.accuracy}, Speed: ${modelInfo.speed})`;
    }
}

function updateMetricDescription() {
    const selectedMetric = document.getElementById('metric-select').value;
    const descriptionElement = document.getElementById('metric-description');
    
    if (modelsData.distance_metrics && modelsData.distance_metrics[selectedMetric]) {
        descriptionElement.textContent = modelsData.distance_metrics[selectedMetric];
    }
}

function setupDragAndDrop(uploadAreaId, imageType) {
    const uploadArea = document.getElementById(uploadAreaId);
    
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });
    
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            const fileInput = document.getElementById(imageType === 'image1' ? 'file1' : 'file2');
            fileInput.files = files;
            handleFileUpload({ target: fileInput }, imageType);
        }
    });
    
    uploadArea.addEventListener('click', () => {
        const fileInput = document.getElementById(imageType === 'image1' ? 'file1' : 'file2');
        fileInput.click();
    });
}

async function handleFileUpload(event, imageType) {
    const file = event.target.files[0];
    if (!file) return;

    // Validate file type
    const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/bmp'];
    if (!allowedTypes.includes(file.type)) {
        showError('Please select a valid image file (JPEG, PNG, GIF, BMP)');
        return;
    }

    // Validate file size (max 16MB)
    if (file.size > 16 * 1024 * 1024) {
        showError('File size must be less than 16MB');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);
    formData.append('type', imageType);

    try {
        showLoading(true);
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();
        
        if (result.success) {
            uploadedFiles[imageType] = result;
            displayPreview(imageType, result.image_data);
            updatePredictButton();
        } else {
            showError(result.error);
        }
    } catch (error) {
        showError('Upload failed: ' + error.message);
    } finally {
        showLoading(false);
    }
}

function displayPreview(imageType, imageData) {
    const uploadArea = document.getElementById(imageType === 'image1' ? 'upload1' : 'upload2');
    const placeholder = uploadArea.querySelector('.upload-placeholder');
    const preview = uploadArea.querySelector('img');
    
    placeholder.style.display = 'none';
    preview.src = imageData;
    preview.style.display = 'block';
}

function updatePredictButton() {
    const predictBtn = document.getElementById('predict-btn');
    predictBtn.disabled = !(uploadedFiles.image1 && uploadedFiles.image2);
}

async function predictSimilarity() {
    if (!uploadedFiles.image1 || !uploadedFiles.image2) {
        showError('Please upload both images first');
        return;
    }

    // Get selected model and metric
    const selectedModel = document.getElementById('model-select').value;
    const selectedMetric = document.getElementById('metric-select').value;

    try {
        showLoading(true);
        hideError();
        hideResult();

        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                image1_path: uploadedFiles.image1.filepath,
                image2_path: uploadedFiles.image2.filepath,
                model_name: selectedModel,
                distance_metric: selectedMetric
            })
        });

        const result = await response.json();
        
        if (result.success) {
            displayResult(result);
        } else {
            showError(result.error);
        }
    } catch (error) {
        showError('Prediction failed: ' + error.message);
    } finally {
        showLoading(false);
    }
}

function displayResult(result) {
    // Update similarity score
    document.getElementById('similarity-percentage').textContent = result.similarity_percentage + '%';
    
    // Update basic details
    document.getElementById('verified-status').textContent = result.verified ? 'Yes' : 'No';
    document.getElementById('distance-value').textContent = result.distance.toFixed(4);
    document.getElementById('threshold-value').textContent = result.threshold.toFixed(4);
    document.getElementById('model-name').textContent = result.selected_model;
    document.getElementById('distance-metric').textContent = result.selected_metric;
    document.getElementById('processing-time').textContent = result.time_taken.toFixed(2) + 's';
    
    // Update progress bar
    const progressFill = document.getElementById('progress-fill');
    progressFill.style.width = result.similarity_percentage + '%';
    
    // Update similarity score color and status
    const scoreElement = document.getElementById('similarity-percentage');
    const statusElement = document.getElementById('similarity-status');
    
    if (result.verified) {
        scoreElement.style.color = '#2ecc71';
        statusElement.className = 'similarity-status verified';
        statusElement.innerHTML = '<i class="fas fa-check-circle"></i><span>Same Person</span>';
    } else {
        if (result.similarity_percentage >= 50) {
            scoreElement.style.color = '#f39c12';
            statusElement.className = 'similarity-status not-verified';
            statusElement.innerHTML = '<i class="fas fa-exclamation-triangle"></i><span>Possibly Same</span>';
        } else {
            scoreElement.style.color = '#e74c3c';
            statusElement.className = 'similarity-status not-verified';
            statusElement.innerHTML = '<i class="fas fa-times-circle"></i><span>Different Person</span>';
        }
    }
    
    // Update detailed explanations
    if (result.explanation) {
        document.getElementById('distance-meaning').textContent = result.explanation.distance_meaning;
        document.getElementById('threshold-meaning').textContent = result.explanation.threshold_meaning;
        document.getElementById('verification-meaning').textContent = result.explanation.verification_meaning;
        document.getElementById('similarity-meaning').textContent = result.explanation.similarity_meaning;
    }
    
    document.getElementById('result').style.display = 'block';
}

async function clearAll() {
    try {
        // Clear uploaded files from server
        await fetch('/clear', { method: 'POST' });
        
        // Reset UI
        uploadedFiles = { image1: null, image2: null };
        
        // Reset upload areas
        resetUploadArea('upload1');
        resetUploadArea('upload2');
        
        // Reset file inputs
        document.getElementById('file1').value = '';
        document.getElementById('file2').value = '';
        
        // Hide result and error
        hideResult();
        hideError();
        
        // Disable predict button
        updatePredictButton();
        
    } catch (error) {
        showError('Failed to clear files: ' + error.message);
    }
}

function resetUploadArea(uploadAreaId) {
    const uploadArea = document.getElementById(uploadAreaId);
    const placeholder = uploadArea.querySelector('.upload-placeholder');
    const preview = uploadArea.querySelector('img');
    
    placeholder.style.display = 'flex';
    preview.style.display = 'none';
    preview.src = '';
}

function showLoading(show) {
    document.getElementById('loading').style.display = show ? 'block' : 'none';
}

function showError(message) {
    const errorElement = document.getElementById('error');
    errorElement.textContent = message;
    errorElement.style.display = 'block';
}

function hideError() {
    document.getElementById('error').style.display = 'none';
}

function hideResult() {
    document.getElementById('result').style.display = 'none';
}