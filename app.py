from flask import Flask, request, send_file, render_template_string
import os
import numpy as np
import torch
import torch.nn as nn
import rasterio
from PIL import Image
import io
import base64
import segmentation_models_pytorch as smp

# Initialize Flask app
app = Flask(__name__)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parameters
NUM_CHANNELS = 12
BACKBONE = 'resnet34'
LABEL_DIR = 'data/labels'  # Directory for true masks

# Load pretrained model
def load_model():
    model = smp.Unet(
        encoder_name=BACKBONE,
        encoder_weights=None,  # Weights loaded from file
        in_channels=NUM_CHANNELS,
        classes=1,
        activation=None
    )
    model.load_state_dict(torch.load('pretrained_unet_model.pth', map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()

# Preprocess input image for inference
def preprocess_image(file_path):
    with rasterio.open(file_path) as src:
        image = src.read()  # shape: (12, H, W)
        image = np.moveaxis(image, 0, -1)  # (H, W, 12)
    # Band-wise standard scaling (same as notebook)
    image = image.astype(np.float32)
    for c in range(NUM_CHANNELS):
        mean = image[:, :, c].mean()
        std = image[:, :, c].std() + 1e-6  # Avoid division by zero
        image[:, :, c] = (image[:, :, c] - mean) / std
    # Convert to tensor
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()  # (C, H, W)
    image_tensor = image_tensor.unsqueeze(0)  # (1, C, H, W)
    return image_tensor, image

# Generate segmentation mask
def generate_mask(image):
    with torch.no_grad():
        image = image.to(device)
        pred = model(image)
        pred = torch.sigmoid(pred)  # Convert logits to probabilities
        pred = (pred > 0.5).float()  # Binary mask
    return pred.squeeze(0).squeeze(0).cpu().numpy()  # (1, H, W) -> (H, W)

# Convert images to base64 and buffer
def images_to_output(uploaded_image, pred_mask, true_mask=None):
    # Predicted mask to PNG
    pred_mask = (pred_mask * 255).astype(np.uint8)
    pred_image = Image.fromarray(pred_mask, mode='L')
    pred_buffer = io.BytesIO()
    pred_image.save(pred_buffer, format='PNG')
    pred_buffer.seek(0)
    pred_base64 = base64.b64encode(pred_buffer.getvalue()).decode('utf-8')

    # Uploaded image to RGB (using bands 4, 3, 2)
    rgb_image = uploaded_image[:, :, [3, 2, 1]]  # Bands 4, 3, 2 for RGB
    rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min() + 1e-6)  # Normalize to [0,1]
    rgb_image = (rgb_image * 255).astype(np.uint8)
    rgb_image = Image.fromarray(rgb_image, mode='RGB')
    rgb_buffer = io.BytesIO()
    rgb_image.save(rgb_buffer, format='PNG')
    rgb_buffer.seek(0)
    rgb_base64 = base64.b64encode(rgb_buffer.getvalue()).decode('utf-8')

    # True mask to PNG (if provided)
    true_base64 = None
    if true_mask is not None:
        true_mask = (true_mask > 0).astype(np.uint8) * 255  # Binarize and scale to 0 or 255
        true_image = Image.fromarray(true_mask, mode='L')
        true_buffer = io.BytesIO()
        true_image.save(true_buffer, format='PNG')
        true_buffer.seek(0)
        true_base64 = base64.b64encode(true_buffer.getvalue()).decode('utf-8')

    return pred_buffer, pred_base64, rgb_base64, true_base64

# HTML template for upload page
@app.route('/')
def index():
    return render_template_string('''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Water Segmentation</title>
            <script src="https://cdn.tailwindcss.com"></script>
        </head>
        <body class="bg-gray-100 min-h-screen flex items-center justify-center">
            <div class="bg-white p-8 rounded-lg shadow-lg max-w-md w-full">
                <h1 class="text-2xl font-bold text-center mb-6">Water Segmentation</h1>
                <form method="post" enctype="multipart/form-data" action="/predict" class="space-y-4">
                    <div>
                        <label class="block text-sm font-medium text-gray-700">Upload Multispectral Image (.tif)</label>
                        <input type="file" name="image" accept=".tif" class="mt-1 block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100">
                    </div>
                    <button type="submit" class="w-full bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700">Segment</button>
                </form>
            </div>
        </body>
        </html>
    ''')

# Endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return 'No image uploaded', 400
    image_file = request.files['image']
    if image_file.filename == '':
        return 'No image selected', 400
    if not image_file.filename.endswith('.tif'):
        return 'Invalid image format. Please upload a .tif file', 400

    # Save uploaded image temporarily
    upload_dir = 'Uploads'
    os.makedirs(upload_dir, exist_ok=True)
    image_filename = image_file.filename
    image_path = os.path.join(upload_dir, image_filename)
    image_file.save(image_path)

    # Fetch true mask from labels directory
    true_mask = None
    true_mask_filename = image_filename.replace('.tif', '.png')
    true_mask_path = os.path.join(LABEL_DIR, true_mask_filename)
    if os.path.exists(true_mask_path):
        true_mask = np.array(Image.open(true_mask_path).convert('L'))  # Load as grayscale
        true_mask = (true_mask > 0).astype(np.float32)  # Binarize
    else:
        true_mask_filename = None  # No true mask available

    try:
        # Preprocess and predict
        image_tensor, uploaded_image = preprocess_image(image_path)
        pred_mask = generate_mask(image_tensor)
        pred_buffer, pred_base64, rgb_base64, true_base64 = images_to_output(uploaded_image, pred_mask, true_mask)

        # Render HTML with images
        return render_template_string('''
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Segmentation Result</title>
                <script src="https://cdn.tailwindcss.com"></script>
            </head>
            <body class="bg-gray-100 min-h-screen flex items-center justify-center">
                <div class="bg-white p-8 rounded-lg shadow-lg max-w-4xl w-full">
                    <h1 class="text-2xl font-bold text-center mb-6">Segmentation Result</h1>
                    <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                        <div class="text-center">
                            <h2 class="text-lg font-semibold mb-2">Uploaded Image (RGB)</h2>
                            <img src="data:image/png;base64,{{ rgb_base64 }}" alt="Uploaded Image" class="w-full rounded-md shadow">
                        </div>
                        <div class="text-center">
                            <h2 class="text-lg font-semibold mb-2">Predicted Mask</h2>
                            <img src="data:image/png;base64,{{ pred_base64 }}" alt="Predicted Mask" class="w-full rounded-md shadow">
                        </div>
                        {% if true_base64 %}
                        <div class="text-center">
                            <h2 class="text-lg font-semibold mb-2">True Mask</h2>
                            <img src="data:image/png;base64,{{ true_base64 }}" alt="True Mask" class="w-full rounded-md shadow">
                        </div>
                        {% else %}
                        <div class="text-center">
                            <h2 class="text-lg font-semibold mb-2">True Mask</h2>
                            <p class="text-gray-500">No true mask found in {{ label_dir }}</p>
                        </div>
                        {% endif %}
                    </div>
                    <div class="flex justify-center space-x-4">
                        <a href="/" class="bg-gray-600 text-white py-2 px-4 rounded-md hover:bg-gray-700">Upload Another Image</a>
                    </div>
                </div>
            </body>
            </html>
        ''', rgb_base64=rgb_base64, pred_base64=pred_base64, true_base64=true_base64, label_dir=LABEL_DIR)
    except Exception as e:
        os.remove(image_path)
        return f'Error processing image: {str(e)}', 500


if __name__ == '__main__':
    app.run(debug=True)