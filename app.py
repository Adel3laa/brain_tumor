from flask import Flask, request, render_template, jsonify
import os
import cv2
import joblib
import numpy as np

# Load the saved scalers and model
scaler = joblib.load('scaler.pkl')
pca = joblib.load('pca.pkl')
model = joblib.load('model.pkl')

def predict_tumor_type(image_path):

    # Load and preprocess the image
    img = cv2.imread(image_path, 0)  # Read in grayscale mode
    
    if img is None:
        raise ValueError(f"Error loading image from path: {image_path}")
    
    img = cv2.resize(img, (200, 200))  # Resize the image
    img = np.array(img).flatten().reshape(1, -1)  # Flatten the image
    
    # Normalize the image
    img_scaled = scaler.transform(img)
    
    # Apply PCA
    img_pca = pca.transform(img_scaled)
    
    # Predict using the trained model
    prediction = model.predict(img_pca)
    
    return prediction[0]

# Flask API for deployment
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('index.html', error='No file part in the request')
        
        file = request.files['image']
        if file.filename == '':
            return render_template('index.html', error='No selected file')
        
        file_path = os.path.join('uploads', file.filename)
        
        # Ensure the uploads directory exists
        if not os.path.exists('uploads'):
            os.makedirs('uploads')
        
        file.save(file_path)
        
        try:
            prediction = predict_tumor_type(file_path)
            return render_template('result.html', prediction=prediction)
        except ValueError as e:
            return render_template('index.html', error=str(e))
        except Exception as e:
            return render_template('index.html', error='An error occurred during prediction')
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
