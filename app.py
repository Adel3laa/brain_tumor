from flask import Flask, request, render_template, jsonify
import os
import cv2
import joblib
import numpy as np
import json

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
    
    result = {
        "prediction": prediction[0]
    }
    
    return result

def save_result_to_json(result, filename):
    with open(filename, 'w') as json_file:
        json.dump(result, json_file)

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
            result = predict_tumor_type(file_path)
            save_result_to_json(result, 'result.json')
            return render_template('result.html', prediction=result["prediction"])
        except ValueError as e:
            return render_template('index.html', error=str(e))
        except Exception as e:
            return render_template('index.html', error='An error occurred during prediction')
    
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    if 'image' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    file_path = os.path.join('uploads', file.filename)
    
    # Ensure the uploads directory exists
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    
    file.save(file_path)
    
    try:
        result = predict_tumor_type(file_path)
        save_result_to_json(result, 'result.json')
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": "An error occurred during prediction"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
