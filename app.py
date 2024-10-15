from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import requests
import os
import tempfile

app = Flask(__name__)

# INSERT MODEL BELOW
model = load_model('saved_model.h5')

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(32, 32))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.
    return img

def download_image(url):
    """Downloads the image from the provided URL and saves it temporarily."""
    response = requests.get(url)
    
    if response.status_code != 200:
        raise Exception('Failed to download image')
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(response.content)
        return tmp.name  # Return the temp file path

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    if 'url' not in data:
        return jsonify({'error': 'No URL provided'})

    image_url = data['url']

    try:
        img_path = download_image(image_url)
    except Exception as e:
        return jsonify({'error': str(e)})
    
    img = preprocess_image(img_path)
    prediction = model.predict(img)

    response = {
        'probabilities': prediction.tolist(),  # Convert numpy array to list
        'class_index': int(np.argmax(prediction)),  # Convert numpy int64 to Python int
        'confidence': float(np.max(prediction))  # Convert numpy float64 to Python float
    }

    # Clean up the temporary file
    os.remove(img_path)

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
