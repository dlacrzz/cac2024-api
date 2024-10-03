from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import tempfile

app = Flask(__name__)

# INSERT MODEL BELOW
model = load_model('models/saved_model.h5')

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(32, 32))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.
    return img

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']

    # Use tempfile to handle the temporary directory
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        file.save(tmp.name)
        img_path = tmp.name

    img = preprocess_image(img_path)
    prediction = model.predict(img)

    response = {
        'probabilities': prediction.tolist(),  # Convert numpy array to list
        'class_index': int(np.argmax(prediction)),  # Convert numpy int64 to Python int
        'confidence': float(np.max(prediction))  # Convert numpy float64 to Python float
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)