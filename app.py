from flask import Flask, request, jsonify, render_template
from PIL import Image
import io
from transformers import AutoImageProcessor, AutoModelForImageClassification
import requests

app = Flask(__name__)

# Load the model and processor
processor = AutoImageProcessor.from_pretrained("yusuf802/Leaf-Disease-Predictor")
model = AutoModelForImageClassification.from_pretrained("yusuf802/Leaf-Disease-Predictor")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file:
        image = Image.open(file.stream)
        inputs = processor(image, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        label = model.config.id2label[predicted_class_idx]
        return jsonify({'label': label})

@app.route('/predict_url', methods=['POST'])
def predict_url():
    url = request.json.get('url')
    response = requests.get(url)
    image = Image.open(io.BytesIO(response.content))
    inputs = processor(image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    label = model.config.id2label[predicted_class_idx]
    return jsonify({'label': label})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
