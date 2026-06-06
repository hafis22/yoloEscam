import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "0"
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import numpy as np

app = Flask(__name__)
CORS(app)

# Load model lazy (hindari import cv2 di top-level)
model = None

def get_model():
    global model
    if model is None:
        from ultralytics import YOLO
        print("Loading YOLO model...")
        model = YOLO("best.pt")
        print("Model loaded!")
    return model

@app.route('/detect', methods=['POST'])
def detect():
    if 'foto' not in request.files:
        return jsonify({'error': 'Tidak ada foto'}), 400

    file  = request.files['foto']
    image = Image.open(io.BytesIO(file.read())).convert('RGB')

    m = get_model()
    results = m.predict(np.array(image), conf=0.5)

    detections = []
    for result in results:
        for box in result.boxes:
            detections.append({
                'penyakit':   m.names[int(box.cls)],
                'confidence': round(float(box.conf) * 100, 1),
                'bbox': {
                    'x1': round(float(box.xyxy[0][0])),
                    'y1': round(float(box.xyxy[0][1])),
                    'x2': round(float(box.xyxy[0][2])),
                    'y2': round(float(box.xyxy[0][3])),
                }
            })

    if not detections:
        return jsonify({
            'penyakit':   'Tidak terdeteksi',
            'confidence': 0,
            'detections': []
        })

    best = max(detections, key=lambda x: x['confidence'])
    return jsonify({
        'penyakit':   best['penyakit'],
        'confidence': best['confidence'],
        'detections': detections
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model': 'vanili-disease'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
