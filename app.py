import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "0"
os.environ["QT_QPA_PLATFORM"] = "offscreen"
# Batasi thread torch — kurangi memory footprint di Railway
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import numpy as np

app = Flask(__name__)
CORS(app)

# Load model saat startup (bukan lazy) — lebih predictable memory
print("Loading YOLO model...")
from ultralytics import YOLO
import torch
torch.set_num_threads(1)  # paksa single thread — hemat RAM
model = YOLO("best.pt")
print("Model loaded!")

@app.route('/detect', methods=['POST'])
def detect():
    if 'foto' not in request.files:
        return jsonify({'error': 'Tidak ada foto'}), 400

    file  = request.files['foto']
    image = Image.open(io.BytesIO(file.read())).convert('RGB')

    # Resize ke max 416x416 sebelum inference — kurangi RAM saat inference
    MAX_SIZE = 416
    if image.width > MAX_SIZE or image.height > MAX_SIZE:
        image.thumbnail((MAX_SIZE, MAX_SIZE), Image.LANCZOS)

    # imgsz eksplisit 320 — lebih ringan dari default 640
    results = model.predict(np.array(image), conf=0.4, imgsz=320, verbose=False)

    detections = []
    for result in results:
        for box in result.boxes:
            detections.append({
                'penyakit':   model.names[int(box.cls)],
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
