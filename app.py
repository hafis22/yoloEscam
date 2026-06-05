from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

# Load model
print("Loading YOLO model...")
model = YOLO("best.pt")
print("Model loaded!")

@app.route('/detect', methods=['POST'])
def detect():
    if 'foto' not in request.files:
        return jsonify({'error': 'Tidak ada foto'}), 400

    file  = request.files['foto']
    image = Image.open(io.BytesIO(file.read())).convert('RGB')

    # Jalankan deteksi
    results = model.predict(image, conf=0.5)

    detections = []
    for result in results:
        for box in result.boxes:
            detections.append({
                'penyakit':    model.names[int(box.cls)],
                'confidence':  round(float(box.conf) * 100, 1),
                'bbox': {
                    'x1': round(float(box.xyxy[0][0])),
                    'y1': round(float(box.xyxy[0][1])),
                    'x2': round(float(box.xyxy[0][2])),
                    'y2': round(float(box.xyxy[0][3])),
                }
            })

    # Kalau tidak ada deteksi
    if not detections:
        return jsonify({
            'penyakit':   'Tidak terdeteksi',
            'confidence': 0,
            'detections': []
        })

    # Ambil deteksi dengan confidence tertinggi
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
    app.run(host='0.0.0.0', port=8000, debug=False)