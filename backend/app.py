import os
import yaml
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from ultralytics import YOLO

# Tentukan path absolut ke folder frontend (satu level di atas folder backend)
FRONTEND_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'frontend')

app = Flask(__name__, static_folder=FRONTEND_FOLDER, static_url_path='')
CORS(app)  # Enable CORS for all routes

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Route untuk halaman utama (serve index.html dari folder frontend)
@app.route('/')
def index():
    return app.send_static_file('index.html')

# Load YOLOv8 model
# Akan mencoba meload 'models/best.pt' terlebih dahulu (model buatan sendiri).
# Jika belum ada, otomatis menggunakan model bawaan YOLOv8s untuk testing API.
MODEL_PATH = os.path.join('models', 'best.pt')
os.makedirs('models', exist_ok=True)

# Variabel global untuk menyimpan metrik model
model_metrics = {
    'precision': 0,
    'recall': 0,
    'f1_score': 0,
    'map50': 0
}

try:
    if os.path.exists(MODEL_PATH):
        model = YOLO(MODEL_PATH)
        print(f"Loaded custom model from {MODEL_PATH}")
    else:
        print("Custom model not found. Loading default YOLOv8s model for testing...")
        model = YOLO('yolov8s.pt')
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# --- EVALUASI MODEL SAAT STARTUP (Hitung Precision, Recall, F1) ---
DATASET_YAML = os.path.join('dataset', 'master_dataset_v2', 'data.yaml')
if model is not None and os.path.exists(DATASET_YAML):
    try:
        # Pastikan path dataset di data.yaml sudah benar
        dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset', 'master_dataset_v2')
        with open(DATASET_YAML, 'r') as f:
            data = yaml.safe_load(f)
        data['path'] = dataset_dir
        with open(DATASET_YAML, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)

        print("\n[INFO] Menghitung metrik model (Precision, Recall, F1)...")
        print("[INFO] Mohon tunggu sebentar, proses validasi sedang berjalan...\n")
        metrics = model.val(data=DATASET_YAML, split='val', verbose=False)
        
        p = metrics.box.mp
        r = metrics.box.mr
        f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
        
        model_metrics['precision'] = p
        model_metrics['recall'] = r
        model_metrics['f1_score'] = f1
        model_metrics['map50'] = metrics.box.map50
        
        print("\n" + "=" * 50)
        print("   METRIK MODEL BERHASIL DIHITUNG (VALIDATION SET)")
        print("=" * 50)
        print(f"Precision  : {p * 100:.2f}%")
        print(f"Recall     : {r * 100:.2f}%")
        print(f"F1-Score   : {f1 * 100:.2f}%")
        print(f"mAP@50     : {metrics.box.map50 * 100:.2f}%")
        print("=" * 50)
        print("[INFO] Server siap menerima request deteksi!\n")
    except Exception as e:
        print(f"[WARNING] Gagal menghitung metrik: {e}")
        print("[INFO] Server tetap berjalan tanpa metrik.\n")
else:
    print("[INFO] Dataset tidak ditemukan, metrik tidak dihitung.")

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Warna bounding box per kelas (BGR format untuk OpenCV)
CLASS_COLORS = {
    'lele': (0, 255, 100),    # Hijau terang
    'nila': (255, 180, 0),    # Biru-cyan
}
DEFAULT_COLOR = (0, 255, 255)  # Kuning (fallback)

def draw_bounding_boxes(image_path, detections, output_path):
    """Gambar bounding box pada gambar dan simpan hasilnya."""
    img = cv2.imread(image_path)
    if img is None:
        return False

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det['bbox']]
        cls_name = det['class']
        conf = det['confidence']
        color = CLASS_COLORS.get(cls_name.lower(), DEFAULT_COLOR)

        # Gambar kotak bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Label: class + confidence
        label = f"{cls_name} {conf:.0%}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        # Background label agar teks terbaca
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

    cv2.imwrite(output_path, img)
    return True

# Route untuk serve gambar hasil deteksi
@app.route('/uploads/<filename>')
def serve_upload(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/detect', methods=['POST'])
def detect_objects():
    if model is None:
        return jsonify({'error': 'Model failed to load on server.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file part in request'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        ext = file.filename.rsplit('.', 1)[1].lower()
        filename = secure_filename(file.filename)
        if not filename or '.' not in filename:
            filename = f"upload.{ext}"

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        if ext in {'png', 'jpg', 'jpeg'}:
            results = model(filepath, conf=0.40, iou=0.70, imgsz=640, max_det=3000)
            
            detections = []
            counts = {}
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    cls_name = model.names[cls_id]
                    
                    if cls_name not in counts:
                        counts[cls_name] = 0
                    counts[cls_name] += 1
                    
                    detections.append({
                        'class': cls_name,
                        'confidence': round(conf, 2),
                        'bbox': box.xyxy[0].tolist() # [x1, y1, x2, y2]
                    })
            
            # Gambar bounding box pada gambar dan simpan hasilnya
            result_filename = f"result_{filename}"
            result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
            drawn = draw_bounding_boxes(filepath, detections, result_path)
            # Jika gagal menggambar (mis. gambar tidak bisa dibaca cv2), file hasil
            # tidak dibuat -> jangan tunjuk ke file yang tidak ada, pakai gambar asli.
            result_image_url = f'/uploads/{result_filename}' if drawn else f'/uploads/{filename}'
            
            total_ikan = sum(counts.values())
            
            # --- CETAK LAPORAN KE TERMINAL ---
            if detections:
                all_confs = [d['confidence'] for d in detections]
                avg_conf = sum(all_confs) / len(all_confs)
                high_conf_count = sum(1 for c in all_confs if c >= 0.5)
            else:
                avg_conf = 0
                high_conf_count = 0
            
            print("\n" + "="*50)
            print("        LAPORAN DETEKSI BENIH IKAN")
            print("="*50)
            print(f"File Diproses   : {filename}")
            print(f"Total Ikan      : {total_ikan} ekor")
            print("Rincian per Jenis:")
            if counts:
                for jenis, jumlah in counts.items():
                    print(f"  - {jenis.capitalize():<10}: {jumlah} ekor")
            else:
                print("  - Tidak ada ikan yang terdeteksi.")
            print("-"*50)
            print("STATISTIK CONFIDENCE DETEKSI:")
            print(f"  Rata-rata Confidence   : {avg_conf * 100:.2f}%")
            print(f"  Deteksi Keyakinan Tinggi (>=50%) : {high_conf_count} dari {len(detections)}")
            print("="*50 + "\n")
            # ----------------------------------
            
            return jsonify({
                'success': True,
                'counts': counts,
                'total_detected': total_ikan,
                'detections': detections,
                # URL gambar hasil deteksi (dengan bounding box)
                'result_image': result_image_url,
                # Statistik confidence untuk gambar ini
                'image_stats': {
                    'avg_confidence': round(avg_conf * 100, 2),
                    'high_conf_count': high_conf_count,
                    'total_detections': len(detections),
                }
            })
            
    return jsonify({'error': 'Invalid file type. Please upload an image.'}), 400

if __name__ == '__main__':
    import webbrowser
    import threading
    # Buka browser otomatis setelah server siap (delay 1.5 detik)
    def open_browser():
        import time
        time.sleep(1.5)
        webbrowser.open('http://localhost:5000')
    threading.Thread(target=open_browser, daemon=True).start()
    print("\n" + "="*50)
    print("  SERVER SIAP! Buka browser di: http://localhost:5000")
    print("  Tekan CTRL+C untuk menghentikan server.")
    print("="*50 + "\n")
    app.run(debug=False, port=5000)
