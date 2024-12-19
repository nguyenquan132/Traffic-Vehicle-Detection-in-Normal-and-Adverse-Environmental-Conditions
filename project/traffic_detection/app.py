from flask import Flask, request, render_template
import os
from utils.preprocess import check_darkness, enhance_image
from utils.predict import yolo_predict

import sys
import os

# Thêm thư mục gốc vào sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'

label_mapping = {
    0: "Xe máy",
    1: "Xe ô tô con",
    2: "Xe vận tải du lịch (xe khách)",
    3: "Xe vận tải container"
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Kiểm tra độ sáng
    is_dark, mean_brightness = check_darkness(filepath)
    if is_dark:
        enhanced_path = os.path.join(app.config['UPLOAD_FOLDER'], f"enhanced_{file.filename}")
        enhance_image(filepath, enhanced_path)
        filepath = enhanced_path

    # Dự đoán
    result_path, detection_data, message = yolo_predict(filepath, app.config['RESULT_FOLDER'], label_mapping)

    return render_template('index.html',
                           original_image=file.filename,
                           detected_image=os.path.basename(result_path) if result_path else None,
                           detection_data=detection_data,
                           message=message)

if __name__ == '__main__':
    app.run(debug=True)
