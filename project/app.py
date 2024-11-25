from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
from ultralytics import YOLO

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load YOLO model
model = YOLO('weights/best.pt')  # Thay bằng đường dẫn đến file YOLO của bạn

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return 'No file part', 400

    file = request.files['video']
    if file.filename == '':
        return 'No selected file', 400

    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        return redirect(url_for('process_video', filename=file.filename))
    return 'Upload failed', 500

@app.route('/process/<filename>')
def process_video(filename):
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], f"output_{filename}")

    # Process video
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), 
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)  # Detect with YOLO
        for r in results:
            for box in r.boxes.xyxy:  # Duyệt qua các bounding box
                x1, y1, x2, y2 = map(int, box[:4])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        out.write(frame)

    cap.release()
    out.release()
    return redirect(url_for('display_video', filename=f"output_{filename}"))

@app.route('/video/<filename>')
def display_video(filename):
    video_url = url_for('static', filename=f"uploads/{filename}")
    return render_template('video.html', video_url=video_url)

if __name__ == '__main__':
    app.run(debug=True)
