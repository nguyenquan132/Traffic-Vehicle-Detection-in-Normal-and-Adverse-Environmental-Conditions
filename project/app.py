from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file, Response
import os
import cv2
from ultralytics import YOLO
from flask_cors import CORS
import tempfile

app = Flask(__name__)
CORS(app)
# Load YOLO model
model = YOLO('project/weights/yolov9s_17_11.pt')  # Thay bằng đường dẫn đến file YOLO của bạn

@app.route('/')
def index():
    return render_template('traffic_vehicle.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    global uploaded_video_path  # Sử dụng biến toàn cục để lưu đường dẫn video

    if 'video' not in request.files:
        return 'No file part', 400

    file = request.files['video']
    if file.filename == '':
        return 'No selected file', 400

    if file:
        # Lưu video tạm thời trong bộ nhớ hoặc ổ đĩa
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
            temp_video.write(file.read())
            uploaded_video_path = temp_video.name

        return jsonify({'ok': True, 'message': 'Video uploaded successfully'})

@app.route('/video_stream', methods=['GET'])
def video_stream():
    if uploaded_video_path is None:
        return 'No video uploaded', 400

    def generate():
        cap = cv2.VideoCapture(uploaded_video_path)
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            # Chuyển đổi frame sang JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            # Truyền frame tới frontend
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        cap.release()

        # Xóa file video tạm sau khi stream xong
        os.remove(uploaded_video_path)

    return Response(generate(), content_type='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
