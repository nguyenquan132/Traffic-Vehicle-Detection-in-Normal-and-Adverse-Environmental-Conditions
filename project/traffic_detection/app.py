import eventlet
eventlet.monkey_patch()
from flask import Flask, request, render_template, jsonify
import os
import json
from flask_socketio import SocketIO
from flask_cors import CORS
import time, base64
from threading import Event
from ultralytics import YOLO

import sys
import cv2
import supervision as sv

def load_yolo():
    """
    Load YOLO model.
    """
    return YOLO("weights/yolov9s_18_11.pt")

# Thêm thư mục gốc vào sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
model = load_yolo()

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'
socketio = SocketIO(app)

class PredictionState:
    def __init__(self):
        self.is_predicting = Event()
        self.is_predicting.set()  # Bắt đầu ở trạng thái hoạt động

prediction_state = PredictionState()

label_mapping = {
    0: "motorbike",
    1: "car",
    2: "coach",
    3: "container"
}
label_vehicle = {
    0: "Xe máy",
    1: "Xe ô tô",
    2: "Xe khách",
    3: "Xe chở hàng container"
}
filename = ''

@app.route('/')
def home():
    return render_template('traffic.html')

# Hàm để dừng predict
def stop_prediction():
    prediction_state.is_predicting.clear()

def resume_prediction():
    prediction_state.is_predicting.set()

def predict_video_stream(filepath):
    global box_size, old_frame
    cap = cv2.VideoCapture(filepath)
    fps = cap.get(cv2.CAP_PROP_FPS)
    vehicle_counters = {
        "motorbike": set(),
        "car": set(),
        "coach": set(),
        "container": set()
    }
    delay = 1 / fps
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=1, text_thickness=2, text_position=sv.Position.TOP_CENTER)
    trace_annotator = sv.TraceAnnotator(position=sv.Position.BOTTOM_CENTER, color_lookup=sv.ColorLookup.TRACK)
    tracker = sv.ByteTrack(frame_rate=fps)

    while cap.isOpened():
        # Kiểm tra trạng thái
        prediction_state.is_predicting.wait() # Đợi đến khi trạng thái là "resume"

        success, image = cap.read()
        detection_data = []

        if not success:
            break
        results = model.predict(image, save=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = tracker.update_with_detections(detections)

        # Đếm số xe
        for class_id, tracker_id in zip(detections.class_id, detections.tracker_id):
            vehicle_type = label_mapping[int(class_id)]
            if tracker_id not in vehicle_counters[vehicle_type]:
                vehicle_counters[vehicle_type].add(int(tracker_id))

        # Tạo counter dictionary với số lượng của mỗi loại xe
        counter = {
            vehicle_type: len(track_ids) 
            for vehicle_type, track_ids in vehicle_counters.items()
        }

        box_size = len(detections.xyxy)
        for i in range(len(detections.xyxy)):
            detection_data.append({
                "label": label_vehicle[int(detections.class_id[i])],
                "x": float(detections.xyxy[i][0]),
                "y": float(detections.xyxy[i][1]),
                "w": float(detections.xyxy[i][2]),
                "h": float(detections.xyxy[i][3]),
                "confidence": float(detections.confidence[i])
            })

        labels = [f"#{tracker_id} {label_mapping[class_id]}" for _, _, conf, class_id, tracker_id, *_ in detections]

        annotated_frame = trace_annotator.annotate(
            scene=image.copy(), detections=detections)
        annotated_frame = box_annotator.annotate(
            scene=annotated_frame, detections=detections)
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame, detections=detections, labels=labels)
        
        # Encode frame thành JPEG
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = base64.b64encode(buffer).decode('utf-8')

        # Lưu frame trước khi dừng 
        old_frame = frame 
        # Emit frame and detection_data
        socketio.emit('video_stream', {'frame': frame, 'detection_data': json.dumps(detection_data, ensure_ascii=False),
                                       'counter': json.dumps(counter, ensure_ascii=False)})

        time.sleep(delay)
    
    cap.release()

@app.route('/upload', methods=['POST'])
def upload_file():
    global filename
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    filename = filepath

    return jsonify({'ok': True, 
                    'message': 'Video uploaded successfully'})

@socketio.on('start_prediction')
def handle_start_prediction(data):
    print(data)
    resume_prediction() # Đảm bảo trạng thái được kích hoạt
    if filename:
        predict_video_stream(filename)

@socketio.on('stop_prediction')
def handle_stop_prediction(data):
    print(data)
    stop_prediction() # Chuyển trạng thái dừng

@socketio.on('resume_prediction')
def handle_resume_prediction(data):
    print(data)
    resume_prediction() # Chuyển trạng thái tiếp tục

@socketio.on('capture_prediction')
def handle_capture_prediction(data):
    print(data)
    if prediction_state.is_predicting.is_set():
        return socketio.emit('capture_frame', {"ok": False, "message": "Vui lòng nhấn Stop trước khi Capture"})
    
    return socketio.emit('capture_frame', {"ok": True, 'frame': old_frame})

@socketio.on('connect')
def handle_connect():
    print('Client connected to WebSocket')
    
if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
