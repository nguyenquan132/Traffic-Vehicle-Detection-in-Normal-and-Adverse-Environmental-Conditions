import os
import glob
import shutil
from .model_loader import load_yolo

def yolo_predict(image_path, result_folder, label_mapping):
    """
    Dự đoán đối tượng trong ảnh bằng YOLO và trả về kết quả.
    """
    model = load_yolo()
    results = model.predict(source=image_path, save=True, save_dir="runs/detect")

    # Nếu không phát hiện đối tượng
    if not results or len(results[0].boxes) == 0:
        return None, None, None

    # Tìm đường dẫn kết quả của YOLO
    yolo_output_dir = results[0].save_dir
    predicted_file = glob.glob(os.path.join(yolo_output_dir, "*.jpg"))[0]
    result_path = os.path.join(result_folder, os.path.basename(predicted_file))
    shutil.copy(predicted_file, result_path)

    # Trích xuất thông tin bounding box
    boxes = results[0].boxes.xywh.cpu().numpy()
    labels = results[0].boxes.cls.cpu().numpy()
    confidences = results[0].boxes.conf.cpu().numpy()

    detection_data = []
    for i in range(len(boxes)):
        label_index = int(labels[i])
        detection_data.append({
            "label": label_mapping.get(label_index, f"Unknown ({label_index})"),
            "x": boxes[i][0],
            "y": boxes[i][1],
            "w": boxes[i][2],
            "h": boxes[i][3],
            "confidence": confidences[i],
        })

    return result_path, detection_data, None
