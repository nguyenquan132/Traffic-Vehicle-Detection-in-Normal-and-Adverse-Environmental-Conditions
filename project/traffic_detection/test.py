from ultralytics import YOLO

model = YOLO("weights/yolov9s_18_11.pt")
results = model.predict(source="uploads/cam_03_00805_jpg.rf.e5751f15cb6001bf1f11c08224b821f2.jpg", save=True, save_dir="results")

print("Kết quả lưu tại:", results)  # Kiểm tra thư mục và file kết quả
