from torch.utils.data import DataLoader
import torchvision
import torch
from tqdm.auto import tqdm
from torchmetrics.detection import MeanAveragePrecision

def evaluate(val_dataloader: DataLoader,
             model: torchvision.models,
             num_class: int,
             iou_threshold: float,
             device: torch.device):
    # Di chuyển mô hình đến device (GPU hoặc CPU)
    model = model.to(device)
    model.eval()

    # Khởi tạo metric MeanAveragePrecision
    metric = MeanAveragePrecision(
        iou_type="bbox",  # Sử dụng bounding box
        iou_thresholds=[iou_threshold],  # Sử dụng ngưỡng IoU cụ thể (ví dụ: 0.5)
        box_format="xyxy",  # Định dạng bounding box là [xmin, ymin, xmax, ymax]
        class_metrics=True  # Tính toán AP cho từng lớp
    ).to(device)

    with torch.inference_mode():
        for batch, (images, targets) in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
            # Di chuyển ảnh và targets đến device
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Dự đoán
            outputs = model(images)

            # Chuẩn bị dữ liệu cho torchmetrics
            preds = [
                {
                    "boxes": output["boxes"],  # Bounding boxes
                    "scores": output["scores"],  # Độ tin cậy
                    "labels": output["labels"],  # Nhãn lớp
                }
                for output in outputs
            ]

            # Chuẩn bị ground truth cho torchmetrics
            gt = [
                {
                    "boxes": target["boxes"],  # Bounding boxes
                    "labels": target["labels"],  # Nhãn lớp
                }
                for target in targets
            ]

            # Cập nhật metric
            metric.update(preds, gt)

    # Tính toán kết quả
    result = metric.compute()

    # Trích xuất các giá trị cần thiết
    mAP = result["map_50"].item()  # mAP tại IoU threshold 0.5
    AP_per_class = {
        class_id: result["map_per_class"][class_id - 1].item()
        for class_id in range(1, num_class + 1)
    }


    return mAP, AP_per_class