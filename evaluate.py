from torch.utils.data import DataLoader
import torchvision
import torch
from .function import evaluate_predictions
from tqdm.auto import tqdm
import numpy as np

def evaluate(val_dataloader: DataLoader,
              model: torchvision.models,
              num_class: int,
              iou_threshold: float,
              device: torch.device):
    model = model.to(device)
    model.eval()

    precisions_per_class = {i: [] for i in range(1, num_class + 1)}
    recalls_per_class = {i: [] for i in range(1, num_class + 1)}
    confidence_per_class = {i: [] for i in range(1, num_class + 1)}
    ap_per_class = {i: [] for i in range(1, num_class + 1)}

    with torch.inference_mode():
        for images, targets in tqdm(val_dataloader, total=len(val_dataloader)):
            image = list(image.to(device) for image in images)
            target = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(image)

            batch_ap, batch_precision, batch_recall, batch_confidence = evaluate_predictions(predictions=outputs,
                                                                                             targets=target,
                                                                                             num_class=num_class,
                                                                                             iou_threshold=iou_threshold)

            # Cập nhật AP, precision, recall, confidence cho từng class
            for class_id in range(1, num_class + 1):
                ap_per_class[class_id].extend(batch_ap[class_id])
                
                # Chỉ gọi np.concatenate nếu có giá trị
                if batch_precision[class_id]:
                    precisions_per_class[class_id].extend(np.concatenate(batch_precision[class_id]))
                if batch_recall[class_id]:
                    recalls_per_class[class_id].extend(np.concatenate(batch_recall[class_id]))
                if batch_confidence[class_id]:
                    confidence_per_class[class_id].extend(np.concatenate(batch_confidence[class_id]))

        # Tính AP cho mỗi class
        AP_per_class = {
            class_id: np.mean(aps) if len(aps) > 0 else 0
            for class_id, aps in ap_per_class.items()
        }

        # Tính mAP
        mAP = np.mean(list(AP_per_class.values()))
        


        return mAP, AP_per_class, precisions_per_class, recalls_per_class, confidence_per_class