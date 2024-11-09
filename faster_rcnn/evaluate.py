from torch.utils.data import DataLoader
import torchvision
import torch
from function import iou, calculate_ap, calculate_precision_recall
import torchvision.ops as ops

def evaluate(val_dataloader: DataLoader,
              model: torchvision.models,
              device: torch.device):
    model = model.to(device)
    model.eval()
    precisions = []
    recalls = []

    with torch.inference_mode():
        for images, targets in val_dataloader:
            image = list(image.to(device) for image in images)
            target = [{k: v.to(device) for k, v in t.items()} for t in targets]

            output = model(image)

            for i, target in enumerate(target):
                true_boxes = target['box'].cpu().detach().numpy()
                pred_boxes = output[i]['boxes'].cpu().detach().numpy()
                scores = output[i]['scores'].cpu().detach().numpy()

                # Áp dụng NMS để loại bỏ các box dư thừa
                keep = ops.nms(pred_boxes, scores, iou_threshold=0.5)
                pred_boxes = pred_boxes[keep]
                scores = scores[keep]

                precision, recall = calculate_precision_recall(true_boxes=true_boxes, pred_boxes=pred_boxes, iou_threshold=0.5)
                precisions.append(precision)
                recalls.append(recall)

        AP = calculate_ap(precisions=precisions, recalls=recalls)

        return AP
            

