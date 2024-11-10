from torch.utils.data import DataLoader
import torchvision
import torch
from .function import calculate_ap, calculate_precision_recall
import torchvision.ops as ops
from tqdm.auto import tqdm

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

    with torch.inference_mode():
        for images, targets in tqdm(val_dataloader, total=len(val_dataloader)):
            image = list(image.to(device) for image in images)
            target = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(image)

            for img_idx, (target, output) in enumerate(zip(target, outputs)):
                true_boxes = target['boxes'].cpu().detach().numpy()
                true_labels = target['labels'].cpu().detach().numpy()
                
                pred_boxes = output['boxes'].cpu().detach().numpy()
                pred_scores = output['scores'].cpu().detach().numpy()
                pred_labels = output['labels'].cpu().detach().numpy()

                # Áp dụng NMS để loại bỏ các box dư thừa
                keep = ops.nms(torch.from_numpy(pred_boxes),
                               torch.from_numpy(pred_scores),
                               iou_threshold=iou_threshold)
                pred_boxes = pred_boxes[keep]
                pred_scores = pred_scores[keep]
                pred_labels = pred_labels[keep]

                for class_id in range(1, num_class + 1):
                    # Lọc true và predicted boxes theo class_id
                    true_boxes_class = true_boxes[true_labels == class_id]
                    pred_boxes_class = pred_boxes[pred_labels == class_id]
                    pred_scores_class = pred_scores[pred_labels == class_id]

                    precision, recall = calculate_precision_recall(true_boxes=true_boxes_class,
                                                                   pred_boxes=pred_boxes_class,
                                                                   iou_threshold=iou_threshold)
                    
                    # Lưu precision, recall và confidence scores cho từng class
                    precisions_per_class[class_id].append(precision)
                    recalls_per_class[class_id].append(recall)
                    confidence_per_class[class_id].extend(pred_scores_class)


        AP_per_class = {}
        for class_id in range(1, num_class + 1):
            precisions = precisions_per_class[class_id]
            recalls = recalls_per_class[class_id]

            # Tính AP cho class này
            AP_per_class[class_id] = calculate_ap(precisions=precisions, recalls=recalls)

        mAP = sum(AP_per_class.values) / num_class
        
        return mAP, precisions_per_class, recalls_per_class, confidence_per_class
            

