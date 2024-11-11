import numpy as np
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt

def collate_fn(batch):
    return tuple(zip(*batch))

def iou(boxA, boxB):
    # Toạ độ hình chữ nhật tương ứng phần giao nhau
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Tính diện tích phần giao nhau
    interArea = (xB - xA + 1) * (yB - yA + 1)

    # Diện tích của predicted và ground-truth bounding box
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # Diện tích phần hợp = tổng diện tích trừ diện tích phần giao
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou

def calculate_precision_recall(true_boxes, pred_boxes, iou_threshold=0.5):
    y_true = []
    y_pred = []
    
    # Đánh dấu true_boxes đã được match
    matched_true_boxes = set()
    
    for pred_box in pred_boxes:
        best_iou = 0
        best_true_idx = -1
        
        # Tìm true_box có IoU cao nhất với pred_box hiện tại
        for i, true_box in enumerate(true_boxes):
            current_iou = iou(pred_box, true_box)
            if current_iou > best_iou:
                best_iou = current_iou
                best_true_idx = i
        
        # Nếu tìm được match và true_box này chưa được match trước đó
        if best_iou >= iou_threshold and best_true_idx not in matched_true_boxes:
            y_true.append(1)  # True Positive
            y_pred.append(1)
            matched_true_boxes.add(best_true_idx)
        else:
            y_true.append(0)  # False Positive
            y_pred.append(1)
    
    # Thêm False Negatives cho các true_boxes chưa được match
    unmatched_true_boxes = len(true_boxes) - len(matched_true_boxes)
    for _ in range(unmatched_true_boxes):
        y_true.append(1)
        y_pred.append(0)

    if len(y_true) == 0 or len(y_pred) == 0:
        return 0.0, 0.0
    else:
        return precision_score(y_true, y_pred), recall_score(y_true, y_pred)

def calculate_ap(precisions, recalls):
    precisions.append(1)
    recalls.append(0)

    precisions = np.array(precisions)
    recalls = np.array(recalls)

    AP = np.sum((recalls[:-1] - recalls[1:]) * precisions[:-1])

    return AP

def loss_curve(results, title):
    """
    results={
        "epoch_value" = [],
        "loss" = []
        }
    """
    epoch = results['epoch_value']
    loss = results['loss']
    plt.figure(figsize=(15, 7))
    plt.plot(epoch, loss, marker='o', color='#87CEEB')
    plt.title(f"train/{title}")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.show()

def precision_recall_curve(metric):
    """
    Precision, recall của mỗi class
    """
    plt.figure(figsize=(15, 7))


