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

    for pred_box in pred_boxes:
        best_iou = 0
        for true_box in true_boxes:
            current_iou = iou(pred_box, true_box)
            best_iou = max(best_iou, current_iou)
        
        y_pred.append(1 if best_iou >= iou_threshold else 0)

    y_true = [1] * len(true_boxes) + [0] * (len(pred_boxes) - len(true_boxes))
    
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    return precision, recall

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


