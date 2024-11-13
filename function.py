import numpy as np
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt

def collate_fn(batch):
    return tuple(zip(*batch))

def iou(boxA, boxB):
    # Tọa độ hình chữ nhật tương ứng phần giao nhau
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Đảm bảo không có giá trị âm cho diện tích phần giao
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # Diện tích của các bounding boxes
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # Tính IoU
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou

def calculate_precision_recall(true_boxes, pred_boxes, iou_threshold=0.5):
    y_true = []
    y_pred = []
    
    matched_true_boxes = set()  # Đánh dấu các true_boxes đã được match
    
    for pred_box in pred_boxes:
        best_iou = 0
        best_true_idx = -1
        
        # Tìm true_box có IoU cao nhất với pred_box hiện tại
        for i, true_box in enumerate(true_boxes):
            current_iou = iou(pred_box, true_box)
            if current_iou > best_iou:
                best_iou = current_iou
                best_true_idx = i
        
        # Nếu match và true_box này chưa được match
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
        y_true.append(1)  # False Negative
        y_pred.append(0)
    
    # Tính precision và recall
    precision = precision_score(y_true, y_pred, zero_division=1)
    recall = recall_score(y_true, y_pred, zero_division=1)
    
    return precision, recall

def calculate_ap(precisions, recalls):
    # Thêm giá trị boundary để tránh lỗi khi tính tích phân
    precisions = np.concatenate(([0], precisions, [0]))
    recalls = np.concatenate(([0], recalls, [1]))

    # Sắp xếp theo thứ tự recall giảm dần
    for i in range(len(precisions) - 1, 0, -1):
        precisions[i - 1] = max(precisions[i - 1], precisions[i])

    # Tính AP bằng cách lấy diện tích dưới đường cong precision-recall
    indices = np.where(recalls[1:] != recalls[:-1])[0]
    AP = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])

    return AP

def calculate_ap_per_class(precisions, recalls, confidences):
    confidences = np.array(confidences)
    sorted_indices = np.argsort(-confidences)  # Sắp xếp theo confidence giảm dần
    precisions = np.array(precisions)[sorted_indices]
    recalls = np.array(recalls)[sorted_indices]

    precisions, recalls, _ = precision_recall_curve(recalls, precisions)
    
    # Tính AP dựa trên đường cong Precision-Recall
    return calculate_ap(precisions, recalls)

def loss_curve(results, title):
    """
    results={
        "epoch_value" = [],
        "loss" = []
        }
    """
    epoch = results['epoch_value']
    loss = results['loss']
    plt.figure(figsize=(10, 8))
    plt.plot(epoch, loss, marker='o', color='#87CEEB')
    plt.title(f"train/{title}")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.show()

def precision_recall_curve(matrix, num_class, class_name):
    """
    Precision, recall của mỗi class
    matrix = {
        precisions per class: [],
        recalls per class: [],
        confidence per class: []
    }
    """
    plt.figure(figsize=(10, 8))
    list_color = ['orange', 'purple', 'red', 'green']
    for class_id in range(1, num_class + 1):
        plt.plot(matrix['precisions per class'][class_id], matrix['recalls per class'][class_id], 
                 label=f'{class_name[class_id]}', color=list_color[class_id])
        
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='best')
    plt.show()

def confidence_matrix(matrix, confidence, num_class, class_name, matrix_name):
    plt.figure(figsize=(10, 8))
    list_color = ['orange', 'purple', 'red', 'green']
    for class_id in range(1, num_class + 1):
        plt.plot(matrix[class_id], confidence[class_id], 
                 label=f'{class_name[class_id]}', color=list_color[class_id])
        
    plt.xlabel('Confidence')
    plt.ylabel(f'{matrix_name}')
    plt.title(f'{matrix_name}-Confidence Curve')
    plt.legend(loc='best')
    plt.show()


