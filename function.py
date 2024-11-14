import numpy as np
import matplotlib.pyplot as plt
import torchvision.ops as ops
import torch

def collate_fn(batch):
    return tuple(zip(*batch))

def iou(boxA, boxB):
    # Tọa độ hình chữ nhật tương ứng phần giao nhau
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Tính diện tích phần giao nhau
    interArea = (xB - xA + 1) * (yB - yA + 1)

    # Diện tích của predicted và ground-truth bounding box
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # Tính IoU
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def calculate_precision_recall_ap(true_boxes, true_labels, pred_boxes, pred_scores, pred_labels, iou_threshold=0.5):
    """
        Tính precision, recall và average precision cho một class

        Args:
            true_boxes: numpy array của ground truth boxes
            true_labels: numpy array của ground truth labels
            pred_boxes: numpy array của predicted boxes
            pred_scores: numpy array của confidence scores
            pred_labels: numpy array của predicted labels """
    # Sắp xếp predictions theo confidence score giảm dần
    sort_idx = np.argsort(-np.array(pred_scores))
    pred_boxes = [pred_boxes[i] for i in sort_idx]
    pred_scores = [pred_scores[i] for i in sort_idx]
    pred_labels = [pred_labels[i] for i in sort_idx]

    num_predictions = len(pred_boxes)
    num_gt = len(true_boxes)

    # Arrays để lưu true positives và false positives
    tp = np.zeros(num_predictions)
    fp = np.zeros(num_predictions)

    # Đánh dấu GT boxes đã được match
    gt_matched = np.zeros(num_gt)

    # Duyệt qua từng prediction
    for pred_idx, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):
        # Lọc GT boxes cùng class
        gt_mask = (true_labels == pred_label)
        gt_boxes_class = true_boxes[gt_mask]

        if len(gt_boxes_class) == 0:
            fp[pred_idx] = 1
            continue

        # Tính IoU với tất cả GT boxes cùng class
        ious = np.array([iou(pred_box, gt_box) for gt_box in gt_boxes_class])
        max_iou = np.max(ious)
        max_idx = np.argmax(ious)
        
        # Lấy chỉ số tương ứng trong true_boxes gốc
        gt_idx = np.where(gt_mask)[0][max_idx]

        # Kiểm tra match
        if max_iou >= iou_threshold and not gt_matched[gt_idx]:
            tp[pred_idx] = 1
            gt_matched[gt_idx] = 1
        else:
            fp[pred_idx] = 1

    # Tính cumulative precision và recall
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)

    recalls = cum_tp / num_gt 
    precisions = cum_tp / (cum_tp + cum_fp)

    # Tính AP using Interpolation all point
    ap = 0
    for i in range(1, num_predictions):
         ap += max(precisions[i], precisions[i - 1]) * (recalls[i] - recalls[i - 1])

    return precisions, recalls, ap

def evaluate_predictions(predictions, targets, num_class, iou_threshold):
        """
        Đánh giá predictions cho một batch

        Args:
            predictions: list của dictionaries chứa predictions cho mỗi ảnh
            targets: list của dictionaries chứa ground truth cho mỗi ảnh

        Returns:
            dict: AP cho mỗi class trong batch
        """
        ap_per_class = {i: [] for i in range(1, num_class + 1)}
        precision_per_class = {i: [] for i in range(1, num_class + 1)}
        recall_per_class = {i: [] for i in range(1, num_class + 1)}
        confidence_per_class = {i: [] for i in range(1, num_class + 1)}
        
        for class_id in range(1, num_class + 1):
            all_true_boxes = []
            all_pred_boxes = []
            all_pred_scores = []
            all_true_labels = []
            all_pred_labels = []

            for target, pred in zip(targets, predictions):
                true_boxes = target['boxes']
                true_labels = target['labels']

                pred_boxes = pred['boxes']
                pred_scores = pred['scores']
                pred_labels = pred['labels']

                # Áp dụng NMS
                keep = ops.nms(
                    pred_boxes,
                    pred_scores,
                    iou_threshold=iou_threshold
                )
                pred_boxes = pred_boxes[keep]
                pred_scores = pred_scores[keep]
                pred_labels = pred_labels[keep]

                true_mask = (true_labels == class_id)
                pred_mask = (pred_labels == class_id)
                
                # Gộp tất cả các giá trị vào list
                all_true_boxes.append(true_boxes[true_mask])
                all_true_labels.append(true_labels[true_mask])
                all_pred_boxes.append(pred_boxes[pred_mask])
                all_pred_scores.append(pred_scores[pred_mask])
                all_pred_labels.append(pred_labels[pred_mask])

                
            # Nối các danh sách để có tất cả predictions và targets của class hiện tại
            all_true_boxes = torch.cat(all_true_boxes).cpu().numpy()
            all_pred_boxes = torch.cat(all_pred_boxes).cpu().numpy()
            all_pred_scores = torch.cat(all_pred_scores).cpu().numpy()
            all_true_labels = torch.cat(all_true_labels).cpu().numpy()
            all_pred_labels = torch.cat(all_pred_labels).cpu().numpy()

            if len(all_true_boxes) == 0 and len(all_pred_boxes) == 0:
                continue

            precisions, recalls, ap = calculate_precision_recall_ap(
                all_true_boxes,
                all_true_labels,
                all_pred_boxes,
                all_pred_scores,
                all_pred_labels,
                iou_threshold=iou_threshold
            )
            
            ap_per_class[class_id].append(ap)
            precision_per_class[class_id].append(precisions)
            recall_per_class[class_id].append(recalls)

            # Lưu lại confidence scores tương ứng với predictions của class
            confidence_per_class[class_id].append(all_pred_scores)

        return ap_per_class, precision_per_class, recall_per_class, confidence_per_class

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
        precisions = np.array(matrix['precisions per class'][class_id])
        recalls = np.array(matrix['recalls per class'][class_id])
        plt.plot(precisions, recalls, label=f'{class_name[class_id]}', color=list_color[class_id])

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='best')
    plt.show()

def confidence_matrix(matrix, confidence_per_class, num_class, class_name, matrix_name):
    plt.figure(figsize=(10, 8))
    list_color = ['orange', 'purple', 'red', 'green']
    for class_id in range(1, num_class + 1):
        confidence = np.array(confidence_per_class[class_id])
        sorted_indices = np.argsort(-confidence)
        sorted_matrix = np.array(matrix[class_id])[sorted_indices]
        sorted_confidence = confidence[sorted_indices]
        plt.plot(sorted_matrix, sorted_confidence, label=f'{class_name[class_id]}', color=list_color[class_id])

    plt.xlabel('Confidence')
    plt.ylabel(f'{matrix_name}')
    plt.title(f'{matrix_name}-Confidence Curve')
    plt.legend(loc='best')
    plt.show()


