from evaluate import evaluate
from .train import train_step
import numpy as np
from ssd import inference
from torch import nn
from function import loss_mAP_curve, precision_recall_curve, confidence_metric
import torch

class TrafficModel():
    def __init__(self, model):
        self.model = model
    def train(self, train_data, train_dataloader, val_dataloader, epochs, momentum=0, weight_decay=0):
        # Đóng băng các parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Finetuning model 
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        num_class = 5 # 4 object + 1 background class
        self.model.roi_heads.box_predictor.cls_score = nn.Linear(in_features=in_features, out_features=num_class)
        self.model.roi_heads.box_predictor.bbox_pred = nn.Linear(in_features=in_features, out_features=num_class * 4)
        class_name = train_data.class_name

        # Thiết lập optimizer, device
        optimizer = torch.optim.SGD(params=self.model.parameters(), lr=0.01, 
                                    momentum=momentum, weight_decay=weight_decay)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Bắt đầu quá trình training
        print(f"-----------------------Training-----------------------\n")
        results = {
            "epoch_value": [],
            "loss": [],
            "loss_classifier": [],
            "loss_box_reg": [],
            "loss_objectness": [],
            "loss_rpn_box_reg": [],
            "mAP50": []
        }
        metrics = {
            "precisions per class": {class_id: [] for class_id in range(1, num_class)},  
            "recalls per class": {class_id: [] for class_id in range(1, num_class)},    
        }
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{1}")
            loss, loss_classifier, loss_box_reg, loss_objectness, loss_rpn_box_reg = train_step(train_dataloader=train_dataloader,
                                                                                                model=self.model,
                                                                                                optimizer=optimizer,
                                                                                                device=device)
            mAP, AP_per_class, precisions_per_class, recalls_per_class = evaluate(val_dataloader=val_dataloader,
                                                                                model=self.model,
                                                                                num_class=num_class-1,
                                                                                iou_threshold=0.5,
                                                                                device=device)
            
            results["epoch_value"].append(epoch + 1)
            results["loss"].append(loss.item if isinstance(loss, torch.Tensor) else loss)
            results["loss_classifier"].append(loss_classifier.item if isinstance(loss_classifier, torch.Tensor) else loss_classifier)
            results["loss_box_reg"].append(loss_box_reg.item if isinstance(loss_box_reg, torch.Tensor) else loss_box_reg)
            results["loss_objectness"].append(loss_objectness.item if isinstance(loss_objectness, torch.Tensor) else loss_objectness)
            results["loss_rpn_box_reg"].append(loss_rpn_box_reg.item if isinstance(loss_rpn_box_reg, torch.Tensor) else loss_rpn_box_reg)

            print(f"Loss: {loss:.4f}, Loss classifier: {loss_classifier:.4f}, Loss box: {loss_box_reg:.4f}, Loss objectness: {loss_objectness:.4f}, Loss rpn_box: {loss_rpn_box_reg:.4f}")

            for class_id in range(1, num_class):
                # Tính giá trị trung bình của precision, recall cho từng class 
                mean_precision = np.mean(list(precisions_per_class[class_id]))
                mean_recall = np.mean(list(recalls_per_class[class_id]))
                
                metrics["precisions per class"][class_id].append(mean_precision)
                metrics["recalls per class"][class_id].append(mean_recall)

                print(f"Average Precision of {train_data.class_name[class_id]}: {AP_per_class[class_id]}")
            
            # Calculate overall mAP
            mAP = np.mean(list(AP_per_class.values()))
            results["mAP50"].append(mAP)

            print(f"Mean Average Precision (mAP@IoU=0.5): {mAP:.4f}")

        # Visualize loss và mAP 
        loss_mAP_curve(results)
        # Visualize precision-recall curve
        precision_recall_curve(metrics=metrics, num_class=num_class-1, class_name=class_name)
        # Visualize precision-confidence curve
        confidence_metric(metric=metrics["precisions per class"], num_class=num_class-1,
                        class_name=class_name, metric_name="Precision")
        # Visualize recall-confidence curve
        confidence_metric(metric=metrics["recalls per class"], num_class=num_class-1,
                        class_name=class_name, metric_name="Recall")
    
    def test(self):
        inference()