from evaluate import evaluate
from .train import train_step
import numpy as np
from torch import nn
from function import loss_mAP_curve
import torch
import cv2
import supervision as sv

class_name = {0: "motorbike",
              1: "car",
              2: "coach",
              3: "container"}

def MMS(score_threshold, output):
    boxes_array, labels_array, scores_array = [], [], []
    for i in range(len(output['scores'])):
        if output['scores'][i] >= score_threshold:
            boxes_array.append(output['boxes'][i])
            labels_array.append(output['labels'][i])
            scores_array.append(output['scores'][i])

    return np.array(boxes_array), np.array(labels_array), np.array(scores_array)

def draw_box(image, output, score_threshold):
    bounding_box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_position=sv.Position.TOP_CENTER)

    xyxy, class_id, confidence = MMS(score_threshold, output)

    detections = sv.Detections(
        xyxy=xyxy,
        confidence=confidence,
        class_id=class_id
    )

    labels = [f"{class_name[class_id]} {conf:.2f}" for _, _, conf, class_id, *_ in detections]

    frame = image.copy()
    image_annotator = bounding_box_annotator.annotate(scene=frame, detections=detections)
    image_annotator = label_annotator.annotate(scene=image_annotator, detections=detections, labels=labels)

    return image_annotator

class TrafficModel():
    def __init__(self, model):
        self.model = model
    def train(self, train_data, train_dataloader, val_dataloader, epochs, momentum=0, weight_decay=0):
        # Đóng băng các parameters
        for param in self.model.backbone.parameters():
            param.requires_grad = False

        num_classes = 5

        # Thiết lập optimizer, device
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(
            params,
            lr=0.005,
            momentum=0.9,
            weight_decay=0.0005
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=3,
            gamma=0.1
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Bắt đầu quá trình training
        results = {
            "epoch_value": [],
            "loss": [],
            "loss_classifier": [],
            "loss_box_reg": [],
            "loss_objectness": [],
            "loss_rpn_box_reg": [],
            "mAP50": []
        }
        epochs = 20

        patience = 3  # Số epoch không cải thiện mAP
        best_mAP = 0  # Lưu mAP tốt nhất
        patience_counter = 0  # Đếm số epoch mà mAP không cải thiện

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            loss, loss_classifier, loss_box_reg, loss_objectness, loss_rpn_box_reg = train_step(train_dataloader=train_dataloader,
                                                                                                model=model,
                                                                                                optimizer=optimizer,
                                                                                                device=device)

            lr_scheduler.step()
            
            mAP, AP_per_class = evaluate(val_dataloader=val_dataloader,
                                        model=self.model,
                                        num_class=num_classes-1,
                                        iou_threshold=0.5,
                                        device=device)

            results["epoch_value"].append(epoch + 1)
            results["loss"].append(loss.item() if isinstance(loss, torch.Tensor) else loss)
            results["loss_classifier"].append(loss_classifier.item() if isinstance(loss_classifier, torch.Tensor) else loss_classifier)
            results["loss_box_reg"].append(loss_box_reg.item() if isinstance(loss_box_reg, torch.Tensor) else loss_box_reg)
            results["loss_objectness"].append(loss_objectness.item() if isinstance(loss_objectness, torch.Tensor) else loss_objectness)
            results["loss_rpn_box_reg"].append(loss_rpn_box_reg.item() if isinstance(loss_rpn_box_reg, torch.Tensor) else loss_rpn_box_reg)

            print(f"Loss: {loss:.4f}, Loss classifier: {loss_classifier:.4f}, Loss box: {loss_box_reg:.4f}, Loss objectness: {loss_objectness:.4f}, Loss rpn_box: {loss_rpn_box_reg:.4f}")

            for class_id in range(1, num_classes):
                class_ap = AP_per_class.get(class_id, 0)
                print(f"Average Precision of {train_data.class_name[class_id]}: {class_ap:.4f}")

            results["mAP50"].append(mAP)

            print(f"Mean Average Precision (mAP@IoU=0.5): {mAP:.4f}")

            # Early stopping check
            if mAP > best_mAP:
                best_mAP = mAP
                patience_counter = 0  # Reset counter nếu mAP không cải thiện
                # Save model
                torch.save(obj=self.model.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1} due to no improvement in mAP.")
                    break  # Dừng huấn luyện khi không cải thiện mAP trong nhiều epoch

        # Visualize loss và mAP 
        loss_mAP_curve(results)
    
    def test_one_image(self, img, score_threshold):
        with torch.inference_mode():
            output = self.model(img.unsqueeze(0))[0]
        image_annotator = draw_box(img, output, score_threshold)

        cv2.imshow('Video Preview', image_annotator)

        cv2.waitKey(0)
        cv2.destroyAllWindows()