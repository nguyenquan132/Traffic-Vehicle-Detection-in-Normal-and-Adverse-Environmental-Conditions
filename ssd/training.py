import torch
from torchvision.ops import nms
from torchmetrics.detection.mean_ap import MeanAveragePrecision

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for images, targets in loader:
        images = list(img.to(device) for img in images)
        targets = [{ 
                        "box": t['box'].to(device, dtype=torch.float32),
                        "label": t['label'].to(device, dtype=torch.int64)
                    }
                    for t in targets]

        optimizer.zero_grad()
        losses = model(images, targets)
        loss = sum(loss for loss in losses.values())
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    with torch.no_grad():
        # Accumulate predictions and targets for mAP calculation
        all_predictions = []
        all_targets = []

        for images, targets in loader:
            images = list(img.to(device) for img in images)
            outputs = model(images)
            
            # Post-process outputs and store them
            for i, output in enumerate(outputs):
                boxes = output["boxes"]
                scores = output["scores"]
                labels = output["labels"]

                # Apply NMS for the boxes and collect predictions
                keep = nms(boxes, scores, 0.5)
                boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

                # Append to final predictions for mAP calculation
                all_predictions.append({"boxes": boxes.cpu(), "scores": scores.cpu(), "labels": labels.cpu()})
                all_targets.append(targets[i])

        # TODO: Implement mAP calculation here or use a library
