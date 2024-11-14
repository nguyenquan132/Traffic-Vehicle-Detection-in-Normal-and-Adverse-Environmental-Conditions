import os
from dataloader import TrafficVehicle
from torch.utils.data import DataLoader
import albumentations as A
from tqdm.auto import tqdm
import torch
from .model import create_model
from torch.optim.lr_scheduler import StepLR
from albumentations.pytorch import ToTensorV2
from .utils import * 
from evaluate import evaluate
from torchmetrics.detection.mean_ap import MeanAveragePrecision

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def train(train_data_loader, model):
    print('Training')
    model.train()
    
     # initialize tqdm progress bar
    prog_bar = tqdm(train_data_loader, total=len(train_data_loader))
    
    for i, data in enumerate(prog_bar):
        optimizer.zero_grad()
        images, targets = data
        
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()

        train_loss_hist.send(loss_value)

        losses.backward()
        optimizer.step()
    
        # update the loss value beside the progress bar for each iteration
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
    return loss_value

def validate(valid_data_loader, model):
    print('Validating')
    model.eval()
    
    # Initialize tqdm progress bar.
    prog_bar = tqdm(valid_data_loader, total=len(valid_data_loader))
    target = []
    preds = []
    for i, data in enumerate(prog_bar):
        images, targets = data
        
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        
        with torch.no_grad():
            outputs = model(images, targets)

        # For mAP calculation using Torchmetrics.
        #####################################
        for i in range(len(images)):
            true_dict = dict()
            preds_dict = dict()
            true_dict['boxes'] = targets[i]['boxes'].detach().cpu()
            true_dict['labels'] = targets[i]['labels'].detach().cpu()
            preds_dict['boxes'] = outputs[i]['boxes'].detach().cpu()
            preds_dict['scores'] = outputs[i]['scores'].detach().cpu()
            preds_dict['labels'] = outputs[i]['labels'].detach().cpu()
            preds.append(preds_dict)
            target.append(true_dict)
        #####################################

    metric = MeanAveragePrecision()
    metric.update(preds, target)
    metric_summary = metric.compute()
    return metric_summary

if __name__ == '__main__':
    train_transform = A.Compose([
        A.Resize(300, 300),
        #A.HorizontalFlip(p=0.5),
        #A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        A.ToFloat(max_value=255.0),
        ToTensorV2(p=1.0),  # Convert to tensor
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

    val_transform = A.Compose([
            A.Resize(300, 300),
            A.ToFloat(max_value=255.0),
            ToTensorV2(p=1.0),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

    train_data_path = os.path.join(os.path.dirname(__file__), '..', 'data/train')
    train_dataset = TrafficVehicle(train_data_path, transforms=train_transform, transform_box_type="corner")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    val_data_path = os.path.join(os.path.dirname(__file__), '..', 'data/val')
    val_dataset = TrafficVehicle(val_data_path, transforms=val_transform, transform_box_type="corner")
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=lambda x: tuple(zip(*x))) 

    model = create_model(num_classes=5)
    model = model.to(DEVICE)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, nesterov=True)
    scheduler = StepLR(
        optimizer=optimizer, step_size=10, gamma=0.1, verbose=True
    )
    num_epoch = 40
    train_loss_hist = Averager()

    # To store training loss and mAP values.
    train_loss_list = []
    map_list = []
    map_per_class_list = []
    best_valid_map = 0
    patience = 4
    patience_counter = 0

    for epoch in range(num_epoch):
        print(f"\nEPOCH {epoch+1} / {num_epoch}")
        train_loss_hist.reset()
        train_loss = train(train_loader, model)
        val_metric = validate(val_loader, model)
        print(f"Epoch #{epoch+1} train loss: {train_loss_hist.value:.3f}")   
        print(f"Epoch #{epoch+1} mAP: {val_metric['map']}")

        train_loss_list.append(train_loss)
        map_list.append(val_metric['map'])
        map_per_class_list.append(val_metric['map_per_class'])
        current_valid_map = val_metric['map']
        
        if current_valid_map > best_valid_map:
            best_valid_map = current_valid_map
            patience_counter = 0
            print(f"\nBEST VALIDATION mAP: {best_valid_map}")
            print(f"\nSAVING BEST MODEL FOR EPOCH: {epoch+1}\n")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                }, "best_model.pth")
        else:
            patience_counter += 1
            print(f"No improvement in map score for {patience_counter} epochs.")
        if patience_counter >= patience:
            print("Early stopping triggered. No improvement in map score.")
            break
        scheduler.step()