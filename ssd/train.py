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

if __name__ == '__main__':
    transform = A.Compose([
        A.Resize(300, 300),
        #A.HorizontalFlip(p=0.5),
        #A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),  # Normalize for ImageNet
        ToTensorV2(p=1.0),  # Convert to tensor
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

    train_data_path = os.path.join(os.path.dirname(__file__), '..', 'data/train')
    train_dataset = TrafficVehicle(train_data_path, transforms=transform, transform_box_type="corner")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    val_data_path = os.path.join(os.path.dirname(__file__), '..', 'data/val')
    val_dataset = TrafficVehicle(val_data_path, transforms=None, transform_box_type="corner")
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=lambda x: tuple(zip(*x))) 
    # collate_fn=lambda x: tuple(zip(*x)): change from [(data1, label1), (data2, label2), (data3, label3)] to ((data1, data2, data3), (label1, label2, label3))
    
    # for i in range(1):
    #     images, targets = next(iter(train_loader))
    #     images = list(image.to(DEVICE) for image in images)
    #     targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
    #     boxes = targets[i]['boxes'].cpu().numpy().astype(np.int32)
    #     sample = images[i].permute(1, 2, 0).cpu().numpy()
    #     sample = cv2.cvtColor(sample, cv2.COLOR_RGB2BGR)
    #     for box in boxes:
    #         cv2.rectangle(sample, (box[0],box[1]), (box[2],box[3]), (0,0,255),2)
    #         #print(box)
    #     cv2.imshow('img', sample)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    model = create_model(num_classes=5)
    model = model.to(DEVICE)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.0001, momentum=0.9, nesterov=True)
    scheduler = StepLR(
        optimizer=optimizer, step_size=15, gamma=0.1, verbose=True
    )
    save_best_model = SaveBestModel()
    num_epoch = 20

    # To monitor training loss
    train_loss_hist = Averager()
    # To store training loss and mAP values.
    train_loss_list = []
    map_50_list = []

    for epoch in range(num_epoch):
        print(f"\nEPOCH {epoch+1} / {num_epoch}")
        train_loss_hist.reset()
        train_loss = train(train_loader, model)
        val_metric = evaluate(val_dataloader=val_loader, model=model, num_class=5, iou_threshold=0.5, device=DEVICE)
        print(f"Epoch #{epoch+1} train loss: {train_loss_hist.value:.3f}")   
        print(f"Epoch #{epoch+1} mAP: {val_metric['mAP']}")

        train_loss_list.append(train_loss)
        map_50_list.append(val_metric['mAP'])

        save_best_model(
            model, float(val_metric['map']), epoch, 'outputs'
        )
        scheduler.step()
