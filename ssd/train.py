import sys
sys.path.append(r'E:\Traffic-Vehicle-Detection-in-Normal-and-Adverse-Environmental-Conditions')
from dataloader import TrafficVehicle
from torch.utils.data import DataLoader
import albumentations as A
from tqdm.auto import tqdm
import torch
from model import create_model
from torch.optim.lr_scheduler import StepLR

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

        #train_loss_hist.send(loss_value)

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
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

    train_dataset = TrafficVehicle(r'E:\Vehicle Detection\train', transforms=transform, transform_box_type="corner")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, collate_fn=lambda x: tuple(zip(*x))) 
    # collate_fn=lambda x: tuple(zip(*x)): change from [(data1, label1), (data2, label2), (data3, label3)] to ((data1, data2, data3), (label1, label2, label3))

    model = create_model(num_classes=5)
    model = model.to(DEVICE)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.0001, momentum=0.9, nesterov=True)
    scheduler = StepLR(
        optimizer=optimizer, step_size=15, gamma=0.1, verbose=True
    )

    for epoch in range(2):
        print(epoch)
        train_loss = train(train_loader, model)