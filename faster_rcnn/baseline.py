import argparse
from function import collate_fn
from torch.utils.data import DataLoader
import albumentations as A 
from dataloader import TrafficVehicle
import torch
from torchvision import models
from albumentations.pytorch import ToTensorV2
from .model import TrafficModel

torch.manual_seed(42)

arg = argparse.ArgumentParser(description="Các tham số truyền vào")
arg.add_argument("--epoch", type=int, default=10, help="Số lượng epoch cho training")
arg.add_argument("--momentum", type=float, default=0, help="Hệ số momentum cho optimizer")
arg.add_argument("--weight_decay", type=float, default=0, help="Hệ số weight decay cho optimizer")
arg.add_argument("--mode", type=str, default="train", help="Trạng thái cho model train hoặc test hoặc test one image")

parse = arg.parse_args()

train_transform = A.Compose([
    A.Resize(300, 300),
    A.Normalize(mean=0.0, std=1.0),
    ToTensorV2()
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

if __name__ == '__main__':

    # Khởi tạo model
    weights = models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = models.detection.fasterrcnn_resnet50_fpn(weights=weights)
    
    if parse.model == "train":
        MODE = TrafficModel(model=model)

        train_data = TrafficVehicle(folder="train", transforms=train_transform, transform_box_type="corner")
        val_data = TrafficVehicle(folder="val", transforms=train_transform, transform_box_type="corner")
        
        # Load DataLoader
        train_dataloader = DataLoader(dataset=train_data, batch_size=32, shuffle=True, num_workers=1, collate_fn=collate_fn)
        val_dataloader = DataLoader(dataset=val_data, batch_size=32, shuffle=True, num_workers=1, collate_fn=collate_fn)

        MODE.train(train_data=train_data, 
                   train_dataloader=train_dataloader, 
                   val_dataloader=val_dataloader,
                   epochs=parse.epoch,
                   momentum=parse.momentum,
                   weight_decay=parse.weight_decay)

    if parse.model == "test":
        pass

    if parse.model == "test one image":
        pass

    

    
 




