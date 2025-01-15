import argparse
from function import collate_fn
from torch.utils.data import DataLoader
import albumentations as A 
from dataloader import TrafficVehicle
import torch
import torchvision
from albumentations.pytorch import ToTensorV2
from .model import TrafficModel
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models.detection import FasterRCNN
from PIL import Image
import numpy as np

torch.manual_seed(42)

arg = argparse.ArgumentParser(description="Các tham số truyền vào")
arg.add_argument("--epoch", type=int, default=10, help="Số lượng epoch cho training")
arg.add_argument("--momentum", type=float, default=0, help="Hệ số momentum cho optimizer")
arg.add_argument("--weight_decay", type=float, default=0, help="Hệ số weight decay cho optimizer")
arg.add_argument("--mode", type=str, default="train", help="Trạng thái cho model train hoặc test hoặc test one image")
arg.add_argument("--score_threshold", type=float, default=0.2, help="Threshold để lọc confidence")
arg.add_argument("--image_path", type=str, default="public test/public test/cam_08_00500_jpg.rf.5ab59b5bcda1d1fad9131385c5d64fdb.jpg",
                 help="Image cho test one image")

parse = arg.parse_args()

train_transform = A.Compose([
    A.Resize(300, 300),
    A.GaussianBlur(blur_limit=(3, 7), p=0.5),
    A.RandomBrightnessContrast(brightness_limit=(-0.3, -0.3), contrast_limit=(0.2, 0.2), p=0.5),
    A.HueSaturationValue(hue_shift_limit=(1, 1), p=0.5),
    A.Sharpen(p=0.5),
    A.ToFloat(max_value=255.0),
    ToTensorV2(p=1.0)
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], min_visibility=0.3))

test_transform = A.Compose([
    A.Resize(300, 300),
    A.ToFloat(max_value=255.0),
    ToTensorV2(p=1.0)
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

if __name__ == '__main__':

    # Khởi tạo model
    weights = torchvision.models.ResNet50_Weights.DEFAULT
    resnet = torchvision.models.resnet50(weights=weights)

    # Xác định returned layers
    return_layers = {'layer1': '0', 'layer2': '1',  'layer3': '2',  'layer4': '3' }

    # Tạo backbone với FPN
    backbone = BackboneWithFPN(
        resnet,
        return_layers=return_layers,
        in_channels_list=[256, 512, 1024, 2048],
        out_channels=256
    )

    # Khởi tạo FasterRCNN
    num_classes = 5
    model = FasterRCNN(backbone, 
                       num_classes=num_classes)
    
    if parse.mode == "train":
        MODE = TrafficModel(model=model)

        train_data = TrafficVehicle(folder="train", transforms=train_transform, transform_box_type="corner")
        val_data = TrafficVehicle(folder="val", transforms=test_transform, transform_box_type="corner")
        
        # Load DataLoader
        train_dataloader = DataLoader(dataset=train_data, batch_size=8, shuffle=True, num_workers=1, collate_fn=collate_fn)
        val_dataloader = DataLoader(dataset=val_data, batch_size=8, shuffle=True, num_workers=1, collate_fn=collate_fn)

        MODE.train(train_data=train_data, 
                   train_dataloader=train_dataloader, 
                   val_dataloader=val_dataloader,
                   epochs=parse.epoch,
                   momentum=parse.momentum,
                   weight_decay=parse.weight_decay)

    if parse.mode == "test one image":
        # Load state dict of model
        model.load_state_dict(torch.load("faster_rcnn/model1/model/model1.pth", weights_only=True, map_location=torch.device('cpu')))
        MODE = TrafficModel(model=model)
        # Read image
        img = Image.open(parse.image_path)
        img = np.array(img)
        # Transform image
        transformed_img = test_transform(image=img)
        # Display image after predicting
        MODE.test_one_image(transformed_img['image'], score_threshold=parse.score_threshold)



    

    
 




