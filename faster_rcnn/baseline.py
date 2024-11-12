import argparse
from function import collate_fn
from torch.utils.data import DataLoader, SubsetRandomSampler
import albumentations as A 
from evaluate import evaluate
from .train import train_step
from dataloader import TrafficVehicle
import torch
import numpy as np
from torchvision import models
from torch import nn

torch.manual_seed(42)

arg = argparse.ArgumentParser(description="Các tham số truyền vào")
arg.add_argument("--epoch", type=int, default=10, help="Số lượng epoch cho training")
arg.add_argument("--momentum", type=float, default=0, help="Hệ số momentum cho optimizer")
arg.add_argument("--weight_decay", type=float, default=0, help="Hệ số weight decay cho optimizer")

parse = arg.parse_args()

transform = A.Compose([
    A.Resize(300, 300),
    A.normalize(mean=0.0, std=1.0)
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

if __name__ == '__main__':
    train = TrafficVehicle(folder="train", transforms=transform, transform_box_type="corner")
    val = TrafficVehicle(folder="val", transforms=transform, transform_box_type="corner")
    
    # Load DataLoader
    train_dataloader = DataLoader(dataset=train, batch_size=32, shuffle=True, num_workers=1, collate_fn=collate_fn)
    val_dataloader = DataLoader(dataset=train, batch_size=32, shuffle=True, num_workers=1, collate_fn=collate_fn)

    # Khởi tạo model
    weights = models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = models.detection.fasterrcnn_resnet50_fpn(weights=weights)

    # Đóng băng các parameters
    for param in model.parameters():
        param.requires_grad = False

    # Finetuning model 
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    num_class = 5 # 4 object + 1 background class
    model.roi_heads.box_predictor.cls_score = nn.Linear(in_features=in_features, out_features=num_class)
    model.roi_heads.box_predictor.bbox_pred = nn.Linear(in_features=in_features, out_features=num_class * 4)

    # Thiết lập optimizer, device
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01, 
                                momentum=parse.momentum, weight_decay=parse.weight_decay)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Bắt đầu quá trình training
    print(f"-----------------------Training-----------------------\n")
    results = {
        "epoch_value": [],
        "loss": [],
        "loss_classifier": [],
        "loss_box_reg": [],
        "loss_objectness": [],
        "loss_rpn_box_reg": []
    }
    for epoch in range(parse.epoch):
        print(f"Epoch {epoch}/{parse.epoch}")
        loss, loss_classifier, loss_box_reg, loss_objectness, loss_rpn_box_reg = train_step(train_dataloader=train_dataloader, 
                                                                                            model=model,
                                                                                            optimizer=optimizer,
                                                                                            device=device)
        mAP, precisions_per_class, recalls_per_class, confidence_per_class = evaluate(val_dataloader=val_dataloader,
                                                                                    model=model,
                                                                                    num_class=num_class-1,
                                                                                    iou_threshold=0.5,
                                                                                    device=device)
    
 




