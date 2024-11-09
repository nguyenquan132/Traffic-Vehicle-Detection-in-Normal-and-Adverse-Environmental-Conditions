import os 
import torch 
import cv2
from torchvision import transforms
import albumentations as A
from torchvision.models.detection import ssd300_vgg16
from torch.utils.data import Dataset, DataLoader
from dataloader import TrafficVehicle
import torch.optim as optim

transform = A.Compose([
    A.Resize(300, 300),
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

train_dataset = TrafficVehicle(r'E:\Vehicle Detection\train', transforms=transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False, collate_fn=lambda x: tuple(zip(*x))) 
# collate_fn=lambda x: tuple(zip(*x)): change from [(data1, label1), (data2, label2), (data3, label3)] to ((data1, data2, data3), (label1, label2, label3))

model = ssd300_vgg16(pretrained=True)
num_classes = 4
model.head.classification_head.num_classes = num_classes

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
criterion = model.head.loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

