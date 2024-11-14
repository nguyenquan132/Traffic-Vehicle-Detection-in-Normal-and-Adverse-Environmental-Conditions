import os
from dataloader import TrafficVehicle
from torch.utils.data import DataLoader
import albumentations as A
from tqdm.auto import tqdm
import torch
from .model import create_model
from albumentations.pytorch import ToTensorV2
from torch.optim.lr_scheduler import StepLR

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

test_transform = A.Compose([
        A.Resize(300, 300),
        A.ToFloat(max_value=255.0),
        ToTensorV2(p=1.0),
    ])

test_data_path = '/kaggle/input/traffic-vehicle-detection/public test/public test'
test_dataset = TrafficVehicle(test_data_path, transforms=test_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = create_model(num_classes=5)
model = model.to(DEVICE)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, nesterov=True)
scheduler = StepLR(
    optimizer=optimizer, step_size=10, gamma=0.1, verbose=True
)

checkpoint = torch.load('/kaggle/input/traffic-vehicle-detection/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()
output_lines = []
with torch.inference_mode():
    print('Testing')

     # initialize tqdm progress bar
    prog_bar = tqdm(test_loader, total=len(test_loader))
    for i, data in enumerate(prog_bar): 
        images, file_names = data
        images = list(image.to(DEVICE) for image in images)
        outputs = model(images)
        outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
        for i, (output, file_name) in enumerate(zip(outputs, file_names)):
            boxes = output['boxes'].data.numpy()
            labels = output['labels'].data.numpy()
            scores = output['scores'].data.numpy()
            boxes = boxes[scores >= 0.25]
            labels = labels[scores >= 0.25] - 1
            scores = scores[scores >= 0.25]
            for box, label, score in zip(boxes, labels, scores):
                xmin = (box[0] / 300) * 1080
                ymin = (box[1] / 300) * 720
                xmax = (box[2] / 300) * 1080
                ymax = (box[3] / 300) * 720
                x_center = (xmin + xmax) / 2.0
                y_center = (ymin + ymax) / 2.0
                width = xmax - xmin
                height = ymax - ymin
                line = f"{file_name} {label.item()} {xmin.item()} {ymin.item()} {width.item()} {height.item()} {score.item()}"
                output_lines.append(line)

# Save to a file
with open("predict.txt", "w") as f:
    f.write("\n".join(output_lines))
