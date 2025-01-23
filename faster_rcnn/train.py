import torch
from torch.utils.data import DataLoader
import torchvision
from tqdm.auto import tqdm

def train_step(train_dataloader: DataLoader,
               model: torchvision.models,
               optimizer: torch.optim,
               device: torch.device,
               model_dce=None):
    model = model.to(device)
    model.train()
    if model_dce is not None: model_dce.eval()
    loss = 0
    loss_classifier, loss_box_reg, loss_objectness, loss_rpn_box_reg = 0, 0, 0, 0

    for batch, (images, targets) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        image = list(image.to(device) for image in images)
        if model_dce is not None:
            enhanced_images = []
            for image in images:
                _, enhance_image, _ = model_dce(image.unsqueeze(0))
                enhanced_images.append(enhance_image.squeeze())
        target = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if model_dce is not None: loss_dict = model(enhanced_images, target)
        else: loss_dict = model(image, target)
        
        if batch == 1:
            print(f"batch 1: \n{loss_dict}")

        losses = sum(loss for loss in loss_dict.values())
        loss += losses

        loss_classifier += loss_dict['loss_classifier']
        loss_box_reg += loss_dict['loss_box_reg']
        loss_objectness += loss_dict['loss_objectness']
        loss_rpn_box_reg += loss_dict['loss_rpn_box_reg']

        optimizer.zero_grad()

        losses.backward()

        optimizer.step()

    loss /= len(train_dataloader)
    loss_classifier /= len(train_dataloader)
    loss_box_reg /= len(train_dataloader)
    loss_objectness /= len(train_dataloader)
    loss_rpn_box_reg /= len(train_dataloader)

    return loss, loss_classifier, loss_box_reg, loss_objectness, loss_rpn_box_reg
