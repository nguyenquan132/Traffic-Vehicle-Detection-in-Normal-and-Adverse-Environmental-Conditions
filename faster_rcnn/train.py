import torch
from torch.utils.data import DataLoader
import torchvision

def train_step(train_dataloader: DataLoader,
               model: torchvision.models,
               optimizer: torch.optim,
               device: torch.device):
    model = model.to(device)
    model.train()
    loss = 0
    loss_classifier, loss_box_reg, loss_objectness, loss_rpn_box_reg = 0, 0, 0, 0

    for batch, (images, targets) in enumerate(train_dataloader):
        image = list(image.to(device) for image in images)
        target = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(image, target)

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
