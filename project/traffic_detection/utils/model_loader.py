import torch
from ultralytics import YOLO
from .zero_dce import ZeroDCE

def load_yolo():
    """
    Load YOLO model.
    """
    return YOLO("weights/yolov9s_18_11.pt")

def load_zero_dce():
    """
    Load Zero-DCE model.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ZeroDCE()
    model.load_state_dict(torch.load("weights/8LE-color-loss2_best_model.pth", map_location=device)['model'])
    model = model.to(device)
    model.eval()
    return model, device
