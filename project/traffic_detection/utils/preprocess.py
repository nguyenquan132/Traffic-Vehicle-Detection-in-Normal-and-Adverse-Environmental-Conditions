import cv2
import numpy as np
import torch
from .model_loader import load_zero_dce

def check_darkness(image_path, threshold=0.4):
    """
    Kiểm tra độ sáng trung bình của ảnh.
    """
    image = cv2.imread(image_path)
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    normalized_gray = grayscale / 255.0
    mean_brightness = np.mean(normalized_gray)
    return mean_brightness < threshold, mean_brightness

def enhance_image(image_path, save_path):
    """
    Tăng sáng ảnh bằng Zero-DCE và lưu kết quả.
    """
    model, device = load_zero_dce()
    image = cv2.imread(image_path)
    original_size = image.shape[:2]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    input_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        _, enhanced_image = model(input_tensor)
    
    enhanced_image = enhanced_image.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    enhanced_image = (enhanced_image * 255).astype(np.uint8)
    enhanced_image_resized = cv2.resize(enhanced_image, (original_size[1], original_size[0]))
    enhanced_image_bgr = cv2.cvtColor(enhanced_image_resized, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, enhanced_image_bgr)
