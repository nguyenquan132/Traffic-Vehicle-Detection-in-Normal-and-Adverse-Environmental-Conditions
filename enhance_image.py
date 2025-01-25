import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from pathlib import Path
import requests

class ZeroDCE(nn.Module):
    def __init__(self):
        super(ZeroDCE, self).__init__()
        self.iterations = 4
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3),
                               stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3),
                               stride=1, padding=1, bias=True)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3),
                               stride=1, padding=1, bias=True)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3),
                               stride=1, padding=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=32*2, out_channels=32, kernel_size=(3, 3),
                               stride=1, padding=1, bias=True)
        self.conv6 = nn.Conv2d(in_channels=32*2, out_channels=32, kernel_size=(3, 3),
                               stride=1, padding=1, bias=True)
        self.conv7 = nn.Conv2d(in_channels=32*2, out_channels=24, kernel_size=(3, 3),
                               stride=1, padding=1, bias=True)
        
        self.relu = nn.ReLU(inplace=True)
  
    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))
        x4 = self.relu(self.conv4(x3))
        x5 = self.relu(self.conv5(torch.cat([x3, x4], dim=1)))
        x6 = self.relu(self.conv6(torch.cat([x2, x5], dim=1)))
        x_r = F.tanh(self.conv7(torch.cat([x1, x6], dim=1)))
        
        r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(x_r, 3, dim=1)
        x = x + r1 * (torch.pow(x, 2) - x)
        x = x + r2 * (torch.pow(x, 2) - x)
        x = x + r3 * (torch.pow(x, 2) - x)
        enhance_image_1 = x + r4 * (torch.pow(x, 2) - x)
        x = enhance_image_1 + r5 * (torch.pow(enhance_image_1, 2) - enhance_image_1)
        x = x + r6 * (torch.pow(x, 2) - x)
        x = x + r7 * (torch.pow(x, 2) - x)
        enhance_image = x + r8 * (torch.pow(x, 2) - x)
        return enhance_image_1, enhance_image
    
# Tải trọng số nếu chưa có
if not Path("8LE-color-loss2_best_model.pth").is_file():
    print("Downloading model weights...")
    url = "https://github.com/bsun0802/Zero-DCE/raw/refs/heads/master/train-jobs/ckpt/8LE-color-loss2_best_model.pth"
    response = requests.get(url)
    with open("8LE-color-loss2_best_model.pth", "wb") as f:
        f.write(response.content)
    print("Download completed.")
else:
    print("Model weights already exist.")



# Load mô hình và chuyển sang thiết bị phù hợp
device = "cuda" if torch.cuda.is_available() else "cpu"
model_dce = ZeroDCE()
model_dce.load_state_dict(torch.load("8LE-color-loss2_best_model.pth", map_location=device)['model'])
model_dce = model_dce.to(device)
model_dce.eval()

# Hàm xử lý ảnh
def preprocess_image(image_path):
    """
    Đọc ảnh, chuyển đổi sang tensor phù hợp để đưa vào mô hình.
    """
    image = cv2.imread(image_path)
    original_size = image.shape[:2]  # Lưu kích thước ảnh gốc (H, W)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return image_tensor, original_size

def enhance_image_and_save(model, input_tensor, original_size, device, save_path):
    """
    Tăng sáng ảnh sử dụng mô hình Zero-DCE và lưu ảnh đã xử lý giữ nguyên kích thước gốc.
    """
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        _, enhanced_image = model(input_tensor)
    enhanced_image = enhanced_image.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    enhanced_image = (enhanced_image * 255).astype(np.uint8)
    
    # Resize về kích thước gốc và lưu
    enhanced_image_resized = cv2.resize(enhanced_image, (original_size[1], original_size[0]))
    enhanced_image_bgr = cv2.cvtColor(enhanced_image_resized, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, enhanced_image_bgr)
    print(f"Enhanced image saved at: {save_path}")




# Load mô hình và chuyển sang thiết bị phù hợp
device = "cuda" if torch.cuda.is_available() else "cpu"
model_dce = ZeroDCE()
model_dce.load_state_dict(torch.load("8LE-color-loss2_best_model.pth", map_location=device)['model'])
model_dce = model_dce.to(device)
model_dce.eval()

# Hàm xử lý ảnh
def preprocess_image(image_path):
    """
    Đọc ảnh, chuyển đổi sang tensor phù hợp để đưa vào mô hình.
    """
    image = cv2.imread(image_path)
    original_size = image.shape[:2]  # Lưu kích thước ảnh gốc (H, W)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return image_tensor, original_size

def enhance_image_and_save(model, input_tensor, original_size, device, save_path):
    """
    Tăng sáng ảnh sử dụng mô hình Zero-DCE và lưu ảnh đã xử lý giữ nguyên kích thước gốc.
    """
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        _, enhanced_image = model(input_tensor)
    enhanced_image = enhanced_image.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    enhanced_image = (enhanced_image * 255).astype(np.uint8)
    
    # Resize về kích thước gốc và lưu
    enhanced_image_resized = cv2.resize(enhanced_image, (original_size[1], original_size[0]))
    enhanced_image_bgr = cv2.cvtColor(enhanced_image_resized, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, enhanced_image_bgr)
    print(f"Enhanced image saved at: {save_path}")


# Đường dẫn tới ảnh cần xử lý và nơi lưu ảnh kết quả
input_image_path = "enhanced_image.jpg"  # Thay bằng đường dẫn ảnh gốc của bạn
output_image_path = "enhanced_image_22.jpg"  # Đường dẫn lưu ảnh sau xử lý


# Tiền xử lý ảnh
input_tensor, original_size = preprocess_image(input_image_path)

# Xử lý và lưu kết quả
enhance_image_and_save(model_dce, input_tensor, original_size, device, output_image_path)