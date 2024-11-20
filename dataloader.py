from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
import torch 
epsilon = 1e-6

def transform_box(box, height, width):
    
    x_center, y_center, box_width, box_height = box
    # Giới hạn biên cho các bounding box 
    x_min = max((x_center - box_width / 2) * width, 0 + epsilon)
    x_max = min((x_center + box_width / 2) * width, width - epsilon)
    y_min = max((y_center - box_height / 2) * height, 0 + epsilon)
    y_max = min((y_center + box_height / 2) * height, height - epsilon)

    return [x_min, y_min, x_max, y_max]
    

def read_filetxt(file_txt):
    results = {'labels': [], 'boxes': []}
    with open(file_txt, "r", newline='') as file:
        for line in file.readlines():
            values = line.strip().split()
            if int(values[0]) >= 4:
                values[0] = int(values[0]) - 3
            else:
                values[0] = int(values[0]) + 1
            
            results['labels'].append(int(values[0]))
            results['boxes'].append([float(value) for value in values[1: 5]])

    return results

def get_file_path(folder):
    data = []
    sub_folders = list(sorted(os.listdir(folder)))
    sub_folder_paths = [os.path.join(folder, sub_folder) for sub_folder in sub_folders]
    # Kiểm tra tập train hay test
    mode = folder.split('/')
    if mode[1] == "train":
        contents = os.listdir(sub_folder_paths[0])
        if contents:
            # Duyêt qua tập daytime và nighttime
            for sub_folder_path in sub_folder_paths:
                data_paths = list(sorted(os.listdir(sub_folder_path)))
                data_path_file = [os.path.join(sub_folder_path, data_path) for data_path in data_paths]
                data.extend(data_path_file)

            return {'folder': 'train', 'file_train': data}
        else:
            return {'folder': 'train', 'file_train': sub_folder_paths}
    else:
        return {'folder': 'test', 'file_test': sub_folder_paths} 


class TrafficVehicle(Dataset):
    def __init__(self, folder: str, transforms=None, 
                 transform_box_type: str="center"):
        """ 
            transform_box sẽ chuyển tọa độ bounding box sang để phù hợp với các model, có hai kiểu để chuyển đổi 
            là "center" hoặc "corner", mặc định là "center". Nếu transform_type="center" thì mặc định các model sử dụng bounding box 
            theo tọa độ tâm và kích thước. Ngược lại, transform_type="corner" các model sẽ sử dụng bounding box theo tọa độ góc.

            Ví dụ: trong bài toán object detection các tọa độ bounding box thường chia thành hai kiểu:
                - (x_min, y_min, x_max, y_max)
                - (x_center, y_center, width, height)
            
            Lưu ý: Do trong quá trình gán nhãn theo định dạng YOLO (x_center, y_center, width, height) nên không cần transform_box nữa.
        """
        self.data = get_file_path(folder)
        self.transforms = transforms
        self.transform_box_type = transform_box_type

        if self.data['folder'] == 'train':
            self.image, self.txt = [], []
            for train_path in self.data['file_train']:
                train_path_split = train_path.split('.')
                # Kiểm tra là txt hay là image
                if train_path_split[-1] == 'txt':
                    self.txt.append(train_path)
                else:
                    self.image.append(train_path)
        else:
            self.image = [image_path for image_path in self.data['file_test']]
        self.class_name = {1: 'motocycle', 2: 'car', 3: 'coach', 4: 'container truck'}
    def load_image(self, index):
        img = Image.open(self.image[index])
        img = np.array(img)
        return img
    def __len__(self):
        return len(self.image)
    def __getitem__(self, index: int):
        img = self.load_image(index)
        file_path = os.path.basename(self.image[index])
        if self.data['folder'] == 'train': 
            target = read_filetxt(self.txt[index])
    
            # kiểm tra transform_box có phải là center hoặc corner không
            if self.transform_box_type not in ['center', 'corner']:
                raise ValueError("transform_box_type phải là 'center' hoặc 'corner'!")
            
            # transform_box bounding box 
            if self.transform_box_type == "corner":
                height, width = img.shape[:2]
                target['boxes'] = [transform_box(box, height, width) for box in target['boxes']]
    
            # Áp dụng transforms cho image và box
            if self.transforms is not None:
                transformed = self.transforms(image=img, bboxes=target['boxes'], class_labels=target['labels'])
                img = transformed['image']
                target['boxes'] = transformed['bboxes']
                target['labels'] = transformed['class_labels']
    
            target['boxes'] = torch.tensor(target['boxes'], dtype=torch.float32)
            target['labels'] = torch.tensor(target['labels'], dtype=torch.int64)
            return img, target
        else:
            if self.transforms is not None:
                transformed = self.transforms(image=img)
                img = transformed['image']
            return img, file_path




