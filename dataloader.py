import torch
from torch import nn
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
from typing import List

def read_filetxt(file_txt):
    results = {'label': [], 'box': []}
    with open(file_txt, "r", newline='') as file:
        for line in file.readlines():
            values = line.strip().split()
            if int(values[0]) >= 4:
                values[0] = int(values[0]) - 4
            
            results['label'].append(int(values[0]))
            results['box'].append([float(value) for value in values[1: 5]])

    return results

def get_file_path(folder):
    data = []
    sub_folders = sorted(os.listdir(folder))
    sub_folder_paths = [os.path.join(folder, sub_folder) for sub_folder in sub_folders]
    # Kiểm tra tập train hay test
    if os.listdir(sub_folder_paths[0]) is not None:
        # Duyêt qua tập daytime và nighttime
        for sub_folder_path in sub_folder_paths:
            data_paths = sorted(os.listdir(sub_folder_path))
            data_path_file = [os.path.join(sub_folder_path, data_path) for data_path in data_paths]
            data.extend(data_path_file)

        return {'folder': 'train', 'file_train': data}
    else:
        return {'folder': 'test', 'file_test': sub_folder_paths}


class TrafficVehicle(Dataset):
    def __init__(self, folder: str, transforms=None):
        self.data = get_file_path(folder)
        self.transforms = transforms

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
        self.class_name = {0: 'motocycle', 1: 'car', 2: 'coach', 3: 'container truck'}
    def load_image(self, index):
        img = Image.open(self.image[index])
        img = np.array(img)
        return img
    def __len__(self):
        return len(self.image)
    def __getitem__(self, index: int):
        img = self.load_image(index)
        target = read_filetxt(self.txt[index])
        

        if self.transforms is not None:
            img = self.transforms(img, target['box'])

            return img, target
        else:
            return img, target




