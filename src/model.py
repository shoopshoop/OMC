import numpy as np
import pandas as pd
import cv2
from matplotlib import pyplot as plt
import os
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision

# size of image fed into CNN
scaled_size = (256,256)

# filepaths
train_json_file = 'train_annotation.json'
train_img_dir = 'train/'
val_json_file = 'val_annotation.json'
val_img_dir = 'val/'

# OpenMonkeyChallenge data pipeline
class MonkeyDataset(Dataset):
    def __init__(self, img_dir, js_file, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.img_labels = pd.read_json(js_file, orient='split')
        
        # convert landmark locations to pytorch tensors
        self.img_labels['landmarks'] = self.img_labels['landmarks'].apply(torch.tensor)
        
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        return self.img_labels.shape[0]

    def __getitem__(self, idx):
        # Read image file
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = cv2.imread(img_path)
        # Crop Image to bounding box
        bbox =  self.img_labels.iloc[idx, 2]
        x1 = bbox[0]
        y1 = bbox[1]
        x2 = x1 + bbox[2]
        y2 = y1 + bbox[3]
        image = image[y1:y2, x1:x2]
        #resize image to 500x500
        image = torch.from_numpy(cv2.resize(image, scaled_size, interpolation = cv2.INTER_LINEAR)).float()
        
        # get landmark locations
        label = self.img_labels.iloc[idx, 3]
        # rescale to fit bounding box
        for i in range(17):
            label[2*i] = (label[2*i] - x1)*scaled_size[0]/(x2-x1)
            label[2*i+1] = (label[2*i+1] - y1)*scaled_size[1]/(y2-y1)
        
        # Optional transforms
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label
        
training_data = MonkeyDataset(train_img_dir, train_json_file)
val_data = MonkeyDataset(val_img_dir, val_json_file)

batch_size = 32

train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

