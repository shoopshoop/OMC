'''
This file is derived from the following source:
- https://github.com/yaoxx340/MonkeyDataset/blob/main/OpenMonkey.py

Modifications were required to accomodate the annotation format used in
the OpenMonkeyCompetetion dataset
'''

import json
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
import copy
import os
import sys

import torchvision
import torch
from torch.utils.data import Dataset

def _isArrayLike(obj):
  return hasattr(obj, '__iter__') and hasattr(obj, '__len__')
    
#REye-LEye-Nose-Head-Neck-RShoulder-RElbow-RHand-LShoulder-LElbow-LHand-Hip-RKnee-RFoot-LKnee-LFoot-Tail
#  0   1    2    3    4       5       6      7      8        9      10   11   12    13   14     15   16
colors = [
            (255, 153, 204),  # REye-Nose
            (255, 153, 204),  # LEye-Nose    
            (153, 51, 255),  # nose-head
            (51, 51, 255),  # head-neck
            (204, 102, 0),  # neck-RShoulder
            (230, 140, 61),  # RShoulder-RElbow
            (255, 178, 102),  # RElbow-RHand
            (255, 102, 102),  # neck-LShoulder
            (255, 179, 102),  # LShoulder-LElbow
            (255, 255, 102),  # LElbow-LHand
            (51, 153, 255),  # neck-hip
            (102, 204, 0),  # hip-RKnee
            (204, 255, 153),  # RKnee-RFoot
            (0, 204, 102),  # hip-LKnee
            (102, 255, 178),  # LKnee-LFoot
            (102, 255, 255),  # hip-tail
        ]
I   = np.array([1,2,3,4,5,6,4,8,9,4,11,12,11,14,11]) 
J   = np.array([2,0,4,5,6,7,8,9,10,11,12,13,14,15,16])

class OpenMonkeyDataset(Dataset):
  def __init__(self, root=None, mode="train",transform=None, target_transform=torch.tensor, scaled_size = (224,224)):
    # load dataset
    # self.dataset,self.landmarks,self.specs,self.imgs = dict(),dict(),dict(),dict()

    assert mode in ["train", "val", "test"]
    self.mode = mode

    self.root = root
    self.annfilepath = os.path.join(root, mode+"_annotation.json") if mode != "test" else None
    self.imgpath = os.path.join(root, mode)

    if not self.annfilepath == None:
      dataset = json.load(open(self.annfilepath, 'r'))
      assert type(dataset)==dict, 'annotation file format {} not supported'.format(type(dataset))
      self.dataset = dataset
      self.createIndex()
      print('Annotations loaded.')

    self.scaled_size = scaled_size

    self.transform = transform
    self.target_transform = target_transform
          
  def createIndex(self):
    # create index
    landmarks, specs, imgs, bbox = {}, {}, {}, {}
    if 'data' in self.dataset:
      for i, sample in enumerate(self.dataset['data']):
        landmarks[i] = sample['landmarks']
        imgs[i] = sample['file']
        specs[i] = sample['species']
        bbox[i] = sample['bbox']
    # create class members
    self.landmarks = landmarks
    self.imgs = imgs
    self.specs = specs
    self.bbox = bbox
  
  def __len__(self):
    return len(self.imgs)

  def __getitem__(self, idx):
    # Read image file
    img_path = os.path.join(self.imgpath, self.imgs[idx])
    image = torchvision.io.read_image(img_path)
    # Crop Image to bounding box
    bbox =  self.bbox[idx]
    
    image = torchvision.transforms.functional.crop(image, bbox[1], bbox[0], bbox[3], bbox[2])
    #resize image to scaled_size (256x256)
    image = torchvision.transforms.Resize(size=self.scaled_size)(image).float()
    
    # get landmark locations
    label = self.landmarks[idx]
    # rescale to fit bounding box
    for i in range(17):
        label[2*i] = (label[2*i] - bbox[0])*self.scaled_size[0]/(bbox[2])
        label[2*i+1] = (label[2*i+1] - bbox[1])*self.scaled_size[1]/(bbox[3])
    
    # Optional transforms
    if self.transform:
        image = self.transform(image)
    if self.target_transform:
        label = self.target_transform(label)
    
    return image, label



if __name__ == "main":
  datapath = "C:/Users/jiang/Documents/Data/open-monkey"

  training_data = OpenMonkeyDataset(datapath, mode="train")
  val_data = OpenMonkeyDataset(datapath, mode="val")

  image, label = training_data.__getitem__(1)
