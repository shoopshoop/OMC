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

def guassian_kernel(size_w, size_h, center_x, center_y, sigma):
    gridy, gridx = np.mgrid[0:size_h, 0:size_w]
    D2 = (gridx - center_x) ** 2 + (gridy - center_y) ** 2
    return np.exp(-D2 / 2.0 / sigma / sigma)

class OpenMonkeyDataset(Dataset):
  def __init__(self, root=None, mode="train", transform=None, scaled_size = (224,224), stride=1, sigma=10.0):
    # load dataset
    # self.dataset,self.landmarks,self.specs,self.imgs = dict(),dict(),dict(),dict()

    assert mode in ["train", "val", "test"]
    self.mode = mode

    self.root = root
    self.annfilepath = os.path.join(root, mode+"_annotation.json") if mode != "test" else os.path.join(root, "test_prediction.json")
    self.imgpath = os.path.join(root, mode)

    if not self.annfilepath == None:
      dataset = json.load(open(self.annfilepath, 'r'))
      assert type(dataset)==dict, 'annotation file format {} not supported'.format(type(dataset))
      self.dataset = dataset
      self.createIndex()
      print('Annotations loaded.')

    self.scaled_size = scaled_size

    self.transform = transform

    # Heatmap parameter
    self.stride = stride
    self.sigma = sigma
          
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
    #resize image to scaled_size (224x224)
    image = torchvision.transforms.Resize(size=self.scaled_size)(image).float()
    
    # Optional transforms
    if self.transform:
        image = self.transform(image)

    if self.mode != "test":
      # get landmark locations
      label = self.landmarks[idx]
      # rescale to fit bounding box
      for i in range(17):
          label[2*i] = (label[2*i] - bbox[0])*self.scaled_size[0]/(bbox[2])
          label[2*i+1] = (label[2*i+1] - bbox[1])*self.scaled_size[1]/(bbox[3])
      
      label = self.label_to_heatmap(label)
      
      return image, label
    else:
      return image, idx


  def label_to_heatmap(self, kpt):
    H, W = self.scaled_size
    heatmap = np.zeros((17 + 1, H // self.stride, W // self.stride), dtype=np.float32)
    for i in range(17):
        x = int(kpt[2*i]) * 1.0  / self.stride
        y = int(kpt[2*i+1]) * 1.0  / self.stride
        heat_map = guassian_kernel(size_h=H // self.stride , size_w=W // self.stride, center_x=x, center_y=y, sigma=self.sigma)
        heat_map[heat_map > 1] = 1
        heat_map[heat_map < 0.0099] = 0
        heatmap[i + 1, :, :] = heat_map

    heatmap[0, :, :] = 1.0 - np.max(heatmap[1:, :, :], axis=0)  # for background
    heatmap = torch.tensor(heatmap)
    
    return heatmap

  def save_landmarks(self, idx_list, landmarks):
    for idx, landmark in zip(idx_list, landmarks):
      # mapping landmarks to original images
      bbox = self.bbox[idx]
      for i in range(17):
          landmark[2*i] = landmark[2*i]/self.scaled_size[0]*bbox[2]+bbox[0]
          landmark[2*i+1] = landmark[2*i+1]/self.scaled_size[1]*bbox[3]+bbox[1]
      # save landmarks
      self.landmarks[idx]  = landmark

  def write_landmarks_to_file(self, path):
    # update json dataset
    if 'data' in self.dataset:
      for i in range(len(self.dataset['data'])):
        self.dataset['data'][i]['landmarks'] = self.landmarks[i] 
    # writing to sample.json
    pred_path = os.path.join(path, "test_prediction.json")
    with open(pred_path, "w") as fd:
      fd.write(json.dumps(self.dataset, indent=4))

  def showImgs(self, imgs):
    for i in range(len(imgs)):
      img = cv2.imread(os.path.join(self.imgpath, imgs[i]))
      plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
      plt.axis('off')
      plt.show()
      
  def showBbox(self, imgs, bboxs):
    for i in range(len(imgs)):
      img = cv2.imread(os.path.join(self.imgpath, imgs[i]))
      x1 = bboxs[i][0]
      y1 = bboxs[i][1]
      x2 = x1 + bboxs[i][2]
      y2 = y1 + bboxs[i][3]
      img = cv2.rectangle(img, (x1,y1), (x2,y2),(0,255,0), 2)
      plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
      plt.axis('off')
      plt.show()

  def showAnns(self, imgs, landmarks, bboxs=None, keypoints=False):
    for i in range(len(imgs)):
      img = cv2.imread(os.path.join(self.imgpath, imgs[i]))
      if bboxs != None:
        x1 = bboxs[i][0]
        y1 = bboxs[i][1]
        x2 = x1 + bboxs[i][2]
        y2 = y1 + bboxs[i][3]
        img = cv2.rectangle(img, (x1,y1), (x2,y2),(0,255,0), 2)
      x = landmarks[i][::2]
      y = landmarks[i][1::2]
      for j in range(len(I)):
        cv2.line(img,(int(round(x[I[j]])),int(round(y[I[j]]))),(int(round(x[J[j]])),int(round(y[J[j]]))),colors[j], 2)
      if keypoints == True:
        for j in range(len(x)):
          cv2.circle(img, (int(round(x[j])),int(round(y[j]))), 5, (255,255,255), -1)
          cv2.circle(img, (int(round(x[j])),int(round(y[j]))), 3, (0,0,0), -1)
      plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
      plt.axis('off')
      plt.show()


if __name__ == "main":
  datapath = "C:/Users/jiang/Documents/Data/open-monkey"

  training_data = OpenMonkeyDataset(datapath, mode="train")
  # val_data = OpenMonkeyDataset(datapath, mode="val")

  image, label = training_data.__getitem__(1)
