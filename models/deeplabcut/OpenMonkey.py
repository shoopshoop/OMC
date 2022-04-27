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

class OpenMonkey:
  def __init__(self, annotation_file=None, root=None):
    # load dataset
    self.dataset,self.landmarks,self.specs,self.imgs = dict(),dict(),dict(),dict()
    self.root = root
    if not annotation_file == None:
      dataset = json.load(open(annotation_file, 'r'))
      assert type(dataset)==dict, 'annotation file format {} not supported'.format(type(dataset))
      self.dataset = dataset
      self.createIndex()
      print('Annotations loaded.')
          
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
  
  def showImgs(self, imgs):
    for i in range(len(imgs)):
      img = cv2.imread(os.path.join(self.root, imgs[i]))
      plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
      plt.axis('off')
      plt.show()
      
  def showBbox(self, imgs, bboxs):
    for i in range(len(imgs)):
      img = cv2.imread(os.path.join(self.root, imgs[i]))
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
      img = cv2.imread(os.path.join(self.root, imgs[i]))
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

  def cropImgs(self, display=True, write=False, directory='../../data/train_cropped/'):
    cropped = []
    for i in range(len(self.imgs)):
      img = cv2.imread(os.path.join(self.root, self.imgs[i]))
      x1 = self.bbox[i][0]
      y1 = self.bbox[i][1]
      x2 = x1 + self.bbox[i][2]
      y2 = y1 + self.bbox[i][3]
      img = img[y1:y2, x1:x2,:]
      if write == True:
        cv2.imwrite(directory + self.imgs[i], img)
      else:
        cropped.append(img)
        if display == True:
          plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
          plt.axis('off')
          plt.show()
    return cropped
      
  def cropAnns(self):
    for i in range(len(self.landmarks)):
      x1 = self.bbox[i][0]
      y1 = self.bbox[i][1]
      self.landmarks[i][::2] = [j - x1 for j in self.landmarks[i][::2]]
      self.landmarks[i][1::2] = [j - y1 for j in self.landmarks[i][1::2]]

  def uncropAnns(self):
    for i in range(len(self.landmarks)):
      x1 = self.bbox[i][0]
      y1 = self.bbox[i][1]
      self.landmarks[i][::2] = [j + x1 for j in self.landmarks[i][::2]]
      self.landmarks[i][1::2] = [j + y1 for j in self.landmarks[i][1::2]]
    
  def croppedAnns(self, imgs, landmarks, bboxs, display=True):
    cropped = []
    for i in range(len(imgs)):
      img = cv2.imread(os.path.join(self.root, imgs[i]))
      x1 = bboxs[i][0]
      y1 = bboxs[i][1]
      x2 = x1 + bboxs[i][2]
      y2 = y1 + bboxs[i][3]
      img = img[y1:y2, x1:x2,:]
      x = landmarks[i][::2]
      y = landmarks[i][1::2]
      x = [j - x1 for j in x]
      y = [j - y1 for j in y]
      for j in range(len(I)):
        cv2.line(img,(int(round(x[I[j]])),int(round(y[I[j]]))),(int(round(x[J[j]])),int(round(y[J[j]]))),colors[j], 2)
      cropped.append(img)
      if display == True:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
    return cropped, [x, y]

  def csvAnns(self, filename):
    # read file
    try:
      f = open(self.root + filename, "r")
      # skip header lines
      for i in range(3):
        f.readline()
      data = f.readlines()
      f.close()
    except:
      print("Error: could not read file")
    # remove newline and split each line using comma as delimiter
    for i,ln in enumerate(data):
      sample = ln.strip().split(',')
      data[i] = list(map(lambda x: float(x), sample[1:]))
      self.imgs[i] = sample[0]
      self.landmarks[i] = [data[i][j] for j in range(len(data[i])) if (j + 1) % 3 != 0]

  def write_landmarks_to_file(self, crop=False, relative=False, filename="test_prediction.json"):
    if crop: # crop landmarks
      self.cropAnns()
    elif relative: # translate landmarks back to global, non-cropped reference frame
      self.uncropAnns()
    # update json dataset
    if 'data' in self.dataset:
      for i in range(len(self.dataset['data'])):
        self.dataset['data'][i]['landmarks'] = self.landmarks[i] 
    # writing to sample.json
    with open(filename, "w") as fd:
      fd.write(json.dumps(self.dataset, indent=4))
