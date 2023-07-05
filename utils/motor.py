
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image
from torchsummary import summary
import torch
from torch.nn import Module
import torch.nn as nn
from sklearn.model_selection import train_test_split
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import random_split, Dataset, DataLoader
from torch.optim import SGD, Adam
import time
import os



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data(dir_path, class_name='all'):
  """
  Availlable class names are ('8bars', 'inner', 'outer', 'ball', '4bars', 'healthy', '1bar')
  """

  encoder = {'8bars':0, 'inner':1, 'outer':2, 'ball':3, '4bars':4, 'healthy':5, '1bar':6}
  inv_encoder = {v:k for k,v in encoder.items()}
  #print(dir_path)
  img_paths = []
  labels = []
  for dirpath, dirnames, filenames in os.walk(dir_path):
          #print(dirpath)
          # Loop over the files and do something with them
          for filename in filenames:
              #print(filename)
              #print('File:', os.path.join(dirpath, filename))
              path = os.path.join(dirpath, filename)
              l = path.split('/')[-2]
              if class_name == 'all':
                img_paths.append(path)
                labels.append(encoder[l])
              elif l == class_name:
                img_paths.append(path)
                labels.append(encoder[l])

  # Convert labels to a NumPy array
  labels = np.array(labels)

  # Extract the unique values of labels and the number of each label
  unique_labels, counts = np.unique(labels, return_counts=True)

  # Print the unique values of labels and the number of each label
  for label, count in zip(unique_labels, counts):
      print(f"Label {inv_encoder[label]}: {count} samples")

  print(55*"-")
  return img_paths, labels



class MotorDataset(Dataset):
    def __init__(self, images_p, labels, transform=None):
        #self.root_dir = root_dir
        self.transform = transform
        #self.images = []
        #self.targets = []
        # Loop through all subdirectories (one per subject)
        #print(current_label, uniq_labels[current_label])
                
        self.images = images_p
        self.targets = labels
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path, label = self.images[idx], self.targets[idx]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img.to(device), torch.tensor(label, dtype=torch.int64).to(device)



def get_test_images(dir_path, class_name='all', transform='default', large_size=False, get_labels=False):
    img_paths, labels = load_data(dir_path, class_name)
    if transform == 'default':
        if large_size:
          transform = transforms.Compose([
                  transforms.Resize(128),
                  transforms.ToTensor(),
                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
          ])
        else:
          transform = transforms.Compose([
                  transforms.Resize(32),
                  transforms.ToTensor(),
                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
          ])
    img_tensors = []
    for img_path in img_paths:
        img = Image.open(img_path)
        img_tensor = transform(img)
        img_tensors.append(img_tensor)
    img_tensors = torch.stack(img_tensors, dim=0)

    if get_labels:
      return img_tensors, torch.tensor(labels, dtype=torch.int64)
    return img_tensors



def getMotorDataLoader(dir_path, batch_size, class_name='all', transform='default'):

  if transform == 'default':
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                          ])

  img_paths, labels = load_data(dir_path, class_name)

  ds = MotorDataset(img_paths, labels, transform)
  #print(os.getcwd())
  #print(len(ds))
  loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
  
  return loader
