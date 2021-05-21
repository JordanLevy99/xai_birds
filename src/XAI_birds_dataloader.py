import torch
import os
import PIL.Image as Image
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models
import sys

sys.path.insert(0, '../src')
from bird_dataset import *

class XAI_Birds_Dataset(Dataset):
    def __init__(self, bd:BirdDataset, subset=True, transform=None, train=True, val=False, random_seed=42):
        self.bd = bd
        self.transform = transform
        self.subset = subset
        self.train = train
        self.val = val
        np.random.seed(random_seed)
#         if self.train: self.train_test_indices = self.bd.train_indices
#         else: self.train_test_indices = self.bd.test_indices
        
        if self.subset: 
            self.class_dict = self._set_classes('classes-subset')
            self.images = self._filter_images()
        else: 
            self.class_dict = self._set_classes('classes')
            self.images = self.bd.images
#         self.images = self.load_images()
        
#         if self.imag
        if self.train: 
            train_indices = np.random.choice(range(len(self.images)), int(len(self.images)*0.8), replace=False)
#             print(train_indices)
            self.images = np.random.choice(self.images, int(len(self.images)*0.8), replace=False)
            self.ids = [i['image_id'] for i in self.images]
            
        if self.val:
            train_indices = np.random.choice(range(len(self.images)), int(len(self.images)*0.8), replace=False)
#             print(train_indices)
            img_pd = pd.Series(dict(zip(range(len(self.images)), self.images)))
            self.images = img_pd.loc[list(set(range(len(self.images))) - set(train_indices))].tolist()
    def __len__(self):
        return len(self.images)
    

    def __getitem__(self, idx):
        '''
        For now keeping this in the 'master' XAI_Birds_Dataset class, could move this to its own 'species classification' class that inherits this class
        '''
        if torch.is_tensor(idx):
            idx = idx.tolist()
#         img_id = list(self.images.keys())[idx]
        img_path = os.path.join(self.bd.img_dir, self.images[idx]['filepath'])
        image = Image.open(img_path)
        label = self.class_dict[self.images[idx]['class_label']]
        sample = {'image': image, 'label':label}
        
        if self.transform:
            sample['image'] = self.transform(sample['image'])
            sample['label'] = torch.LongTensor([sample['label']])
        return sample
    def _set_classes(self, fname):
        '''
        For now keeping this in the 'master' XAI_Birds_Dataset class, could move this to its own 'species classification' class that inherits this class
        '''
        
        with open(f'../CUB_200_2011/{fname}.txt') as f:
            class_dict = {int(line.split(' ')[0]):i for i, line in enumerate(f.readlines())}
        return class_dict
    
    def _filter_images(self):
        list_classes = list(self.class_dict.keys())
        filt_images = []
        for key in list(self.bd.images.keys()):
            if self.bd.images[key]['class_label'] in list_classes:
                self.bd.images[key]['image_id'] = key
                filt_images.append(self.bd.images[key])
        return filt_images
    
#     def load_images(self):
#         images = {}
#         for key in self.bd.images:
#             class_label = self.bd.images[key]['class_label']
#             if class_label in list(self.class_dict.keys()) and class_label in self.train_test_indices:
#                 images[key] = self.bd.images[key] 
#         return images