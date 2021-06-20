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
from XAI_birds_dataloader import *

class Bird_Attribute_Loader(XAI_Birds_Dataset):
    '''
    Loads in x amount of attributes in bd.attributes
    self.attrs:str or list -- loads in x attributes into a list of labels for Pytorch
    
    
    change this class or make a new one where you can use key of image directly as an attribute...
    '''
    def __init__(self, bd:BirdDataset, attrs, subset=True, species=False, transform=None, train=True, val=False, random_seed=42):
        XAI_Birds_Dataset.__init__(self, bd, subset=subset, transform=transform, train=train, val=val, random_seed=random_seed)
#         print(f'num_images: {len(self.images)}')
        self.attrs = sorted(attrs)
        self.species = species
        self.class_dict = self._set_classes_attributes()
        self.images, self.attr_indices = self._filter_images_by_attributes()

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_path = os.path.join(self.bd.img_dir, self.images[idx]['filepath'])
        image = Image.open(img_path)
        if isinstance(self.attrs, str):
            attr = self.images[idx]['attributes'][self.attr_indices[idx]]
            label = self.class_dict[attr]
            sample = {'image': image, 'label':label}
        elif isinstance(self.attrs, list):
            attrs = [self.images[idx]['attributes'][i] for i in self.attr_indices[idx]]
#             print(attrs)
#             print("ATTRIBUTES TO INDEX:",attrs)
            labels = [self.class_dict[attr.split('::')[0]][attr] for attr in attrs]
            if self.species:
#                 print(self.species)
#                 print(self.class_dict['species'])
#     self.images[idx]['class_label']
                labels.append(self.images[idx]['class_label'])
            sample = {'image': image, 'labels':labels}
        if self.transform:
            sample['image'] = self.transform(sample['image'])
        return sample

    def _set_classes_attributes(self):
        pd_attr = pd.Series(self.bd.attributes)
        if isinstance(self.attrs, str):
            attrs_dict = pd_attr[pd_attr.str.contains(self.attrs)].to_dict()
            class_dict = dict(zip(attrs_dict.values(), range(len(attrs_dict))))
        elif isinstance(self.attrs, list):
            attrs_dict = dict()
            for attribute in self.attrs:
                attr_dict = pd_attr[pd_attr.str.contains(attribute)].to_dict()
                attrs_dict[attribute] = dict(zip(attr_dict.values(), range(len(attr_dict))))
#             print(f'ATTRS DICT: {attrs_dict}')
#             class_dict = dict(zip(attrs_dict.values(), range(len(attrs_dict))))
        if self.species: attrs_dict['species'] = dict(zip(range(len(self.bd.species)), self.bd.species.values()))
        return attrs_dict
    
    def _filter_images_by_attributes(self):
        filt_images = []
        attr_indices = []
        for img in self.images:
            check=0
            attr_index = []
            attrs = []
            
            for idx, attr in enumerate(img['attributes']):
                if isinstance(self.attrs, str):
                    if self.attrs in attr:
                        filt_images.append(img)
                        attr_indices.append(idx)
                        break
                elif isinstance(self.attrs, list):
                    for attribute in self.attrs:
                        if attribute in attr:
                            check+=1
#                             print(attribute)
                            attr_index.append(idx)
                            attrs.append(attribute)
                else: raise(ValueError, "self.attrs must be a string or a list of strings")
            if check==len(self.attrs): # only append to images/indices if all attributes in self.attrs are in the images attributes
                unique_attrs = [attr.split('::')[0] for attr in attrs]
#                 print(unique_attrs)
                if len(set(unique_attrs)) == len(self.attrs):
#                     if img not in filt_images:
                    filt_images.append(img)
    #                     else: pass
    #                             print('img already herre')
    #                         print('wowie')
                    attr_indices.append((list(attr_index)))
#                 else: print("Attributes aren't unique:",attr)
        return filt_images, attr_indices