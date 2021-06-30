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

class Bird_Attribute_Loader(Dataset):
    '''
    Loads in x amount of attributes in bd.attributes
    self.attrs:str or list -- loads in x attributes into a list of labels for Pytorch
    
    
    change this class or make a new one where you can use key of image directly as an attribute...
    '''
    def __init__(self, bd:BirdDataset, attrs, verbose, subset=False, species=False, filter_b_w=True, transform=None, random_seed=42):
#         XAI_Birds_Dataset.__init__(self, bd, subset=subset, transform=transform, train=train, val=val, random_seed=random_seed)
         # true and val arguments deprecated...s
#         print(f'num_images: {len(self.images)}')
        self.bd = bd
        self.attrs = attrs
        self.verbose=verbose
        self.transform = transform
        self.filter_b_w = filter_b_w
#         self.bd.images[key]['image_id'] = key
        self.images = [self.bd.images[key] for key in self.bd.images]
        if self.attrs is not None: self.attrs = sorted(self.attrs)
        self.species = species
        self.class_dict = self._set_classes_attributes()
        if self.attrs is not None: self.images, self.attr_indices = self._filter_images_by_attributes()
        if self.filter_b_w: self.images = self._filter_images_b_w()
#         if self.subset: self.images = self._filter_images_by_species()
            
    def __len__(self):
        return len(self.images)
    
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
        elif self.species:
            label = self.images[idx]['class_label']
            sample = {'image': image, 'labels':label}
        if self.transform:
            sample['image'] = self.transform(sample['image'])
        if self.verbose:
            print('image_id: ', self.images[idx]['image_id'])
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
        elif self.species: 
            attrs_dict = dict()
            attrs_dict['species'] = dict(zip(range(len(self.bd.species)), self.bd.species.values()))
        return attrs_dict
    
    def _filter_images_b_w(self):
        bw_ids = [448, 1401, 3617, 3619, 3780, 5029, 5393, 6321]
        filt_images = []
        for img in self.images:
            if img['image_id'] not in bw_ids: filt_images.append(img)
        return filt_images
#     def _filter_images_by_species(self):
#         for i
        
#         return filt_images
    
    def _filter_images_by_attributes(self):
        filt_images = []
        attr_indices = []
        for i, img in enumerate(self.images):
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