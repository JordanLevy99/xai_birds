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
# from XAI_birds_dataloader import *

class Bird_Attribute_Loader(Dataset):
    '''
    Loads in x amount of attributes in bd.attributes
    self.attrs:str or list -- loads in x attributes into a list of labels for Pytorch
    
    
    change this class or make a new one where you can use key of image directly as an attribute...
    '''
    def __init__(self, bd:BirdDataset, attrs, verbose, subset=False, species=False, filter_b_w=True, transform=None, random_seed=42, test=False):
#         XAI_Birds_Dataset.__init__(self, bd, subset=subset, transform=transform, train=train, val=val, random_seed=random_seed)
         # true and val arguments deprecated...s
#         print(f'num_images: {len(self.images)}')
        self.bd = bd
        self.attrs = attrs
        self.verbose=verbose
        self.transform = transform
        self.filter_b_w = filter_b_w
#         self.bd.images[key]['image_id'] = key
        if self.filter_b_w: self.images = self._filter_images_b_w()
        self.bw_ids =  [448, 1401, 3617, 3619, 3780, 5029, 5393, 6321]
        self.images = [self.bd.images[key] for key in self.bd.images if 'class_label' in self.bd.images[key] and self.bd.images[key]['image_id'] not in self.bw_ids]
        self.test = test
        if self.test: self.images=self.images[:300]
        if self.attrs is not None: 
            self.attrs = sorted(self.attrs)
            self.num_tasks = len(attrs)
        else: self.num_tasks = 0
        self.species = species
        if self.species: self.num_tasks += 2 # now accounting for family as well
        self.class_dict = self._set_classes_attributes()
        if self.attrs is not None: self.images, self.attr_indices = self._filter_images_by_attributes()
#         if self.subset: self.images = self._filter_images_by_species()
        print(f'Number of images: {len(self.images)}')

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
            attrs = self.images[idx]['attrs']#i] for i in self.attr_indices[idx]]       
            labels = [self.class_dict[attr.split('::')[0]][attr] for attr in sorted(attrs)]
            if self.species:
                labels.append(self.images[idx]['class_label'])
                labels.append(self.bd.fam[self.bd.bird_to_fam[self.bd.species[self.images[idx]['class_label']]]])
            sample = {'image': image, 'labels':labels}
        elif self.species:
            labels = []
            labels.append([self.images[idx]['class_label']])
#             print(self.bd.species[self.images[idx]['class_label']])
#             print(self.bd.bird_to_fam[self.bd.species[self.images[idx]['class_label']]])
            labels.append([self.bd.fam[self.bd.bird_to_fam[self.bd.species[self.images[idx]['class_label']]]]])

            sample = {'image': image, 'labels':labels}
        if self.transform:
            sample['image'] = self.transform(sample['image'])
#             sample['labels'] = np.array(sample['labels']).flatten()
                   
            sample['labels'] = torch.LongTensor(sample['labels'])
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
            if self.species: 
                attrs_dict['species'] = dict(zip(range(len(self.bd.species)), self.bd.species.values()))
                attrs_dict['family'] = {val: key for key, val in self.bd.fam.items()}
        elif self.species: 
            attrs_dict = dict()
            attrs_dict['species'] = dict(zip(range(len(self.bd.species)), self.bd.species.values()))
            attrs_dict['family'] = {val: key for key, val in self.bd.fam.items()}
        return attrs_dict
    
    def _filter_images_b_w(self):
        bw_ids = [448, 1401, 3617, 3619, 3780, 5029, 5393, 6321]
        filt_images = []
        for img in self.bd.images.keys():
            if self.bd.images[img]['image_id'] not in bw_ids: filt_images.append(img)
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
                            attr_index.append(idx)
                            attrs.append(attr)
                else: raise(ValueError, "self.attrs must be a string or a list of strings")
            if check==len(self.attrs): # only append to images/indices if all attributes in self.attrs are in the images attributes
                unique_attrs = [attr.split('::')[0] for attr in attrs]
                if len(set(unique_attrs)) == len(self.attrs):
                    img['attrs'] = attrs
                    filt_images.append(img)
                    attr_indices.append((list(attr_index)))
#                 else: print("Attributes aren't unique:",attr)
        return filt_images, attr_indices