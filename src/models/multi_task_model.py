'''With inspiration from https://towardsdatascience.com/multi-task-learning-with-pytorch-and-fastai-6d10dc7ce855'''

import torch
import os
import PIL.Image as Image
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models
import sys
import torch.nn as nn


sys.path.insert(0, '../src')
from bird_dataset import *


class MultiTaskModel(nn.Module):
    """
    Creates a MTL model for the two attributes: 'has_bill_shape' and 'has_wing_color'
    Goal: get a list of attributes and create a list of layers according to the inputs, adjust linear layer based on observed number of unique values per task (e.g. bill_shape has 9 unique values and wing_color has 15)
    """
    def __init__(self, model, bd, ps=0.5):
        super(MultiTaskModel,self).__init__()
#         num_feats = model.classifier[6].in_features
#         features = list(model.classifier.children())[:-1]
#         features.extend([nn.Linear(num_feats, len(train_bird_dataset.class_dict))])
#         vgg16.classifier = nn.Sequential(*features) # Replace the model classifier
        
        self.encoder = model        #fastai function that creates an encoder given an architecture
        
        self.bill_shape = nn.Linear(1000, 9)    
        self.wing_color = nn.Linear(1000, 15)

    def forward(self,x):

#         x = nn.ReLU(self.encoder(x))
        x = self.encoder(x)
        
        species = self.species(x)
        bill_shape = self.bill_shape(x)
        wing_color = self.wing_color(x)

        return bill_shape, wing_color
    
class MultiTaskLossWrapper(nn.Module):
    '''
    Multi-Task loss for two attributes only
    '''
    def __init__(self):
        super(MultiTaskLossWrapper, self).__init__()
#         self.task_num = task_num
#         self.log_vars = nn.Parameter(torch.zeros((task_num)))

    def forward(self, preds, labels):

#         print("PREDICTIONS:",preds[0])
#         print("LABELS:",labels[0])
        loss0 = nn.CrossEntropyLoss()(preds[0].reshape(1, -1), labels[0].reshape(1))
        loss1 = nn.CrossEntropyLoss()(preds[1].reshape(1, -1), labels[1].reshape(1))
        return loss0+loss1