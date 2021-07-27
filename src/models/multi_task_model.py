'''With inspiration from https://towardsdatascience.com/multi-task-learning-with-pytorch-and-fastai-6d10dc7ce855 and https://towardsdatascience.com/tuning-a-multi-task-fate-grand-order-trained-pytorch-network-152cfda2e086'''

import torch
import os
import PIL.Image as Image
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models
import sys
import torch.nn as nn
import torch.nn.functional as F


sys.path.insert(0, '../src')
from bird_dataset import *


class MultiTaskModel(nn.Module):
    """
    Creates a MTL model for the two attributes: 'has_bill_shape' and 'has_wing_color'
    """
    def __init__(self, model, dataset, ps=0.5):
        super(MultiTaskModel,self).__init__()
        
#         num_feats = model.classifier[6].in_features
#         features = list(model.classifier.children())[:-1]
#         features.extend([nn.Linear(num_feats, len(train_bird_dataset.class_dict))])
#         vgg16.classifier = nn.Sequential(*features) # Replace the model classifier
        
        self.encoder = model        #fastai function that creates an encoder given an architecture
        # print(dataset.class_dict)
        self.fc1 = nn.Linear(1000, 512)
        nn.init.xavier_normal_(self.fc1.weight)

        self.fc2 = nn.Linear(512, 256)
        nn.init.xavier_normal_(self.fc2.weight)
        # self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 256)
        # self.bn3 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.25)

        self.dataset = dataset
        self.fc_dict = {}
        for key, value in self.dataset.class_dict.items():
            setattr(self, key, nn.Linear(256, len(value)))
        #     setattr(self, key, nn.Linear(1000, len(value)))

            # print(f'num_unique_vals in {key}: {len(value)}')
            self.fc_dict[key] = getattr(self, key)
            nn.init.xavier_normal_(getattr(self, key).weight)

            # self.eval(f'{key}') = nn.Linear(1000, len(value))
        # self.fc1 = nn.Linear(1000, 9)
        # self.fc2 = nn.Linear(1000, 15)

    def forward(self,x):

#         x = nn.ReLU(self.encoder(x))
        x = self.encoder(x)
        # print(x.shape)
        x = self.dropout(F.relu(self.fc1(x)))
        # print(x.shape)
        x = self.dropout(F.relu(self.fc2(x)))
        # print(x.shape)
        x = self.dropout(F.relu(self.fc3(x)))
        # print(x.shape)
        # bill_shape = self.fc1(x)
        # wing_color = self.fc2(x)
        ret_vals = [F.softmax(self.fc_dict[key](x), dim=1) for key in self.dataset.class_dict]
#         if len(ret_vals) == 1:
#             return ret_vals[0]
        return np.array(ret_vals)
    
class MultiTaskLossWrapper(nn.Module):
    '''
    Multi-Task loss for two attributes only
    '''
    def __init__(self, dataset):
        super(MultiTaskLossWrapper, self).__init__()
#         self.task_num = task_num
#         self.log_vars = nn.Parameter(torch.zeros((task_num)))
        self.dataset = dataset
    def forward(self, preds, labels):

#         print("PREDICTIONS:",preds)
#         print("LABELS:",labels)
# #         print(labels)
#         print(labels[0].shape)
#         print(preds[0].shape)
        labels = labels.reshape(len(self.dataset.class_dict),-1)
        loss = sum([nn.CrossEntropyLoss()(preds[i], labels[i]) for i in range(len(preds))])

        #         loss = sum([nn.CrossEntropyLoss()(preds[i].reshape(1, -1), labels[i].reshape(len(self.dataset.class_dict))) for i in range(len(preds))])

#         loss0 = nn.CrossEntropyLoss()(preds[0].reshape(1, -1), labels[0].reshape(1))
#         loss1 = nn.CrossEntropyLoss()(preds[1].reshape(1, -1), labels[1].reshape(1))
        return loss