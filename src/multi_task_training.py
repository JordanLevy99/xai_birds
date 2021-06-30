import os
import torch
import pandas as pd
#from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models
import PIL.Image as Image
import torch.nn as nn
import torchvision
import torch.optim as optim
import sys
from captum.attr import GuidedGradCam
#import cv2
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap
from captum.attr import visualization as viz

# sys.path.insert(0, '../src')
from bird_dataset import *
from XAI_birds_dataloader import *
from tqdm import tqdm
from models.multi_task_model import *
from XAI_birds_dataloader import *
from XAI_BirdAttribute_dataloader import *
import pickle

from download import *

import shutil


class MultiTaskTraining:
    def __init__(self, model, train_set, val_set, loss_func, data_dir='', batch_size=1, shuffle=True, lr=0.001, momentum=0.9, early_stopping=True, patience=7, epochs=50, print_freq=100):
        self.model = model
        self.train_set = train_set
        self.batch_size = batch_size
        self.epochs = epochs
        self.cur_epoch = 0
        self.patience = patience
        self.print_freq = print_freq
        self.data_dir = data_dir
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available(): self.model.cuda()
        print(self.device)
        self.task_str = '__'.join(list(train_set.dataset.class_dict.keys()))
        
        self.trainloader = DataLoader(train_set, batch_size=self.batch_size, shuffle=shuffle)
        self.valloader = DataLoader(val_set, batch_size=self.batch_size, shuffle=shuffle)
        self.loss_func = loss_func.to(self.device)
        self.lr = lr
        self.momentum = momentum
        self.early_stopping = early_stopping
        self.opt = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)
        
        self.avg_train_losses = []
        self.avg_val_losses = []
        self.val_acc = []
        self.best_score = 0
        self.best_epoch = 0
        self.best_score_lst = []
        self.best_model = None
    
    
    def train(self):
        for epoch in range(self.epochs):
            self.cur_epoch = epoch
            running_loss = 0.0
            for i, data in tqdm(enumerate(self.trainloader, 0)):
                # Get the inputs.
        #         print("LABELS:",data['labels'])
                inputs, labels = data['image'], torch.LongTensor(data['labels'])

                # Move the inputs to the specified device.
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Zero the parameter gradients.
                self.opt.zero_grad()

                # Forward step.
                outputs = self.model(inputs)
                # print('outputs came')
                # print("OUTPUTS:",outputs) 
                # break
                loss = self.loss_func(outputs, labels)
                # print(loss)
                # Backward step.
                loss.backward()

                # Optimization step (update the parameters).
                self.opt.step()

                # Print statistics.
                running_loss += loss.item()
        #         print('running loss', running_loss)
        #         print(outputs)
                if i % self.print_freq == self.print_freq - 1: # Print every several mini-batches.
                    avg_loss = running_loss / self.print_freq
                    print('[epoch: {}, i: {:5d}] avg mini-batch loss: {:.3f}'.format(
                        epoch, i, avg_loss))
                    running_loss = 0.0

            self.avg_train_losses.append(avg_loss)
            self.model.eval()
            with torch.no_grad():
                num_correct=0
                val_losses = []
                data_iter = iter(self.valloader)
                for val_data in tqdm(data_iter):
                    val_inputs, val_labels = val_data['image'].cuda(), torch.LongTensor(val_data['labels']).cuda()
                    val_outputs = self.model(val_inputs)
                    self.opt.zero_grad() # zero the parameter gradients
                    val_predicted = [torch.max(i, 1)[1] for i in val_outputs]
                    num_correct += sum(np.array(val_labels.cpu()).flatten()==np.array(val_predicted[0].cpu()).flatten())
                    val_losses.append(self.loss_func(val_outputs, val_labels).item())
                acc = num_correct/(len(data_iter)*self.batch_size*len(val_labels))
                self.val_acc.append(acc)
                if acc > self.best_score: 
                    self.best_score = acc
                    self.best_model = self.model
                    self.best_epoch = epoch
                self.best_score_lst.append(self.best_score)
                if epoch > self.patience and pd.Series(self.best_score_lst[-self.patience:]).nunique() == 1:
                    print(f'Early Stopping at Epoch {epoch} with Validation Accuracy: {self.best_score}')
                    self.model = self.best_model
                    #epoch += (self.epochs-epoch) # finishes the outer loop
                    self.epochs = epoch
                    break
                print('Validation accuracy:',acc)
                print('Average validation loss:',np.mean(val_losses))
                self.avg_val_losses.append(np.mean(val_losses))
            self.model.train()
        print('Finished Training.')
    
    def eval_model(self):
        self.model.eval()
        with torch.no_grad():
            num_correct=0
            val_losses = []
            data_iter = iter(self.valloader)
            for val_data in data_iter:
                val_inputs, val_labels = val_data['image'].cuda(), torch.LongTensor(val_data['labels']).cuda()
                val_outputs = self.model(val_inputs)
                self.opt.zero_grad() #zero the parameter gradients
                val_predicted = [torch.max(i, 1)[1] for i in val_outputs]
                num_correct += sum(np.array(val_labels.cpu()).flatten()==np.array(val_predicted[0].cpu()).flatten())
                val_losses.append(self.loss_func(val_outputs, val_labels).item())
            acc = num_correct/(len(data_iter)*self.batch_size*len(val_labels))
        return acc
    
    def plot_train_val_loss(self, f_label_size=15, f_title_size=18, f_ticks_size=11):
        plt.plot(self.avg_train_losses)
        plt.plot(self.avg_val_losses)
        plt.xlabel('Epochs', fontsize=f_label_size)
        plt.ylabel('Loss', fontsize=f_label_size)
        plt.title('Training and Validation Loss', fontsize=f_title_size)
        plt.legend(['Training Loss', 'Validation Loss']);
        plt.savefig(f'{self.data_dir}figures/{self.task_str}_{self.best_epoch}_epoch_train_val_loss.png', dpi=800)
    
    def save_model(self):
        fpath = f'{self.data_dir}models/{self.task_str}_{self.cur_epoch}_epoch_state_dict.pth'
        torch.save(self.model.state_dict(), fpath)
        print(f'Model saved at {fpath}')
    
    def save_object(self):
        with open(f'{self.data_dir}models/training_objects/{self.task_str}_{self.cur_epoch}_obj.pkl', 'wb') as f:
            pickle.dump(self, f)

#if __name__ == '__main__':
#    bd = BirdDataset(preload=True, attr_file='attributes')
#    vgg16 = models.vgg16_bn(pretrained=True)
#    trans = transforms.Compose([
#    transforms.Resize((224, 224)),
#    transforms.ToTensor()
#])
#    train_bird_dataset = Bird_Attribute_Loader(bd, attrs=['has_bill_shape'], verbose=False, species=False, transform=trans, train=True)
#    val_bird_dataset = Bird_Attribute_Loader(bd, attrs=['has_bill_shape'], verbose=False, species=False, transform=trans, train=False, val=True)
#    model = MultiTaskModel(vgg16, train_bird_dataset)
#    loss_func = MultiTaskLossWrapper()
#    mtt = MultiTaskTraining(model, train_bird_dataset, val_bird_dataset, loss_func, epochs=50, lr=0.0001, patience=7)
    
#    mtt.train()
#    mtt.plot_train_val_loss()
#    mtt.save_model()
#    mtt.save_object()
