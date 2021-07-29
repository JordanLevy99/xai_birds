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
import copy

# sys.path.insert(0, '../src')
from bird_dataset import *
# from XAI_birds_dataloader import *
from tqdm import tqdm
from models.multi_task_model import *
# from XAI_birds_dataloader import *
from XAI_BirdAttribute_dataloader import *
import pickle

from download import *

import shutil


class MultiTaskTraining:
    def __init__(self, model, train_set, val_set, loss_func, data_dir='', batch_size=1, shuffle=True, lr=0.001, momentum=0.9, early_stopping=True, patience=7, epochs=50, print_freq=100):
        self.model = model
        self.train_set = train_set
        self.val_set = val_set
        self.num_tasks = self.train_set.dataset.num_tasks

        self.batch_size = batch_size
        self.epochs = epochs
        print("NUM TASKS: ", self.num_tasks)
        print("NUM EPOCHS: ", self.epochs)

        self.cur_epoch = 0
        self.patience = patience
        self.print_freq = print_freq
        self.data_dir = data_dir
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available(): self.model.cuda()
        print(self.device)
        self.label_dict = dict(zip(self.train_set.dataset.class_dict.keys(), range(len(self.train_set.dataset.class_dict.keys()))))
        self.task_str = '__'.join(list(self.train_set.dataset.class_dict.keys()))
        print(self.task_str)
        self.trainloader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=shuffle)
        self.valloader = DataLoader(self.val_set, batch_size=self.batch_size, shuffle=shuffle)
        self.loss_func = loss_func.to(self.device)
        self.lr = lr
        self.momentum = momentum
        self.early_stopping = early_stopping
        self.opt = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)

        self.avg_train_losses = []
        self.avg_val_losses = []
        self.val_acc = []
        self.best_score = 0
        self.best_loss = 10e10
        self.best_epoch = 0
        self.best_score_lst = []
        self.best_loss_lst = []
        self.best_model = copy.deepcopy(model)


    def train(self):
        sys.stdout = open(f"{self.data_dir}logs/{self.task_str}_{self.epochs}_{self.lr}_logs.txt", "w")
        for epoch in range(self.epochs):
            self.cur_epoch = epoch
            running_loss = 0.0
            for i, data in tqdm(enumerate(self.trainloader, 0)):
                # Get the inputs.
#                 print("LABELS:",data['labels'])
                inputs, labels = data['image'], data['labels']

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
                # print('running loss', running_loss)
        #         print(outputs)
                if i % self.print_freq == self.print_freq - 1: # Print every several mini-batches.
                    avg_loss = running_loss / self.print_freq
                    print('[epoch: {}, i: {:5d}] avg mini-batch loss: {:.3f}'.format(
                        epoch, i, avg_loss))
                    running_loss = 0.0
                    sys.stdout.close()
                    sys.stdout = open(f"{self.data_dir}logs/{self.task_str}_{self.epochs}_{self.lr}_logs.txt", "a")

            self.avg_train_losses.append(avg_loss)
            self.model.eval()
            with torch.no_grad():
                num_correct = 0
                k = 3
                # Check several images.

                dataiter = iter(self.valloader)
                batch_size = 1
                corr_dict = dict(zip(range(len(self.val_set.dataset.class_dict)), np.zeros(len(self.val_set.dataset.class_dict))))
                val_losses = []
                for i in tqdm(range(len(dataiter))):
                    sample = dataiter.next()
                    



                    val_inputs, val_labels = sample['image'].cuda(), torch.LongTensor(sample['labels']).cuda()
                    val_outputs = self.model(val_inputs.to(self.device))
                    val_predicted = np.array([int(torch.max(i, 1)[1].cpu()) for i in val_outputs])
                    val_losses.append(self.loss_func(val_outputs, val_labels).item())

                    val_labels = np.array(val_labels.cpu()).flatten()#.reshape(len(mtt.val_set.dataset.class_dict), -1)
                    corr_vals = np.where(val_labels==val_predicted)
                    idx_corr = corr_vals[0]

                    for j in idx_corr:
                        corr_dict[j] += 1
                    
                    
#                     val_inputs, val_labels = sample['image'].cuda(), torch.LongTensor(sample['labels']).cuda()
#                     val_outputs = self.model(val_inputs.to(self.device))
#                     val_predicted = [torch.max(i, 1)[1] for i in val_outputs]

#                     val_losses.append(self.loss_func(val_outputs, val_labels).item())

#                     val_labels = np.array(val_labels.cpu()).reshape(len(self.val_set.dataset.class_dict), -1)
#                     corr_vals = np.where(np.array(val_labels)==np.array(val_predicted))
#                     idx_corr = corr_vals[0]

#                     for j in idx_corr:
#                         corr_dict[j] += 1

#                     if i % 100 == 0:
#                         print("iteration",i)


#                 num_correct=0
#                 val_losses = []
#                 data_iter = iter(self.valloader)
#                 for val_data in tqdm(data_iter):
#                     val_inputs, val_labels = val_data['image'].cuda(), torch.LongTensor(val_data['labels']).cuda()
#                     val_outputs = self.model(val_inputs)
#                     self.opt.zero_grad() # zero the parameter gradients
#                     val_predicted = [torch.max(i, 1)[1] for i in val_outputs]
#                     num_correct += sum(np.array(val_labels.cpu()).flatten()==np.array(val_predicted[0].cpu()).flatten())
#                     # print(val_labels.cpu())
#                     # print(val_predicted[0].cpu())
#                     # break
#                     val_losses.append(self.loss_func(val_outputs, val_labels).item())
#                 acc = num_correct/(len(data_iter)*self.batch_size*len(val_labels))
                avg_validation_loss = np.mean(val_losses)
                pd_corr = pd.Series(corr_dict)
                pd_corr.index = self.label_dict.keys()
                acc = sum(pd_corr) / (len(dataiter) * self.num_tasks)
                print(pd_corr / len(dataiter))

                # Track and Log Best Validation Accuracy, Loss
                if acc > self.best_score:
                    # self.best_score = acc
                    best_str_acc = 'BEST '
                if avg_validation_loss < self.best_loss:
                    self.best_loss = avg_validation_loss
                    self.best_model = copy.deepcopy(self.model)
                    self.best_epoch = epoch
                    self.best_score = acc
                    best_str_loss = 'BEST '

                print(f'Validation scores for epoch {self.cur_epoch}')
                print(f'{best_str_acc}Validation accuracy:',acc)
                print(f'{best_str_loss}Avg Val Loss for epoch {self.cur_epoch}:', avg_validation_loss)
                self.avg_val_losses.append(avg_validation_loss)
                self.val_acc.append(acc)

                best_str_acc = ''
                best_str_loss = ''
                self.best_score_lst.append(self.best_score)
                self.best_loss_lst.append(self.best_loss)

                # Early Stopping Code
                if epoch > self.patience and pd.Series(self.best_loss_lst[-self.patience:]).nunique() == 1:
                    print(f'Early Stopping at Epoch {epoch} with BEST Validation Loss: {self.best_loss} and Validation Accuracy: {self.best_score}')
                    self.model = self.best_model
                    #epoch += (self.epochs-epoch) # finishes the outer loop
                    self.epochs = epoch
                    break
#                 print('Validation accuracy:',acc)
#                 print('Average validation loss:',np.mean(val_losses))
                sys.stdout.close()
                sys.stdout = open(f"{self.data_dir}logs/{self.task_str}_{self.epochs}_{self.lr}_logs.txt", "a")
#                 test_str = '_test' if self.test == True else ''
                # self.avg_val_losses.append(np.mean(val_losses))
            self.model.train()
            if epoch % 10 == 0:
                fpath = f'{self.data_dir}models/tmp__{self.task_str}_{self.epochs}_epoch_state_dict.pth'
                torch.save(self.best_model.state_dict(), fpath)
                print(f'Model saved at {fpath}')

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
        # create models and training_objects folders on first run of run.py
        fpath = f'{self.data_dir}models/{self.task_str}_{self.best_epoch}_epoch_{self.lr}_lr_state_dict.pth'
        torch.save(self.model.state_dict(), fpath)
        print(f'Model saved at {fpath}')

    def save_object(self):
        with open(f'{self.data_dir}models/training_objects/{self.task_str}_{self.best_epoch}_obj.pkl', 'wb') as f:
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
