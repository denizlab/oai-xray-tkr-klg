# ==============================================================================
# Copyright (C) 2020 Kevin Leung, Bofei Zhang, Jimin Tan, Yiqiu Shen, 
# Krzysztof J. Geras, James S. Babb, Kyunghyun Cho, Gregory Chang, Cem M. Deniz
#
# This file is part of oai-xray-tkr-klg
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
# ==============================================================================
from __future__ import print_function, division

import os
import warnings
import h5py
import numpy as np
import time
import copy
import math
import scipy.ndimage as ndimage

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms

import pandas as pd
import XrayDataLoader
from torch.utils.data import DataLoader

import random
from collections import Counter, defaultdict


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--cropped_size', default=896)
parser.add_argument('-i', '--image_size', default=1024)
parser.add_argument('-e', '--num_epochs', default=200)
parser.add_argument('-l', '--learning_rate', default=0.0001)
parser.add_argument('-f', '--num_folds', default=7)
parser.add_argument('-s', '--train_scratch', default=0)
parser.add_argument('-t', '--test_string', default="nestedCV")
parser.add_argument('-d', '--data_folder', default ="./data/00m") # give the data path
parser.add_argument('-g', '--cohort', default ="728_Cohort_KLG_w_Strata.csv")
# Resnet 34 models trained on:
# ImageNet
parser.add_argument('-m', '--model', default="Resnet34") 
args = parser.parse_args()



dataFolder = args.data_folder
dataFileName = args.cohort
torch.manual_seed(1234)
np.random.seed(1234)

tl_model = args.model
cropped_size = int(args.cropped_size)
image_size = int(args.image_size)
train_scratch = int(args.train_scratch)
test_string = args.test_string

class multi_output_model(torch.nn.Module):
    def __init__(self, model_core ,num_ftrs):
        super(multi_output_model, self).__init__()
        
        self.resnet_model = model_core
        
        #heads
        self.y1o = nn.Linear(num_ftrs,2)
        self.y2o = nn.Linear(num_ftrs,5)
        
    def forward(self, x):
       
        x1 = self.resnet_model(x)

        ## only get until the FC 
        
        # heads
        y1o = self.y1o(x1)
        y2o = self.y2o(x1)
        
        return y1o, y2o

# Custom loss layer
class CustomMultiLoss(nn.Module):
    def __init__(self, model, nb_outputs=2, **kwargs):
        super(CustomMultiLoss, self).__init__()
        self.nb_outputs = nb_outputs
        self.is_placeholder = True
        self.model = model
        self.log_vars = nn.Parameter(torch.ones((nb_outputs)), requires_grad=False)     
        #print(self.log_vars)     

    def multi_loss(self, ys_true, ys_pred):
        assert len(ys_true) == self.nb_outputs and len(ys_pred) == self.nb_outputs
        loss = 0
        ct=0
        for y_true, y_pred in zip(ys_true, ys_pred):
            loss += nn.CrossEntropyLoss()(y_pred, y_true)*self.log_vars[ct]
            #print(ct, loss, self.log_vars[ct])
            ct +=1
        return loss

    def forward(self, inputs):
        ys_pred = self.model(inputs[0])
        #print('ys_pred',ys_pred)
        ys_true = inputs[1::]
        #print('ys_true',ys_true)
        loss = self.multi_loss(ys_true, ys_pred)
        return loss

'''From https://www.kaggle.com/jakubwasikowski/stratified-group-k-fold-cross-validation'''
def stratified_group_k_fold(X, y, groups, k, seed=None):
    labels_num = np.max(y) + 1
    y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
    y_distr = Counter()
    #print(y.shape,groups.shape)
    for label, g in zip(y, groups):
        y_counts_per_group[g][label] += 1
        y_distr[label] += 1
    #print(y_counts_per_group)
    y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
    groups_per_fold = defaultdict(set)

    def eval_y_counts_per_fold(y_counts, fold):
        y_counts_per_fold[fold] += y_counts
        std_per_label = []
        for label in range(labels_num):
            label_std = np.std([y_counts_per_fold[i][label] / y_distr[label] for i in range(k)])
            std_per_label.append(label_std)
        y_counts_per_fold[fold] -= y_counts
        return np.mean(std_per_label)
    
    groups_and_y_counts = list(y_counts_per_group.items())
    #print(groups_and_y_counts)
    random.Random(seed).shuffle(groups_and_y_counts)

    for g, y_counts in sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])):
        best_fold = None
        min_eval = None
        for i in range(k):
            fold_eval = eval_y_counts_per_fold(y_counts, i)
            if min_eval is None or fold_eval < min_eval:
                min_eval = fold_eval
                best_fold = i
        y_counts_per_fold[best_fold] += y_counts
        groups_per_fold[best_fold].add(g)
    #print(groups_per_fold)

    all_groups = set(groups)
    for i in range(k):
        train_groups = all_groups - groups_per_fold[i]
        test_groups = groups_per_fold[i]

        train_indices = [i for i, g in enumerate(groups) if g in train_groups]
        test_indices = [i for i, g in enumerate(groups) if g in test_groups]

        yield train_indices, test_indices


'''
    Def:
        Trains PyTorch model and saves weights whenever validation loss improves
    Params:
        model = PyTorch model -- we initialize with ResNet34
        criterion = loss function - we use cross-entropy
        optimizer = optimization function - we use Adam
        num_epochs = number of epochs to train
        dataloaders = dataloading object for PyTorch
        dataset_sizes = size of the train and val sets
        device = device object (i.e. gpu or cpu)
        fold_num = fold we are on for cross validation
        file_path = path to save file
'''
def train_model(model, criterion, optimizer, num_epochs, data_transforms, dataset_sizes, device, fold_num, file_path):
    since = time.time()
    print(model)
    # Initializes best variables for weights, acc, and loss
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = 100.0
    counter = 0
    savedEpoch = 0

    # Creates arrays for plotting training evolution 
    loss2plot = np.zeros([num_epochs,2])
    acc2plot = np.zeros([num_epochs,2])

    BATCH_SIZE = 8
    ## data loaders
    Xray_TrainData = XrayDataLoader.XrayDataset(csv_file=file_path + 'inner_train.csv',
                                    root_dir=dataFolder, transform=data_transforms['train'])
    train_loader = DataLoader(Xray_TrainData, batch_size=BATCH_SIZE,
                        shuffle=True, num_workers=2)

    Xray_ValidationData = XrayDataLoader.XrayDataset(csv_file=file_path + 'inner_validation.csv',
                                    root_dir=dataFolder, transform=data_transforms['val'])
    validation_loader = DataLoader(Xray_ValidationData, batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=2)


    dataloaders = {'train': train_loader,
                   'val': validation_loader
                  }

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)


        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            #print('everything initialized')

            # Iterate over data.
            for sample_batched in dataloaders[phase]:

                inputs = sample_batched['x']
                labels = sample_batched['y']
                ptId = sample_batched['id']
                kls = sample_batched['kl']

                inputs = inputs.to(device, dtype=torch.float)
                labels = labels.to(device, dtype = torch.long)
                kls = kls.to(device, dtype = torch.long)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train' and epoch > 0):
                    outputs, outputs_kl = model(inputs)

                    _, preds = torch.max(outputs, 1)

                    loss = criterion([inputs, labels, kls])

                    # backward + optimize only if in training phase
                    if phase == 'train' and epoch > 0:
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('Fold {}: {} Loss: {:.4f} Acc: {:.4f}'.format(
                fold_num, phase, epoch_loss, epoch_acc))

            if phase == 'train':
                loss2plot[epoch,0] = epoch_loss
                acc2plot[epoch,0] = epoch_acc
            else:
                loss2plot[epoch,1] = epoch_loss
                acc2plot[epoch,1] = epoch_acc

            if phase == 'val':
                weights_path = 'weights-{:02d}-T-{:.3f}-{:.3f}-V-{:.3f}-{:.3f}.pth'.format(epoch, loss2plot[epoch,0], acc2plot[epoch,0], epoch_loss, epoch_acc)
                path = file_path + weights_path

            counter +=1
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc: #
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

                weights_path = 'best_weights.pth'
                path = file_path + weights_path
                #print(path)
                torch.save(model.state_dict(), path)
                print('Best Model saved in epoch#: %d'%epoch)
                counter = 0
                savedEpoch = epoch
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss

        print()

    time_elapsed = time.time() - since
    print('INNER Fold Number: ' + str(fold_num))
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Loss: {:4f}'.format(best_loss))
    print('Best val Acc: {:4f}'.format(best_acc))

    loss_path = file_path + 'loss_curves.csv'
    np.savetxt(loss_path, loss2plot, delimiter=',')

    acc_path = file_path + 'acc_curves.csv'
    np.savetxt(acc_path, acc2plot, delimiter=',')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, savedEpoch

def get_model(tl_model):
    # load the pretrained model, Resnet34 was used in the paper
    if tl_model == "Resnet34":
        if (train_scratch):
            model_ft = models.resnet34(pretrained=False)
        else:
            model_ft = models.resnet34(pretrained=False)
            # Download torchvision pretrained model from: https://download.pytorch.org/models/resnet34-333f7ec4.pth
            model_ft.load_state_dict(torch.load('resnet34-333f7ec4.pth'))
        if image_size == 1024:
            model_ft.avgpool = nn.AvgPool2d(kernel_size=28, stride=1, padding=0) 
    elif tl_model == "Resnet50":
        if (train_scratch):
            model_ft = models.resnet50(pretrained=False)
        else:
            model_ft = models.resnet50(pretrained=True)
        if image_size == 1024:
            model_ft.avgpool = nn.AvgPool2d(kernel_size=28, stride=1, padding=0) 
    elif tl_model == "DenseNet":
        if (train_scratch):
            model_ft = models.densenet201(pretrained=False)
        else:
            model_ft = models.densenet201(pretrained=True)
        if image_size == 1024:
            model_ft.avgpool = nn.AvgPool2d(kernel_size=28, stride=1, padding=0) 


    if tl_model == "DenseNet":
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Sequential()
        model_ft = multi_output_model(model_ft,num_ftrs)
    else:
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Sequential()
        model_ft = multi_output_model(model_ft,num_ftrs)

    return model_ft

def run_inference(model_ft, csv_file, data_transforms, dataFolder, inner_path, tl_model, image_size, ii,innerdataset_sizes,device,test_index,savedEpoch=0):
    output = np.zeros(shape=(innerdataset_sizes['val'], 2))
    output_kl = np.zeros(shape=(innerdataset_sizes['val'], 5))
    ids = np.zeros(shape=(innerdataset_sizes['val']))
    labs = np.zeros(shape=(innerdataset_sizes['val']))
    kneeSide = np.zeros(shape=(innerdataset_sizes['val']))
    klgrade = np.zeros(shape=(innerdataset_sizes['val']))
    analysis_loader = DataLoader(XrayDataLoader.XrayDataset(csv_file= csv_file,
                                root_dir=dataFolder, transform=data_transforms['val']), 
                                batch_size=1, shuffle=False, num_workers=0)
    ci = 0
    for sample_batched in analysis_loader:
        inputs = sample_batched['x']
        labels = sample_batched['y']
        ptId = sample_batched['id']
        kside = sample_batched['side']
        kl = sample_batched['kl']

        inputs = inputs.to(device, dtype=torch.float)
        output_val = model_ft(inputs)
        m = nn.Softmax(dim=1)
        output[ci] = m(output_val[0]).data.cpu().numpy()
        output_kl[ci] = m(output_val[1]).data.cpu().numpy()
        ids[ci] = ptId
        labs[ci] = labels
        kneeSide[ci] = kside
        klgrade[ci] = kl
        ci+=1
    
    # compute softmax in Excel - output in the form [0, 1] so get probability of 1 using softmax 
    return_array = np.empty(shape=(len(test_index), 11))
    return_array[:,0] = ids
    return_array[:,1:3] = output
    return_array[:,3] = labs
    return_array[:,4] = kneeSide
    return_array[:,5] = klgrade
    return_array[:,6:11] = output_kl

    np.savetxt(inner_path + '%s_%d_fold%d.csv'%(tl_model,image_size,ii), return_array, delimiter=',')
    with open(inner_path + '/SavedEpochNo.txt', 'w') as f:
        f.write(str(savedEpoch))
    print()


def nested_cross_validation(learning_rate, num_epochs, num_of_folds, file_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seed = 1234
    
    model_path = file_path + 'lr%s/' % (learning_rate)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    df = pd.read_csv(dataFileName)
    print(df.head())
    labels = df["Label"].values
    strata = df["Strata"].values
    #print(strata)

    i = 1
    model_saved={}
    #outer loop of nested CV
    for indices, (train_index, test_index) in enumerate(stratified_group_k_fold(np.zeros(len(labels)), labels, groups=strata, k=num_of_folds)):

        fold_path = model_path + 'Fold_' + str(i) + '/'
        if not os.path.exists(fold_path):
            os.makedirs(fold_path)    

        #print('SSS',test_index)
        #print('SSSsss',train_index 
        # write train and validation csvs for data loaders
        traindf=df.iloc[train_index,:]
        traindf.to_csv(fold_path + 'outer_train.csv',index=False)

        valdf=df.iloc[test_index,:]
        valdf.to_csv(fold_path + 'outer_validation.csv',index=False)
        print('TestIndex',test_index)

        transRGB = XrayDataLoader.ToRGB() if tl_model != "CC" else XrayDataLoader.Identity()
        transResize =  XrayDataLoader.Identity() if image_size == 1024 else XrayDataLoader.Resize(image_size)

        data_transforms = {'train': transforms.Compose([
                                transResize,
                                XrayDataLoader.RandomCrop(cropped_size),
                                transRGB,
                                XrayDataLoader.ToTensor(),
                                XrayDataLoader.RandomHorizontalFlip(),
                            ]),
                            'val': transforms.Compose([
                                transResize,
                                XrayDataLoader.CenterCrop(cropped_size),
                                transRGB,
                                XrayDataLoader.ToTensor(),
                            ]),
                            }

        dataset_sizes = {'train': len(train_index), 'val': len(test_index)}
        print('#####',dataset_sizes)

        #inner part of the nested CV
        inner_labels = traindf["Label"].values
        inner_strata = traindf["Strata"].values
        ii=1
        #Inner loop of nested CV
        for indices, (inner_train_index, inner_test_index)  in enumerate(stratified_group_k_fold(np.zeros(len(inner_labels)), inner_labels, groups=inner_strata, k=num_of_folds-1)):

            inner_path = fold_path + 'CV' + str(ii) + '/'
            if not os.path.exists(inner_path):
                os.makedirs(inner_path)
            # write train and validation csvs for data loaders
            innertraindf=traindf.iloc[inner_train_index,:]
            innertraindf.to_csv(inner_path + 'inner_train.csv',index=False)

            innervaldf=traindf.iloc[inner_test_index,:]
            innervaldf.to_csv(inner_path + 'inner_validation.csv',index=False)
            #print('InnerTestIndex',inner_test_index)

            innerdataset_sizes = {'train': len(inner_train_index), 'val': len(inner_test_index)}
            print('#####',innerdataset_sizes)

            # get the model to use for learning
            model_ft = get_model(tl_model)

            model_ft = model_ft.to(device)
            print('network created')

            cml = CustomMultiLoss(model=model_ft, nb_outputs=2)
            cml = cml.to(device)
            optimizer_ft = optim.Adam(cml.parameters(), lr=learning_rate)

            print('ready to train')
            print()

            model_saved[ii], savedEpoch = train_model(model_ft, 
                                           cml, 
                                           optimizer_ft, 
                                           num_epochs=num_epochs, 
                                           data_transforms=data_transforms, 
                                               dataset_sizes = innerdataset_sizes,
                                           device=device,
                                               fold_num=ii,
                                               file_path=inner_path) 

        # run inference on validation set
            run_inference(model_saved[ii], inner_path + 'inner_validation.csv', data_transforms, dataFolder, inner_path, tl_model, image_size, ii,innerdataset_sizes,device,inner_test_index,savedEpoch)
            ii+=1

        # this is the place infer each nested model trained with the outer validation data to save for future ensembling like averaging
        for inf_idx in range(1,7):
            print('Inferred model no:', inf_idx)
            run_inference(model_saved[inf_idx], fold_path + 'outer_validation.csv', data_transforms, dataFolder, fold_path, tl_model, image_size, inf_idx, dataset_sizes,device,test_index)

        i+=1
        print()


if __name__ == '__main__':   

    if (train_scratch):                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
        fp = 'model_weights_multiTask_%s/%s_scratch/%d/'%(test_string,tl_model,image_size)
    else:
    	fp = 'model_weights_multiTask_%s/%s/%d/'%(test_string,tl_model,image_size)
    nested_cross_validation(learning_rate=float(args.learning_rate), 
                     num_epochs=int(args.num_epochs), 
                     num_of_folds=int(args.num_folds), 
                     file_path=fp)

