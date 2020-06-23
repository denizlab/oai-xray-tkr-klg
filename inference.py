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
import copy
import math
import scipy.ndimage as ndimage

import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms

import XrayDataLoader
from torch.utils.data import DataLoader

from collections import Counter, defaultdict

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--cropped_size', default=896)
parser.add_argument('-i', '--image_size', default=1024)
parser.add_argument('-f', '--num_folds', default=7)
parser.add_argument('-p', '--model_path', default = "./ModelWeights")

# knee image filename in hdf5 format
parser.add_argument('-d', '--file_name', default ="./data/00m/9011918_00m_LEFT_KNEE.hdf5") 
# Resnet 34 models trained on Knee Radiographs
parser.add_argument('-m', '--model', default="Resnet34") 
args = parser.parse_args()

fileName = args.file_name

tl_model = args.model
model_path = args.model_path
cropped_size = int(args.cropped_size)
image_size = int(args.image_size)

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

def get_model(tl_model):
    # load the pretrained model, Resnet34 was used in the paper
    if tl_model == "Resnet34":
        model_ft = models.resnet34(pretrained=False)
        if image_size == 1024:
            model_ft.avgpool = nn.AvgPool2d(kernel_size=28, stride=1, padding=0) 

        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Sequential()
        model_ft = multi_output_model(model_ft,num_ftrs)

    return model_ft

def run_inference(model_ft, image, device):
    output = np.zeros(7)

    inputs = image.to(device, dtype=torch.float)
    output_val = model_ft(inputs)
    m = nn.Softmax(dim=1)
    output[0:2] = m(output_val[0]).data.cpu().numpy()
    output[2:] = m(output_val[1]).data.cpu().numpy()

    return output

def image_loader(img_name):
    f = h5py.File(img_name, 'r')
    image = f.get('data').value
    image = image[...,np.newaxis]
    f.close()

    transRGB = XrayDataLoader.ToRGB() if tl_model != "CC" else XrayDataLoader.Identity()
    transResize =  XrayDataLoader.Identity() if image_size == 1024 else XrayDataLoader.Resize(image_size)

    data_transforms = transforms.Compose([
                            transResize,
                            XrayDataLoader.CenterCrop(cropped_size),
                            transRGB,
                            XrayDataLoader.ToTensor(),
                        ])
    image = data_transforms(image).float()
    image = image.unsqueeze(0)
    return image

def inference_nested_cross_validation(num_of_folds):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seed = 1234
    average_output=np.zeros(7)
    # get the model to used during training
    model_infer = get_model(tl_model)
    image_in = image_loader(fileName)

    #outer loop of nested CV
    for idx_out in range(num_of_folds):

        #Inner loop of nested CV
        for idx_in  in range(num_of_folds-1):

            loadModelFile = model_path + '/Fold_' + str(idx_out+1) + '/CV' +str(idx_in+1) + '/best_weights.pth'  

            # load model weights
            model_infer.load_state_dict(torch.load(loadModelFile))
            model_infer.eval()

            model_infer = model_infer.to(device)
            
            # run inference on the image
            tmp = run_inference(model_infer, image_in, device)
            average_output +=tmp

    #return average ensample result
    return average_output/(num_of_folds*(num_of_folds-1))

if __name__ == '__main__':   
    out=inference_nested_cross_validation(num_of_folds=int(args.num_folds))
    print('--- Inference Results ---')
    print('Predictions for **',  fileName.split('/')[-1], "**")
    print('Total Knee Replacement (TKR): %.2f'%out[1])
    print('KL grade 0: %.2f'%out[2])
    print('KL grade 1: %.2f'%out[3])
    print('KL grade 2: %.2f'%out[4])
    print('KL grade 3: %.2f'%out[5])
    print('KL grade 4: %.2f'%out[6])
