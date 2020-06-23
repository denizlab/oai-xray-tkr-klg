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
import os
from skimage import io
import torch
from torchvision import transforms
import torchvision
from skimage import color
import pandas as pd
from torch.utils.data import Dataset
import h5py
import numpy as np
import scipy.ndimage as ndimage

class XrayDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file filename information.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):

        img_name = os.path.join(self.root_dir,
                                self.data_frame.iloc[idx, self.data_frame.columns.get_loc('h5Name')])
        
        f = h5py.File(img_name, 'r')
        image = f.get('data').value
        image = image[...,np.newaxis]
        f.close()
            
        image_class = self.data_frame.iloc[idx, self.data_frame.columns.get_loc('Label')]
        patientID = self.data_frame.iloc[idx, self.data_frame.columns.get_loc('ID')]
        kneeSide = self.data_frame.iloc[idx, self.data_frame.columns.get_loc('Side')]
        klgrade = self.data_frame.iloc[idx, self.data_frame.columns.get_loc('KLG')]

        if self.transform:
            image = self.transform(image)

        sample = {'x': image, 'y': image_class, 'id': patientID, 'side': kneeSide, 'kl': klgrade}

        return sample

class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image = sample

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]


        return image

class CenterCrop(object):
    """Center Crop the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image = sample

        h, w = image.shape[:2]
        new_h, new_w = self.output_size


        x = (h - new_h) // 2
        y = (w - new_w) // 2

        image = image[y:(y + new_h),x:(x + new_w)]

        return image

class Resize(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image = sample

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        image_array_rescaled = ndimage.zoom(image, [new_h/h, new_w/w, 1])

        return image_array_rescaled

class RandomHorizontalFlip(object):
    def __call__(self, sample):
        image = sample
        step = np.random.choice([1,-1])
        if step == -1:
            image = torch.flip(image,[2]) # fliplr on the width axis of C x H x W Tensor
        return image

class ToTensor(object):
    def __call__(self, sample):
        image = sample

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return torch.from_numpy(image)

class ToRGB(object):
    def __call__(self, sample):
        image = np.tile(sample,3)
        return image

class Identity(object):
    def __call__(self, sample):
        return sample

class Normalize(object):
    def __call__(self, data):
        new_data = np.empty([data.shape[0], data.shape[1], data.shape[2]], dtype = np.float64)
    
        for i in range(data.shape[0]):
            new_data[i,:,:]  = data[i,:,:] - np.amin(data[i,:,:])
            new_data[i,:,:] /= np.amax([np.amax(data[i,:,:])- np.amin(data[i,:,:]),1e-8])

        return new_data
