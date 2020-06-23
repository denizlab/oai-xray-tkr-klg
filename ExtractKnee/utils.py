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
import numpy as np
import pydicom as dicom
import cv2
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import scipy.ndimage as ndimage
import h5py
import pandas as pd
import time
import os
import numpy as np
import pydicom as dicom
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def image_preprocessing(file_path = '../data/9000296'):
    '''

    :param file_path:
    :return:
    '''
    # read data from DICOM file
    data = dicom.read_file(file_path)
    photoInterpretation = data[0x28,0x04].value # return a string of photometric interpretation
    #print('######### PHOTO INTER {} #########'.format(photoInterpretation))
    if photoInterpretation not in ['MONOCHROME2','MONOCHROME1']:
        raise ValueError('Wrong Value of Photo Interpretation: {}'.format(photoInterpretation))
    img = interpolate_resolution(data).astype(np.float64) # get fixed resolution
    img_before = img.copy()
    if photoInterpretation == 'MONOCHROME1':
        img = invert_Monochrome1(img)
    # apply normalization, move into hist_truncation.
    # img = global_contrast_normalization(img)
    # apply hist truncation
    img = hist_truncation(img)
    # get center part of image if image is large enough
    return img, data, img_before


def invert_Monochrome1(image_array):
    '''
    Image with dicome attribute [0028,0004] == MONOCHROME1 needs to
    be inverted. Otherwise, our way to detect the knee will not work.

    :param image_array:
    :return:
    '''
    image_array = image_array.max() - image_array
    return image_array


def interpolate_resolution(image_dicom, scaling_factor=0.2):
    '''
    Obtain fixed resolution from image dicom
    :param image_dicom:
    :param scaling_factor:
    :return:
    '''
    image_array = image_dicom.pixel_array
    try:
        x = image_dicom[0x28, 0x30].value[0]
        y = image_dicom[0x28, 0x30].value[1]
        image_array = ndimage.zoom(image_array, [x / scaling_factor, y / scaling_factor])
    except KeyError:
        pass
    return image_array


def get_center_image(img,img_size = (2048,2048)):
    '''
    Get the center of image
    :param img:
    :param img_size:
    :return:
    '''
    rows, cols = img.shape
    center_x = rows // 2
    center_y = cols // 2
    img_crop = img[center_x - img_size[0] // 2: center_x + img_size[0] // 2,
                   center_y - img_size[1] // 2: center_y + img_size[1] // 2]
    return img_crop


def padding(img, img_size = (2048,2048)):
    '''
    Padding image array to a specific size
    :param img:
    :param img_size:
    :return:
    '''
    rows,cols = img.shape
    x_padding = img_size[0] - rows
    y_padding = img_size[1] - cols
    if x_padding > 0:
        before_x,after_x = x_padding // 2, x_padding - x_padding // 2
    else:
        before_x,after_x = 0,0
    if y_padding > 0:
        before_y,after_y = y_padding // 2, y_padding - y_padding // 2
    else:
        before_y,after_y = 0,0
    return np.pad(img, ((before_x, after_x), (before_y, after_y), ), 'constant', constant_values=0), before_x, before_y


def global_contrast_normalization_oulu(img,lim1,multiplier = 255):
    '''
    This part is taken from oulu's lab. This how they did global contrast normalization.
    :param img:
    :param lim1:
    :param multiplier:
    :return:
    '''
    img -= lim1
    img /= img.max()
    img *= multiplier
    return img


def global_contrast_normalization(img, s=1, lambda_=10, epsilon=1e-8):
    '''
    Apply global contrast normalization based on image array.
    Deprecated since it is not working ...
    :param img:
    :param s:
    :param lambda_:
    :param epsilon:
    :return:
    '''
    # replacement for the loop
    X_average = np.mean(img)
    img_center = img - X_average

    # `su` is here the mean, instead of the sum
    contrast = np.sqrt(lambda_ + np.mean(img_center ** 2))

    img = s * img_center / max(contrast, epsilon)
    return img


def hist_truncation(img,cut_min=5,cut_max=99):
    '''
    Apply 5th and 99th truncation on the figure.
    :param img:
    :param cut_min:
    :param cut_max:
    :return:
    '''
    lim1,lim2 = np.percentile(img,[cut_min, cut_max])
    img_ = img.copy()
    img_[img < lim1] = lim1
    img_[img > lim2] = lim2
    img_ = global_contrast_normalization(img_)
    return img_


def drawFigureOnOriginal(img, labels, preds, f_name, folder):
    '''
        draw a png figure with rect of ground truth and prediction
        col == x, row == y
        :param img:
        :param labels:
        :param preds:
        :param f_name:
        :return:
    '''
    fig, ax = plt.subplots(1)
    row, col = img.shape
    ax.imshow(img)
    # draw true patch
    if labels is not None:
        x1, y1, x2, y2 = labels[:4]
        x1 = int(x1 * col)
        x2 = int(x2 * col)
        y1 = int(y1 * row)
        y2 = int(y2 * row)
        rect1 = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect1)
        x1, y1, x2, y2 = labels[4:]
        x1 = int(x1 * col)
        x2 = int(x2 * col)
        y1 = int(y1 * row)
        y2 = int(y2 * row)
        rect2 = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect2)
    if preds is not None:
        # draw predict patch
        preds = preds
        x1, y1, x2, y2 = preds[:4]
        x1 = int(x1 * col)
        x2 = int(x2 * col)
        y1 = int(y1 * row)
        y2 = int(y2 * row)
        rect1 = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='b', facecolor='none')
        print('Left:({},{}) - ({},{})'.format(x1, y1, x2, y2))
        ax.add_patch(rect1)
        x1, y1, x2, y2 = preds[4:]
        x1 = int(x1 * col)
        x2 = int(x2 * col)
        y1 = int(y1 * row)
        y2 = int(y2 * row)
        print('Right:({},{}) - ({},{})'.format(x1,y1,x2,y2))
        rect2 = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='b', facecolor='none')
        ax.add_patch(rect2)
    # save image
    plt.savefig(os.path.join(folder, f_name), dpi=300)
    plt.close()


def drawFigure(img, ax, preds=None, labels=None):
    '''
        draw a png figure with rect of ground truth and prediction
        col == x, row == y
        :param img:
        :param labels: label bbox
        :param preds: bbox
        :param f_name:
        :return:
    '''
    row, col = img.shape
    ax.imshow(img, cmap='gray')
    # draw true patch
    if labels is not None:
        print(labels)
        if -1 not in labels[:4]:
            x1, y1, x2, y2 = labels[:4]
            x1 = int(x1 * col)
            x2 = int(x2 * col)
            y1 = int(y1 * row)
            y2 = int(y2 * row)
            rect1 = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect1)
        if -1 not in labels[4:]:
            x1, y1, x2, y2 = labels[4:]
            x1 = int(x1 * col)
            x2 = int(x2 * col)
            y1 = int(y1 * row)
            y2 = int(y2 * row)
            rect2 = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect2)
    if preds is not None:
        # draw predict patch
        if -1 not in preds[:4]:
            x1, y1, x2, y2 = preds[:4]
            x1 = int(x1 * col)
            x2 = int(x2 * col)
            y1 = int(y1 * row)
            y2 = int(y2 * row)
            rect1 = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='b', facecolor='none')
            print('Left:({},{}) - ({},{})'.format(x1, y1, x2, y2))
            ax.add_patch(rect1)
        if -1 not in preds[4:]:
            x1, y1, x2, y2 = preds[4:]
            x1 = int(x1 * col)
            x2 = int(x2 * col)
            y1 = int(y1 * row)
            y2 = int(y2 * row)
            print('Right:({},{}) - ({},{})'.format(x1,y1,x2,y2))
            rect2 = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='b', facecolor='none')
            ax.add_patch(rect2)
    return ax


def drawKneeWithBbox(img, bbox, left=None, right=None, output_dir=None, f_name=None, labels=None):
    fig, axes = plt.subplots(3, 1)
    axes[0] = drawFigure(img, axes[0], bbox, labels=labels)
    axes[0].set_title(f_name)
    if left is not None:
        axes[1] = drawFigure(left, axes[1])
        axes[1].set_title('left')
    if right is not None:
        axes[2] = drawFigure(right, axes[2])
        axes[2].set_title('right')
    plt.savefig(output_dir + '/' + f_name, dpi=300, bbox_inches='tight')
    plt.close()


def getKneeWithBbox(img, bbox):
    row, col = img.shape
    if (bbox[0:4] >= 0).all():
        x1, y1, x2, y2 = bbox[:4]

        x1 = int(x1 * col)
        x2 = int(x2 * col)
        y1 = int(y1 * row)
        y2 = int(y2 * row)
        # max is used to avoid negative index
        h = y2 - y1
        w = x2 - x1
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        tmp = img[max(cy - 512, 0): cy - 512 + 1024, max(cx - 512, 0): cx - 512 + 1024]
        left, _, _ = padding(tmp, img_size=(1024, 1024))

    else:
        left = None
    if (bbox[4:] >= 0).all():
        x1, y1, x2, y2 = bbox[4:]
        x1 = int(x1 * col)
        x2 = int(x2 * col)
        y1 = int(y1 * row)
        y2 = int(y2 * row)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        h = y2 - y1
        w = x2 - x1
        tmp = img[max(cy - 512, 0): cy - 512 + 1024, max(cx - 512, 0): cx - 512 + 1024]
        right, _, _ = padding(tmp, img_size=(1024, 1024))
    else:
        right = None

    return left, right


def create_h5(save_dir,f_name,img):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    data_path = os.path.join(save_dir,f_name)
    f = h5py.File(data_path, 'w')
    f.create_dataset('data', data=img)
    f.close()