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
from utils import *
import pandas as pd
import numpy as np
import os
import time
import sys
import h5py
from tqdm import tqdm
import scipy.ndimage as ndimage
import argparse
import re
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--month', type=str, default='00m', dest='month')
parser.add_argument('-c', '--contents', type=str, default=None, dest='content_dir')
parser.add_argument('-id', '--input-dir', type=str, default='../data-raw', dest='input_dir')
parser.add_argument('-od', '--output-dir', type=str, default='../data', dest='output_dir')

'''
This file used new pipeline to get images preprocessed.
'''
def main():
    args = parser.parse_args()
    df = pd.read_csv(args.content_dir)
    input_folder = args.output_dir
    output_folder = args.output_dir
    save_dir = args.output_dir + '/' + args.month
    fig_save_dir = args.output_dir + '/' + args.month + '_fig'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(fig_save_dir):
        os.makedirs(fig_save_dir)

    assert args.month in save_dir
    bar = tqdm(total=df.shape[0], desc='Processing', ncols=90)
    for idx, row in df.iterrows():
        data_path = row['file_path']
        try:
            data_path = data_path.replace('//','/')
            month = args.month
            p_id = data_path.split('/')[-4]
            bbox = row['pred_bbox'].strip()# .replace('[', '').replace(']', '').split(' ')
            # re to find all numbers postive/negative decimals
            bbox = re.findall(r"[+-]?\d+(?:\.\d+)?", bbox)
            bbox = np.array([float(i) for i in bbox])
            img, data, img_before = image_preprocessing(
                os.path.join(
                    input_folder,
                    data_path
                )
            )
            left, right = getKneeWithBbox(img, bbox)
            f_name_l = '{}_{}_LEFT_KNEE.hdf5'.format(p_id, month)
            f_name_r = '{}_{}_RIGHT_KNEE.hdf5'.format(p_id, month)
            print(p_id, bbox)
            if left is not None:
                create_h5(save_dir, f_name_l, left)
            if right is not None:
                create_h5(save_dir, f_name_r, right)
            fig_name = p_id + '_' + args.month
            drawKneeWithBbox(img, bbox, left, right, fig_save_dir, fig_name, labels=None)
            bar.update(1)
        except BaseException as err:
            print(f'Unable to process {data_path}:\n\t{err}')


if __name__ == '__main__':
    main()