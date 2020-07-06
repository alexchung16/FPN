#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : split_dataset.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : @ Time 11/12/2019 AM 11:13
# @ Software   : PyCharm
#-------------------------------------------------------
import os
import math
import shutil
import numpy as np

from utils.tools import makedir, view_bar

original_dataset_dir = '/media/alex/AC6A2BDB6A2BA0D6/alex_dataset/Pascal_VOC_2012/VOCtrainval/VOCdevkit/VOC2012'
dst_dir = '/media/alex/AC6A2BDB6A2BA0D6/alex_dataset/pascal_split'

#++++++++++++++++++++++++++++++++++++++++split pascal++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def split_pascal(origin_path, dst_path, split_rate=0.8):
    """
    split pascal dataset
    :param origin_path:
    :return:
    """
    image_path = os.path.join(origin_path, 'JPEGImages')
    xml_path = os.path.join(origin_path, 'Annotations')

    image_train_path = os.path.join(dst_path, 'train', 'JPEGImages')
    xml_train_path = os.path.join(dst_path, 'train', 'Annotations')
    image_val_path = os.path.join(dst_path, 'val', 'JPEGImages')
    xml_val_path = os.path.join(dst_path, 'val', 'Annotations')
    makedir(image_train_path)
    makedir(xml_train_path)
    makedir(image_val_path)
    makedir(xml_val_path)

    image_list = os.listdir(image_path)
    image_name = [image.split('.')[0] for image in image_list]
    image_name = np.random.permutation(image_name)
    train_image_name = image_name[:int(math.ceil(len(image_name) * split_rate))]
    val_image_name = image_name[int(math.ceil(len(image_name) * split_rate)):]

    for n, image in enumerate(train_image_name):
        shutil.copy(os.path.join(image_path, image+'.jpg'), os.path.join(image_train_path, image+'.jpg'))
        shutil.copy(os.path.join(xml_path, image + '.xml'), os.path.join(xml_train_path, image + '.xml'))
        view_bar(message="split train dataset:", num=n, total=len(train_image_name))
    print('Total of {0} data split to {1}'.format(len(train_image_name), os.path.dirname(image_train_path)))

    for n, image in enumerate(val_image_name):
        shutil.copy(os.path.join(image_path, image+'.jpg'), os.path.join(image_val_path, image+'.jpg'))
        shutil.copy(os.path.join(xml_path, image + '.xml'), os.path.join(xml_val_path, image + '.xml'))
        view_bar(message="split val dataset:", num=n, total=len(val_image_name))
    print('Total of {0} data split to {1}'.format(len(val_image_name),  os.path.dirname(image_val_path)))


if __name__ == "__main__":


    image_list = os.listdir(os.path.join(original_dataset_dir, 'JPEGImages'))
    image_name = [image.split('.')[0] for image in image_list]
    print(f'number of sample:{len(image_list)}')
    split_pascal(original_dataset_dir, dst_dir, 0.8)

