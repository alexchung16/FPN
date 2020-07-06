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
import json
import shutil
import numpy as np
from collections import defaultdict

from utils.tools import makedir, view_bar

# original_dataset_dir = '/home/alex/Documents/datasets/Pascal_VOC_2012/VOCtrainval/VOCdevkit'

data_type = 'val2017'
dataset_dir = '/media/alex/AC6A2BDB6A2BA0D6/alex_dataset/COCO_2017'

img_dir = os.path.join(dataset_dir, data_type)
instance_dir = '{0}/annotations_trainval2017/annotations/instances_{1}.json'.format(dataset_dir, data_type)

sub_coco = '/media/alex/AC6A2BDB6A2BA0D6/alex_dataset/sub_coco'
sub_annotations = os.path.join(sub_coco, 'Annotations')
sub_images = os.path.join(sub_coco, 'Images')


def split_coco(imgs_path, annotaions_path, dst_dir, num_catetory=20, num_per_category=18):
    """

    :param origin_path:
    :param split_ratio:
    :return:
    """

    dataset = json.load(open(annotaions_path, 'r'))

    sub_annotations_path = os.path.join(dst_dir, 'Annotations')
    sub_img_path = os.path.join(dst_dir, 'Images')


    anns, cats, imgs, img_anns, cate_imgs = create_index(dataset)

    img_id_list, category_id_list = get_img_per_categorise(cate_imgs, num_catetory, num_per_category)

    img_name_dict = {}
    for i, img_id in enumerate(img_id_list):
        img_name_dict[img_id] = '0' * (12 - len(str(img_id))) + '{0}.jpg'.format(img_id)

    #----------------------------write annotaion info-----------------------------------
    images_list, annotations_list = get_images_annotaion_info(img_id_list, imgs, img_anns, category_id_list)
    new_dataset = defaultdict(list)
    new_dataset['info'] = dataset['info']
    new_dataset['licenses'] = dataset['licenses']
    new_dataset['images'] = images_list
    new_dataset['annotations'] = annotations_list
    new_dataset['categories'] = dataset['categories']

    makedir(sub_annotations_path)
    json_path = os.path.join(sub_annotations_path, 'instances.json')
    with open(json_path, 'w') as fw:
        json.dump(new_dataset, fw)
    print('Successful write the number of {0} annotations respect to {1} images to {2}'.
          format(len(new_dataset['annotations']), len(new_dataset['images']), json_path))

    #---------------------------------remove image---------------------------------------
    makedir(sub_img_path)

    num_samples = 0
    for img_id, img_name in img_name_dict.items():
        shutil.copy(os.path.join(imgs_path, img_name), os.path.join(sub_img_path, '{0}.jpg'.format(img_id)))
        num_samples += 1
        view_bar("split coco:", num_samples, len(img_name_dict))

    print('Successful copy the number of {0} images to {1}'.format(len(img_name_dict), sub_img_path))


def create_index(dataset):
    """
    create index
    :param dataset:
    :return:
    """
    print('creating index...')
    anns, cats, imgs = {}, {}, {}
    img_anns, cate_imgs = defaultdict(list), defaultdict(list)
    if 'annotations' in dataset:
        for ann in dataset['annotations']:
            img_anns[ann['image_id']].append(ann)
            anns[ann['id']] = ann

    if 'images' in dataset:
        for img in dataset['images']:
            imgs[img['id']] = img

    if 'categories' in dataset:
        for cat in dataset['categories']:
            cats[cat['id']] = cat

    if 'annotations' in dataset and 'categories' in dataset:
        for ann in dataset['annotations']:
            cate_imgs[ann['category_id']].append(ann['image_id'])
    print('index created!')

    return  anns, cats, imgs, img_anns, cate_imgs


def get_img_per_categorise(category_imgs, num_category=20, per_cate_base_num=18):
    """
    get image id according to per categorise has equal num
    :param cate_img:
    :param pre_cate_base_num:
    :return:
    """
    base_num = per_cate_base_num
    np.random.seed(0)
    random_index = list(np.random.permutation(len(category_imgs)))

    categories_id = []
    for i, index in enumerate(random_index):
        categories_id.append((list(category_imgs.keys())[index]))

    img_id_list = []
    categories_id = categories_id[:num_category]
    for index, category_id in enumerate(categories_id):
        for img_id in category_imgs[category_id]:
            if base_num != 0:
                if img_id not in img_id_list:
                    img_id_list.append(img_id)
                    base_num -= 1
            else:
                base_num = per_cate_base_num
                break

    return img_id_list, categories_id


def get_images_annotaion_info(img_id_list, imgs_raw, img_anns_raw, category_id_list):
    """
    get
    :param img_id_list:
    :param imgs_raw:
    :param img_anns_raw:
    :return:
    """
    images_list = []
    annotations_list = []
    annotation_index = 0
    for img_index, img_id in enumerate(img_id_list):
        img_annotations = img_anns_raw[img_id]
        img_info = imgs_raw[img_id]
        img_info['id'] = img_index
        if len(img_annotations) == 0:
            continue
        else:
            for i, annotation in enumerate(img_annotations):
                if annotation['category_id'] in category_id_list:
                    annotation['id'] = annotation_index
                    annotations_list.append(annotation)
                    annotation_index += 1
        images_list.append(img_info)

    return images_list, annotations_list


if __name__ == "__main__":

    # split coco dataset
    split_coco(img_dir, instance_dir, sub_coco)

    # evaluate split result
    sub_instance_dir = os.path.join(sub_annotations, 'instances.json')
    dataset = json.load(open(sub_instance_dir, 'r'))
    print(dataset.keys())
    images = dataset['images']
    print(len(images))
    print(images[100])
    annotations = dataset['annotations']
    print(annotations[100])
    categories = dataset['categories']
    print(len(categories), categories)
