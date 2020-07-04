#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : coco_parse.py
# @ Description: reference https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoDemo.ipynb
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/1/16  AM 9:45
# @ Software   : PyCharm
#-------------------------------------------------------

import os
import pylab
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from pycocotools.coco import COCO


# data_type = 'train2017'
data_type = 'val2017'
dataset_dir = '/media/alex/AC6A2BDB6A2BA0D6/alex_dataset/COCO_2017'

img_dir = os.path.join(dataset_dir, data_type)
instance_dir = '{0}/annotations_trainval2017/annotations/instances_{1}.json'.format(dataset_dir, data_type)
keypoint_dir = '{0}/annotations_trainval2017/annotations/person_keypoints_{1}.json'.format(dataset_dir, data_type)
caption_dir = '{0}/annotations_trainval2017/annotations/captions_{1}.json'.format(dataset_dir, data_type)



if __name__ == "__main__":


    #-------------------------------------------------------------------------------------------------
    #                                      test part 1: instance test
    #-------------------------------------------------------------------------------------------------

    #++++++++++++++++++++++++++++initialize COCO api for instance annotations+++++++++++++++++++++++++
    coco = COCO(instance_dir)

    #++++++++++++++++++++++++++++display coco dataset info++++++++++++++++++++++++++++++++++++++++++++

    print(coco.info())
    # description: COCO 2017 Dataset
    # url: http://cocodataset.org
    # version: 1.0
    # year: 2017
    # contributor: COCO Consortium
    # date_created: 2017/09/01

    coco_dataset = coco.dataset
    print(coco_dataset.keys()) # dict_keys(['info', 'licenses', 'images', 'annotations', 'categories'])
    #++++++++++++++++++++++++++display coco categories and super categories++++++++++++++++++++++++++++

    cate_id = coco.getCatIds()
    print('categories: ', coco_dataset['categories'])
    print('-' * 40)
    print('cate id: ', cate_id)
    print('-' * 40)
    categories_info = coco.loadCats(cate_id)
    print('categories info: ', categories_info)
    print('-' * 40)
    super_categories = set([cate['supercategory'] for cate in categories_info])
    sub_categories = [cate.pop('supercategory') for cate in categories_info]

    print('sub categories: ', sub_categories)
    print('-' * 40)
    print('super categories: ', super_categories)
    print('-' * 40)
    #+++++++++++++++++++++++++get all image contain given categories+++++++++++++++++++++++++++++++++++++++

    cate_id = coco.getCatIds(catNms=['person', 'vehicle', 'skateboard'])
    img_id =  coco.getImgIds(catIds=cate_id)[12]
    img_info = coco.loadImgs(img_id)[0]
    print(img_info.keys())  # dict_keys(['license', 'file_name', 'coco_url', 'height', 'width', 'date_captured', 'flickr_url', 'id'])

    # img_name = '0'*(12-len(str(img_id))) + '{0}.jpg'.format(str(img_id) )
    img = io.imread(os.path.join(img_dir, img_info['file_name']))
    plt.imshow(img)
    plt.axis('off')

    #++++++++++++++++++++++++++++load and show instance annotation++++++++++++++++++++++++++++++++++++++
    annotation_id = coco.getAnnIds(imgIds=img_id, catIds=cate_id, iscrowd=None)
    annotations = coco.loadAnns(annotation_id)
    print(annotations[0].keys()) # ['segmentation', 'area', 'iscrowd', 'image_id', 'bbox', 'category_id', 'id']
    coco.showAnns(annotations, draw_bbox=True)

    # -------------------------------------------------------------------------------------------------
    #                                     test part 2:  keypoint test
    # -------------------------------------------------------------------------------------------------
    coco_kps = COCO(keypoint_dir)

    annotations_kps_id = coco_kps.getAnnIds(imgIds=img_id, catIds=cate_id, iscrowd=None)
    annotations_kps = coco_kps.loadAnns(annotations_kps_id)
    coco_kps.showAnns(annotations_kps)

    # -------------------------------------------------------------------------------------------------
    #                                     test part 3:  caption test
    # -------------------------------------------------------------------------------------------------

    coco_caps = COCO(caption_dir)
    annIds = coco_caps.getAnnIds(imgIds=img_id)
    anns = coco_caps.loadAnns(annIds)
    coco_caps.showAnns(anns)

    plt.show()