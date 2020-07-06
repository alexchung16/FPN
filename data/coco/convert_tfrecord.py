#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : convert_tfrecord.py
# @ Description: https://github.com/tensorflow/models/blob/master/research/object_detection/dataset_tools/create_coco_tf_record.py
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 10/12/2019 PM 17:05
# @ Software   : PyCharm
#-------------------------------------------------------

import os
import glob
import cv2 as cv
import numpy as np
import json
from collections import defaultdict
import tensorflow  as tf

from utils.tools import view_bar, makedir

coco_dataset_dir = '/media/alex/AC6A2BDB6A2BA0D6/alex_dataset/sub_coco'
coco_tfrecord_dir = '/media/alex/AC6A2BDB6A2BA0D6/alex_dataset/coco_tfrecord'

tf.app.flags.DEFINE_string('dataset_dir', coco_dataset_dir, 'Voc dir')
tf.app.flags.DEFINE_string('anns_dir', 'Annotations', 'annotation dir')
tf.app.flags.DEFINE_string('image_dir', 'Images', 'image dir')
tf.app.flags.DEFINE_string('save_dir', coco_tfrecord_dir, 'save directory')
tf.app.flags.DEFINE_string('save_name', 'train', 'save name')
tf.app.flags.DEFINE_string('img_format', '.jpg', 'format of image')

FLAGS = tf.app.flags.FLAGS


def convert_coco_to_tfrecord(src_path, save_path, record_capacity=2000, raw_coco=True):
   """

   :param src_path:
   :param save_path:
   :return:
   """

   imgs_path = os.path.join(src_path, FLAGS.image_dir)
   anns_path = os.path.join(src_path, FLAGS.anns_dir)

   # img_name_list = glob.glob(os.path.join(img_path,'*'+FLAGS.img_format))
   annotation_list = glob.glob(os.path.join(anns_path, '*.json'))
   anns, cats, imgs, img_anns, cate_imgs = create_index(annotation_list[0])
   image_id_list = [img_id for img_id in img_anns.keys()]

   remainder_num = len(image_id_list) % record_capacity
   if remainder_num == 0:
      num_record = int(len(image_id_list) / record_capacity)
   else:
      num_record = int(len(image_id_list) / record_capacity) + 1
   for index in range(num_record):
      makedir(save_path)
      record_filename = os.path.join(save_path, f'{index}.record')
      write = tf.io.TFRecordWriter(record_filename)
      if index < num_record - 1:
         sub_img_id_list = image_id_list[index * record_capacity: (index + 1) * record_capacity]
      else:
         sub_img_id_list = image_id_list[(index * record_capacity): (index * record_capacity + remainder_num)]

      num_samples = 0
      for index, img_id in enumerate(sub_img_id_list):
         try:
            # get gtbox_label
            gtbox_label = read_json_gtbox_label(img_anns[img_id])
            # get image name
            if raw_coco:
               img_name = '0' * (12 - len(str(img_id))) + f'{img_id}.{FLAGS.img_format}'
            else:
               img_name = '{0}.jpg'.format(img_id)

            img_path = os.path.join(imgs_path, img_name)

            # load image
            bgr_image = cv.imread(img_path)
            # BGR TO RGB
            rgb_image = cv.cvtColor(bgr_image, cv.COLOR_BGR2RGB)
            img_height = rgb_image.shape[0]
            img_width = rgb_image.shape[1]

            image_record = serialize_example(image=rgb_image, img_height=img_height, img_width=img_width,
                                             img_depth=3, filename=img_name, gtbox_label=gtbox_label)
            write.write(record=image_record)

            num_samples += 1
            view_bar(message='\nConversion progress', num=num_samples, total=len(img_anns))

         except Exception as e:
            print(e)
            continue
      write.close()
      print('There are {0} samples convert to {1}'.format(num_samples, save_path))


def create_index(json_path):
   """
   create index
   :param dataset:
   :return:
   """
   dataset = json.load(open(json_path, 'r'))
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

   return anns, cats, imgs, img_anns, cate_imgs


def read_json_gtbox_label(img_anns):
   """

   :param dataset_dict:
   :param img_id:
   :return:
   """
   # gtbox_label = np.zeros((0, 5), dtype=tf.float32)
   gtbox_label = []
   for annotation in img_anns:
      bbox = annotation['bbox']

      x, y, w, h = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
      # (x,y, w, h) -> (x_min, y_min, x_max, y_max)
      x_min, y_min, x_max, y_max = int(np.rint(x)), int(np.rint(y)), int(np.rint(x + w)), int(np.rint(y + h))
      label = annotation['category_id']
      gt_bbox = [x_min, y_min, x_max, y_max]
      gt_bbox.append(label)
      gtbox_label.append(gt_bbox)

   gtbox_label = np.array(gtbox_label, dtype=np.int32)  # [x1, y1. x2, y2, label]

   # xmin, ymin, xmax, ymax, label = gtbox_label[:, 0], gtbox_label[:, 1], gtbox_label[:, 2], gtbox_label[:, 3], \
   #                                 gtbox_label[:, 4]
   # gtbox_label = np.transpose(np.stack([ymin, xmin, ymax, xmax, label], axis=0))

   return gtbox_label


def serialize_example(image, img_height, img_width, img_depth, filename, gtbox_label):
   """
   create a tf.Example message to be written to a file
   :param label: label info
   :param image: image content
   :param filename: image name
   :return:
   """
   # create a dict mapping the feature name to the tf.Example compatible
   # image_shape = tf.image.decode_jpeg(image_string).eval().shape
   feature = {
      'image': _bytes_feature(image.tobytes()),
      'height': _int64_feature(img_height),
      'width': _int64_feature(img_width),
      'depth': _int64_feature(img_depth),
      'filename': _bytes_feature(filename.encode()),
      'gtboxes_and_label': _bytes_feature(gtbox_label.tobytes()),
      'num_objects': _int64_feature(gtbox_label.shape[0])
   }
   # create a feature message using tf.train.Example
   example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
   return example_proto.SerializeToString()


def _int64_feature(value):
   return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
   return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


if __name__ == "__main__":
   # test json
   convert_coco_to_tfrecord(src_path=coco_dataset_dir, save_path=coco_tfrecord_dir, raw_coco=False)


