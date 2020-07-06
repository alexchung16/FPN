#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @ File coco_pascal_tfrecord.py
# @ Description :
# @ Author alexchung
# @ Time 10/12/2019 PM 17:05

import os
import glob
import numpy as np
import tensorflow  as tf
import xml.etree.cElementTree as ET
import cv2 as cv

from utils.tools import makedir, view_bar

original_dataset_dir = '/media/alex/AC6A2BDB6A2BA0D6/alex_dataset/Pascal_VOC_2012/VOCtrainval/VOCdevkit/VOC2012'

tfrecord_dir = '/media/alex/AC6A2BDB6A2BA0D6/alex_dataset/pascal_tfrecord'
# tfrecord_dir = os.path.join(original_dataset_dir, 'tfrecords')


NAME_LABEL_MAP = {
        'back_ground': 0,
        'aeroplane': 1,
        'bicycle': 2,
        'bird': 3,
        'boat': 4,
        'bottle': 5,
        'bus': 6,
        'car': 7,
        'cat': 8,
        'chair': 9,
        'cow': 10,
        'diningtable': 11,
        'dog': 12,
        'horse': 13,
        'motorbike': 14,
        'person': 15,
        'pottedplant': 16,
        'sheep': 17,
        'sofa': 18,
        'train': 19,
        'tvmonitor': 20
    }


tf.app.flags.DEFINE_string('dataset_dir', original_dataset_dir, 'Voc dir')
tf.app.flags.DEFINE_string('xml_dir', 'Annotations', 'xml dir')
tf.app.flags.DEFINE_string('image_dir', 'JPEGImages', 'image dir')
tf.app.flags.DEFINE_string('save_name', 'train', 'save name')
tf.app.flags.DEFINE_string('save_dir', tfrecord_dir, 'save name')
tf.app.flags.DEFINE_string('img_format', '.jpg', 'format of image')
tf.app.flags.DEFINE_string('dataset', 'car', 'dataset')
FLAGS = tf.app.flags.FLAGS


try:
    if os.path.exists(original_dataset_dir) is False:
        raise IOError('dataset is not exist please check the path')
except FileNotFoundError as e:
    print(e)
finally:
    makedir(tfrecord_dir)

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def read_xml_gtbox_and_label(xml_path):
    """
    read gtbox(ground truth) and label from xml
    :param xml_path: the path of voc xml
    :return: a list contains gtboxes and labels, shape is [num_of_gtboxes, 5],
           and has [xmin, ymin, xmax, ymax, label] in a per row
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    img_width = None
    img_height = None
    box_list = []
    for child_of_root in root:
        if child_of_root.tag == 'size':
            for child_item in child_of_root:
                if child_item.tag == 'width':
                    img_width = int(child_item.text)
                if child_item.tag == 'height':
                    img_height = int(child_item.text)

        if child_of_root.tag == 'object':
            label = None
            for child_item in child_of_root:
                if child_item.tag == 'name':
                    # label = NAME_LABEL_MAP[child_item.text]
                    label=NAME_LABEL_MAP[child_item.text]
                if child_item.tag == 'bndbox':
                    tmp_box = []
                    xmin, ymin, xmax, ymax = None, None, None, None
                    for node in child_item:
                        if node.tag == 'xmin':
                            xmin = int(eval(node.text))
                        if node.tag == 'ymin':
                            ymin = int(eval(node.text))
                        if node.tag == 'xmax':
                            xmax = int(eval(node.text))
                        if node.tag == 'ymax':
                            ymax = int(eval(node.text))
                    tmp_box = [xmin, ymin, xmax, ymax]
                    # tmp_box.append()
                    assert label is not None, 'label is none, error'
                    tmp_box.append(label)
                    box_list.append(tmp_box)

    gtbox_label = np.array(box_list, dtype=np.int32)  # [x1, y1. x2, y2, label]

    xmin, ymin, xmax, ymax, label = gtbox_label[:, 0], gtbox_label[:, 1], gtbox_label[:, 2], gtbox_label[:, 3], \
                                    gtbox_label[:, 4]
    gtbox_label = np.transpose(np.stack([ymin, xmin, ymax, xmax, label], axis=0))  # [ymin, xmin, ymax, xmax, label]

    return img_height, img_width, gtbox_label


def convert_pascal_to_tfrecord(img_path, xml_path, save_path, record_capacity=2000):
    """
    convert pascal dataset to rfrecord
    :param img_path:
    :param xml_path:
    :param save_path:
    :param record_capacity:
    :return:
    """
    # record_file = os.path.join(FLAGS.save_dir, FLAGS.save_name+'.tfrecord')

    img_xml_list = [os.path.basename(xml_file) for xml_file in glob.glob(os.path.join(xml_path, '*.xml'))]
    img_name_list = [xml.split('.')[0] + FLAGS.img_format for xml in img_xml_list]

    remainder_num = len(img_name_list) % record_capacity
    if remainder_num == 0:
        num_record = int(len(img_name_list) / record_capacity)
    else:
        num_record = int(len(img_name_list) / record_capacity) + 1

    num_samples = 0
    for index in range(num_record):
        record_filename = os.path.join(save_path, f'{index}.record')
        write = tf.io.TFRecordWriter(record_filename)
        if index < num_record - 1:
            sub_img_list = img_name_list[index * record_capacity: (index + 1) * record_capacity]
            sub_xml_list = img_xml_list[index * record_capacity: (index + 1) * record_capacity]
        else:
            sub_img_list = img_name_list[(index * record_capacity): (index * record_capacity + remainder_num)]
            sub_xml_list = img_xml_list[(index * record_capacity): (index * record_capacity + remainder_num)]

        try:
            for img_name, img_xml in zip(sub_img_list, sub_xml_list):
                img_file = os.path.join(img_path, img_name)
                xml_file = os.path.join(xml_path, img_xml)

                img_height, img_width, gtbox_label = read_xml_gtbox_and_label(xml_file)
                # note image channel format of opencv if rgb
                bgr_image = cv.imread(img_file)
                # BGR TO RGB
                rgb_image = cv.cvtColor(bgr_image, cv.COLOR_BGR2RGB)

                image_record = serialize_example(image=rgb_image, img_height=img_height, img_width=img_width, img_depth=3,
                                                 filename=img_name, gtbox_label=gtbox_label)
                write.write(record=image_record)

                num_samples += 1
                view_bar(message='\nConversion progress', num=num_samples, total=len(img_name_list))
        except Exception as e:
            print(e)
            continue
        write.close()
    print('\nThere are {0} samples convert to {1}'.format(num_samples, save_path))


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
        'image': _bytes_feature(image.tostring()),
        'height': _int64_feature(img_height),
        'width': _int64_feature(img_width),
        'depth':_int64_feature(img_depth),
        'filename': _bytes_feature(filename.encode()),
        'gtboxes_and_label': _bytes_feature(gtbox_label.tostring()),
        'num_objects': _int64_feature(gtbox_label.shape[0])
    }
    # create a feature message using tf.train.Example
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


if __name__ == "__main__":
    image_path = os.path.join(FLAGS.dataset_dir, FLAGS.image_dir)
    xml_path = os.path.join(FLAGS.dataset_dir, FLAGS.xml_dir)

    convert_pascal_to_tfrecord(img_path=image_path, xml_path=xml_path, save_path=FLAGS.save_dir, record_capacity=4000)




