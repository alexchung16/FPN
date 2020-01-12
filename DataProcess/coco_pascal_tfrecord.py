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

# original_dataset_dir = 'F:/datasets/Pascal VOC 2012/VOCdevkit/VOC2012'

original_dataset_dir = '/home/alex/Documents/datasets/Pascal_VOC_2012/VOCtrainval/VOCdevkit_test'
tfrecord_dir = os.path.join(original_dataset_dir, 'tfrecords')

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

def makedir(path):
    """
    create dir
    :param path:
    :return:
    """
    if os.path.exists(path) is False:
        os.makedirs(path)

try:
    if os.path.exists(original_dataset_dir) is False:
        print('dataset is not exist please check the path')
    else:
        if os.path.exists(tfrecord_dir) is False:
            os.mkdir(tfrecord_dir)
            print('{0} has been created'.format(tfrecord_dir))
except FileNotFoundError as e:
    print(e)

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


def convert_pascal_to_tfrecord(image_path, xml_path, save_path):
    """
    convert pascal dataset to rfrecord
    :param img_dir:
    :param annotation_dir:
    :return: None
    """
    # record_file = os.path.join(FLAGS.save_dir, FLAGS.save_name+'.tfrecord')
    write = tf.io.TFRecordWriter(save_path)

    for n, xml in enumerate(glob.glob(os.path.join(xml_path, '*.xml'))):
        img_name = os.path.basename(xml).split('.')[0] + FLAGS.img_format
        img_path = os.path.join(image_path, img_name)
        if not os.path.exists(img_path):
            print('{} is not exist!'.format(img_path))
            continue

        img_height, img_width, gtbox_label = read_xml_gtbox_and_label(xml)
        # note image channel format of opencv if rgb
        bgr_image = cv.imread(img_path)
        # BGR TO RGB
        rgb_image = cv. cvtColor(bgr_image, cv.COLOR_BGR2RGB)


        image_record = serialize_example(image=rgb_image, img_height=img_height, img_width=img_width, img_depth=3,
                                         filename=img_name, gtbox_label=gtbox_label)
        write.write(record=image_record)
    write.close()
    print('Conversion is complete!')


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
    save_path = os.path.join(FLAGS.save_dir, FLAGS.save_name + '.tfrecord')

    convert_pascal_to_tfrecord(image_path=image_path, xml_path=xml_path, save_path=save_path)




