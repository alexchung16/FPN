#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : eval.py
# @ Description:
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2019/12/22 AM 11:11
# @ Software   : PyCharm
#-------------------------------------------------------

import os
import time
import numpy as np
import cv2 as cv
import tensorflow as tf
import argparse
import pickle

from libs.configs import cfgs
from libs.box_utils import draw_box_in_img
from libs.networks.models import FPN
from libs.eval_libs.eval_voc import voc_evaluate_detections


class Evaluate():
    """
    evaluate model
    """
    def __init__(self, base_network_name, pretrain_model_dir, save_path, draw_img=False):
        self.base_network_name = base_network_name
        self.pretrain_model_dir = pretrain_model_dir
        self.draw_img = draw_img
        self.object_bbox_save_path = os.path.join(save_path, 'bbox_pickle')
        self.detect_bbox_save_path = os.path.join(save_path, 'detect_bbox')
        self.detect_net = FPN(base_network_name=base_network_name, is_training=False)


    def execute_evaluate(self, img_dir, annotation_dir, eval_num=None):
        """
        execute evaluate operation
        :param img_dir: evaluate image dir
        :param annotation_dir: the dir save annotation
        :param eval_num: the number of evaluate image
        :return:
        """
        # construct image path list
        format_list = ('.jpg', '.png', '.jpeg', '.tif', '.tiff')
        img_name_list = [img_name for img_name in os.listdir(img_dir) if img_name.endswith(format_list)]
        assert len(img_name_list) != 0, \
            "test_dir has no images there. Note that, we only support image format of {0}".format(format_list)

        if eval_num is None:
            eval_img_list = img_name_list
        else:
            # select specialize number image
            eval_img_list = img_name_list[: eval_num]

        self.exucute_detect(img_dir=img_dir, img_name_list=eval_img_list)

        # load all boxes
        with open(os.path.join(self.object_bbox_save_path, 'detections.pkl'), 'rb') as f:
            all_boxes = pickle.load(f)

        mAP = voc_evaluate_detections(all_boxes=all_boxes,
                                      annotation_path=annotation_dir,
                                      img_name_list=eval_img_list,
                                      detect_bbox_save_path=self.detect_bbox_save_path)

        return mAP

    def exucute_detect(self, img_dir, img_name_list):
        """
        execute object detect
        :param detect_net: detect network
        :param img_list: the image dir of detect
        :return:
        """

        detection_boxes, detection_scores, detection_category = self.detect_net.inference()

        # config gpu to growth train
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        init_op = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer()
        )

        with tf.Session(config=config) as sess:
            sess.run(init_op)

            if self.pretrain_model_dir is not None:
                # restore pretrain weight
                restorer = tf.train.Saver()
                restore_ckpt = tf.train.latest_checkpoint(pretrain_model_dir)
                restorer.restore(sess, restore_ckpt)

            else:
                restorer, restore_ckpt = self.detect_net.get_restorer()
                restorer.restore(sess, save_path=restore_ckpt)
            print('Successful restore model from {0}'.format(restore_ckpt))

            # +++++++++++++++++++++++++++++++++++++start detect+++++++++++++++++++++++++++++++++++++++++++++++++++++=++
            all_boxes = []
            img_path_list = [os.path.join(img_dir, img_name) for img_name in img_name_list]
            for index, img_name in enumerate(img_path_list):
                bgr_img = cv.imread(img_name)
                raw_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2RGB)
                resized_img = self.image_process(raw_img)
                # expend dimension
                image_batch = tf.expand_dims(input=resized_img, axis=0)  # (1, None, None, 3)

                start_time = time.time()

                feed_dict = self.detect_net.fill_feed_dict(image_feed=image_batch.eval())
                resized_img, detected_boxes, detected_scores, detected_categories = \
                    sess.run([resized_img, detection_boxes, detection_scores, detection_category], feed_dict=feed_dict)  # convert channel from BGR to RGB (cv is BGR)
                end_time = time.time()
                print("{} cost time : {} ".format(img_name, (end_time - start_time)))

                # draw object image
                if self.draw_img:
                    # select object according to threshold
                    object_indices = detected_scores >= cfgs.SHOW_SCORE_THRSHOLD
                    object_scores = detected_scores[object_indices]
                    object_boxes = detected_boxes[object_indices]
                    object_categories = detected_categories[object_indices]

                    final_detections = draw_box_in_img.draw_boxes_with_label_and_scores(img_array=resized_img,
                                                                                        boxes=object_boxes,
                                                                                        labels=object_categories,
                                                                                        scores=object_scores)
                    final_detections = cv.cvtColor(final_detections, cv.COLOR_RGB2BGR)
                    object_boxes = self.bbox_resize(bbox=object_boxes,
                                                    inputs_shape=resized_img.shape[1: 3],
                                                    target_shape=raw_img.shape[1: 3])

                # resize boxes and image shape size to raw input image
                detected_boxes = self.bbox_resize(bbox=detected_boxes,
                                                  inputs_shape=(resized_img.shape[0], resized_img.shape[1]),
                                                  target_shape=(raw_img.shape[0], resized_img.shape[1]))

                # construct detect array for evaluation
                detect_bbox_label = np.hstack((detected_categories.reshape(-1, 1).astype(np.int32),
                                             detected_scores.reshape(-1, 1),
                                              detected_boxes))

                all_boxes.append(detect_bbox_label)

            # dump bbox to local
            makedir(self.object_bbox_save_path)
            with open(os.path.join(self.object_bbox_save_path, 'detections.pkl'), 'wb') as fw:
                pickle.dump(all_boxes, fw)

    def image_process(self, image):
        """
        image_process
        :param image:
        :return:
        """
        image = tf.cast(image, dtype=tf.float32)
        # image resize
        image = self.short_side_resize(img_tensor=image,
                                       target_shortside_len=cfgs.IMG_SHORT_SIDE_LEN,
                                       target_length_limitation=cfgs.IMG_MAX_LENGTH)
        # image white
        image = image - tf.constant(cfgs.PIXEL_MEAN)

        return image

    def short_side_resize(self, img_tensor, target_shortside_len, target_length_limitation=1200):
        '''
        :param img_tensor:[h, w, c], gtboxes_and_label:[-1, 5].  gtboxes: [xmin, ymin, xmax, ymax]
        :param target_shortside_len:
        :param length_limitation: set max length to avoid OUT OF MEMORY
        :return:
        '''
        img_h, img_w = tf.shape(img_tensor)[0], tf.shape(img_tensor)[1]
        new_h, new_w = tf.cond(tf.less(img_h, img_w),
                               true_fn=lambda: (target_shortside_len,
                                                self.max_length_limitation(target_shortside_len * img_w // img_h,
                                                                      target_length_limitation)),
                               false_fn=lambda: (self.max_length_limitation(target_shortside_len * img_h // img_w,
                                                                       target_length_limitation),
                                                 target_shortside_len))
        # expend dimension to 3 for resize
        img_tensor = tf.expand_dims(img_tensor, axis=0)
        img_tensor = tf.image.resize_bilinear(img_tensor, [new_h, new_w])
        # recover dimension
        img_tensor = tf.squeeze(img_tensor, axis=0)

        return img_tensor

    def max_length_limitation(self, length, length_limitation):
        """
        get limitation length
        :param length:
        :param length_limitation:
        :return:
        """
        return tf.cond(tf.less(length, length_limitation),
                       true_fn=lambda: length,
                       false_fn=lambda: length_limitation)

    def bbox_resize(self, bbox, inputs_shape, target_shape):
        """
        resize bbox
        :param bbox: [x_min, y_min, x_max, y_max]
        :param inputs_shape: [src_h, src_w]
        :param target_shape: [dst_h, dst_w]
        :return:
        """
        x_min, y_min, x_max, y_max = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]

        x_min = x_min * target_shape[1] / inputs_shape[1]
        y_min = y_min * target_shape[0] / inputs_shape[0]

        x_max = x_max * target_shape[1] / inputs_shape[1]
        y_max = y_max * target_shape[0] / inputs_shape[0]

        # object_boxes = np.stack([x_min, y_min, x_max, y_max], axis=1)
        object_boxes = np.transpose(np.stack([x_min, y_min, x_max, y_max]))
        return object_boxes

def makedir(path):
    """
    create dir
    :param path:
    :return:
    """
    if os.path.exists(path) is False:
        os.makedirs(path)

if __name__ == "__main__":
    base_network_name = 'resnet_v1_101'
    dataset_dir = '/media/alex/AC6A2BDB6A2BA0D6/alex_dataset/pascal_voc/test/VOC2007'
    image_dir = os.path.join(dataset_dir, 'JPEGImages')
    annotation_dir = os.path.join(dataset_dir, 'Annotations')

    pretrain_model_dir = None
    evaluate = Evaluate(base_network_name=base_network_name,
                        pretrain_model_dir=pretrain_model_dir,
                        save_path=cfgs.EVALUATE_DIR,
                        draw_img=False)

    mAP = evaluate.execute_evaluate(img_dir=image_dir,
                                    annotation_dir=annotation_dir,
                                    eval_num=10)

    print(mAP)




