#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : FPN_inference_placeholder.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2019/12/31 下午5:00
# @ Software   : PyCharm
#-------------------------------------------------------

import os
import time
import numpy as np
import cv2 as cv
import tensorflow as tf

from libs.configs import cfgs
from libs.box_utils import draw_box_in_img
from libs.networks.models import FPN
from utils.tools import makedir, view_bar


class ObjectInference():
    def __init__(self, base_network_name, pretrain_model_dir):
        self.base_network_name = base_network_name
        self.pretrain_model_dir = pretrain_model_dir
        self.detect_net = FPN(base_network_name=base_network_name, is_training=False)
        # self._R_MEAN = 123.68
        # self._G_MEAN = 116.779
        # self._B_MEAN = 103.939

    def exucute_detect(self, image_path, save_path):
        """
        execute object detect
        :param detect_net:
        :param image_path:
        :return:
        """
        input_image = tf.placeholder(dtype=tf.uint8, shape=(None, None, 3), name='inputs_images')

        resize_img = self.image_process(input_image)
        # expend dimension
        image_batch = tf.expand_dims(input=resize_img, axis=0)  # (1, None, None, 3)

        self.detect_net.images_batch = image_batch
        # img_shape = tf.shape(inputs_img)
        # load detect network
        detection_boxes, detection_scores, detection_category = self.detect_net.inference()

        # restore pretrain weight
        restorer, restore_ckpt = self.detect_net.get_restorer()
        # config gpu to growth train
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        init_op = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer()
        )
        with tf.Session(config=config) as sess:
            sess.run(init_op)

            if restorer is not None:
                restorer.restore(sess, save_path=restore_ckpt)
                print('*'*80 +'\nSuccessful restore model from {0}\n'.format(restore_ckpt) + '*'*80)

            # construct image path list
            format_list = ('.jpg', '.png', '.jpeg', '.tif', '.tiff')
            if os.path.isfile(image_path):
                image_name_list = [image_path]
            else:
                image_name_list = [img_name for img_name in os.listdir(image_path)
                              if img_name.endswith(format_list) and os.path.isfile(os.path.join(image_path, img_name))]

            assert len(image_name_list) != 0
            print("test_dir has no imgs there. Note that, we only support img format of {0}".format(format_list))
            #+++++++++++++++++++++++++++++++++++++start detect+++++++++++++++++++++++++++++++++++++++++++++++++++++=++
            makedir(save_path)
            fw = open(os.path.join(save_path, 'detect_bbox.txt'), 'w')

            for index, img_name in enumerate(image_name_list):

                detect_dict = {}
                bgr_img = cv.imread(os.path.join(image_path, img_name))
                rgb_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2RGB) # convert channel from BGR to RGB (cv is BGR)

                start_time = time.perf_counter()
                # image resize and white process
                # construct feed_dict
                feed_dict = {input_image: rgb_img}
                resized_img, detected_boxes, detected_scores, detected_categories = \
                    sess.run([resize_img, detection_boxes, detection_scores, detection_category],
                             feed_dict=feed_dict)
                end_time = time.perf_counter()

                # select object according to threshold
                object_indices = detected_scores >= cfgs.SHOW_SCORE_THRSHOLD
                object_scores = detected_scores[object_indices]
                object_boxes = detected_boxes[object_indices]
                object_categories = detected_categories[object_indices]

                final_detections_img = draw_box_in_img.draw_boxes_with_label_and_scores(resized_img,
                                                                                    boxes=object_boxes,
                                                                                    labels=object_categories,
                                                                                    scores=object_scores)
                final_detections_img = cv.cvtColor(final_detections_img, cv.COLOR_RGB2BGR)
                cv.imwrite(os.path.join(save_path, img_name), final_detections_img)
                # resize boxes and image according to raw input image
                raw_h, raw_w = rgb_img.shape[0], rgb_img.shape[1]
                resized_h, resized_w = resized_img.shape[1], resized_img.shape[2]
                x_min, y_min, x_max, y_max = object_boxes[:, 0], object_boxes[:, 1], object_boxes[:, 2], \
                                             object_boxes[:, 3]
                x_min = x_min * raw_w / resized_w
                y_min = y_min * raw_h / resized_h
                x_max = x_max * raw_w / resized_w
                y_max = y_max * raw_h / resized_h

                object_boxes = np.stack([x_min, y_min, x_max, y_max], axis=1)
                # final_detections= cv.resize(final_detections[:, :, ::-1], (raw_w, raw_h))

                # recover to raw size
                detect_dict['score'] = object_scores
                detect_dict['boxes'] = object_boxes
                detect_dict['categories'] = object_categories
                # convert from RGB to BG
                fw.write(f'\n{img_name}')
                for score, boxes, categories in zip(object_scores, object_boxes, object_categories):
                    fw.write('\n\tscore:' + str(score))
                    fw.write('\tbboxes:' + str(boxes))
                    fw.write('\tcategories:' + str(categories))

                view_bar('{} image cost {} second'.format(img_name, (end_time - start_time)), index + 1,
                               len(image_name_list))

            fw.close()

    def image_process(self, img):
        """
        image_process
        :param image:
        :return:
        """
        image = tf.cast(img, dtype=tf.float32)
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
        pass


if __name__ == "__main__":
    base_network_name = 'resnet_v1_101'
    # pretrain_model_dir = cfgs.MODEL_CKPT
    pretrained_model_dir = '/home/alex/Documents/pretrain_model/fpn/fpn'

    inference = ObjectInference(base_network_name=base_network_name,
                                pretrain_model_dir=pretrained_model_dir)

    inference.exucute_detect(image_path='./demos', save_path=cfgs.INFERENCE_SAVE_PATH)

    # for img_name, detect_info in img_detections.items():