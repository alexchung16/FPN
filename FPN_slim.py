#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : FPN_slim.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2019/12/31 AM 10:30
# @ Software   : PyCharm
#-------------------------------------------------------

import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from FPN.fpn_util import cfgs
from FPN.fpn_util.resnet import ResNet
from FPN.fpn_util import anchor_utils


class FPN():
    """
    FPN(Feature Pyramid Network)
    """
    def __init__(self, base_network_name='resnet_v1_101', weight_decay=0.0001, batch_norm_decay=0.997,
                 batch_norm_epsilon=1e-5, batch_norm_scale=True, is_training=True):
        self.base_network_name = base_network_name
        self.weight_decay = weight_decay
        self.batch_norm_decay = batch_norm_decay
        self.batch_norm_epsilon = batch_norm_epsilon
        self.batch_norm_scale = batch_norm_scale

        self.num_anchors_per_location = len(cfgs.ANCHOR_SCALES) * len(cfgs.ANCHOR_RATIOS)
        self.resnet = ResNet(weight_decay=weight_decay,
                             batch_norm_decay=batch_norm_decay,
                             batch_norm_epsilon=batch_norm_epsilon,
                             batch_norm_scale=batch_norm_scale)
        self.is_training = is_training


    def build_base_network(self, inputs_batch):
        """
        base network
        :param inputs_batch:
        :return:
        """
        if self.base_network_name.startswith('resnet_v1'):
            return self.resnet.resnet_base(inputs_batch, scope_name=self.base_network_name, is_training=self.is_training)
        else:
            raise ValueError('Sry, we only support resnet_50 or resnet_101')

    def build_rpn_network(self, feature_list):
        """

        :param feature_list:
        :return:
        """
        with tf.variable_scope('build_rpn',
                               regularizer=slim.l2_regularizer(self.weight_decay)):

            fpn_cls_score =[]
            fpn_box_pred = []
            for level_name, p in zip(cfgs.LEVLES, feature_list):
                if cfgs.SHARE_HEADS:
                    reuse_flag = None if level_name==cfgs.LEVLES[0] else True
                    scope_list=['rpn_conv/3x3', 'rpn_cls_score', 'rpn_bbox_pred']
                else:
                    reuse_flag = None
                    scope_list= ['rpn_conv/3x3_%s' % level_name, 'rpn_cls_score_%s' % level_name, 'rpn_bbox_pred_%s' % level_name]
                rpn_conv3x3 = slim.conv2d(
                    p, 512, [3, 3],
                    trainable=self.is_training, weights_initializer=cfgs.INITIALIZER, padding="SAME",
                    activation_fn=tf.nn.relu,
                    scope=scope_list[0],
                    reuse=reuse_flag)
                rpn_cls_score = slim.conv2d(rpn_conv3x3, self.num_anchors_per_location*2, [1, 1], stride=1,
                                            trainable=self.is_training, weights_initializer=cfgs.INITIALIZER,
                                            activation_fn=None, padding="VALID",
                                            scope=scope_list[1],
                                            reuse=reuse_flag)
                rpn_box_pred = slim.conv2d(rpn_conv3x3, self.num_anchors_per_location*4, [1, 1], stride=1,
                                           trainable=self.is_training, weights_initializer=cfgs.BBOX_INITIALIZER,
                                           activation_fn=None, padding="VALID",
                                           scope=scope_list[2],
                                           reuse=reuse_flag)
                rpn_box_pred = tf.reshape(rpn_box_pred, [-1, 4])
                rpn_cls_score = tf.reshape(rpn_cls_score, [-1, 2])

                fpn_cls_score.append(rpn_cls_score)
                fpn_box_pred.append(rpn_box_pred)
            fpn_cls_score = tf.concat(fpn_cls_score, axis=0, name='fpn_cls_score')
            fpn_box_pred = tf.concat(fpn_box_pred, axis=0, name='fpn_box_pred')
            return fpn_cls_score, fpn_box_pred

    def fpn(self, img_batch, gtboxes_batch):
        """
        construct fpn network
        :param input_img_batch:
        :param gtboxes_batch:
        :return:
        """
        if self.is_training:
            # ensure shape is [M, 5]
            gtboxes_batch = tf.reshape(gtboxes_batch, [-1, 5])
            gtboxes_batch = tf.cast(gtboxes_batch, tf.float32)

        img_shape = tf.shape(img_batch)

        # step 1 build base network
        # get Pyramid feature list
        P_list = self.build_base_network(inputs_batch=img_batch)  #[P2, P3, P4, P5, P6]

        # step 2 build fpn
        fpn_cls_score, fpn_box_pred = self.build_rpn_network(P_list)
        fpn_cls_prob = slim.softmax(fpn_cls_score, scope='fpn_cls_prob')

        # step 3 generate anchor
        all_anchors = []
        for i in range(len(cfgs.LEVLES)):
            level_name, p = cfgs.LEVLES[i], P_list[i]
            # feature shape
            p_height, p_width = tf.shape(p)[1], tf.shape(p)[2]
            feature_height = tf.cast(p_height, dtype=tf.float32)
            feature_width = tf.cast(p_width, dtype=tf.float32)

            anchors = anchor_utils.make_anchors(base_anchor_size=cfgs.BASE_ANCHOR_SIZE_LIST[i],
                                                anchor_scales=cfgs.ANCHOR_SCALES,
                                                anchor_ratios=cfgs.ANCHOR_RATIOS,
                                                feature_height=feature_height,
                                                feature_width=feature_width,
                                                stride=cfgs.ANCHOR_STRIDE_LIST[i],
                                                name="make_anchors_for%s" % level_name)
            all_anchors.append(anchors)
        all_anchors = tf.concat(all_anchors, axis=0, name='all_anchors_of_FPN')

        # step 4 postprocess rpn proposals. such as: decode, clip and NMS








