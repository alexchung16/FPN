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

from libs.configs import cfgs
from libs.networks.resnet_util import ResNet
from libs.box_utils import anchor_utils
from libs.box_utils import boxes_utils
from libs.box_utils import encode_and_decode
from libs.box_utils import show_box_in_tensor
from libs.detect_operations.anchor_target_layer import anchor_target_layer
from libs.detect_operations.proposal_target_layer import proposal_target_layer
from libs.losses import losses


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
        # # is_training flag
        self.global_step = tf.train.get_or_create_global_step()

        self.images_batch = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, None, None, 3],
                                                       name="input_images")
        # y [None, upper_left_x, upper_left_y, down_right_x, down_right_y]

        self.gtboxes_batch = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, None, 5],
                                                          name="gtboxes_label")
    def inference(self):
        """
        inference function
        :return:
        """
        if self.is_training:
            # list as many types of layers as possible, even if they are not used now
            with slim.arg_scope(self.fpn_arg_scope()):
                final_bbox, final_scores, final_category = self.fpn(img_batch=self.images_batch,
                                                                            gtboxes_batch=self.gtboxes_batch)
            self.losses()
            # ------add detect summary----------------
            gtboxes_and_label = tf.reshape(self.gtboxes_batch, [-1, 5])
            gtboxes_in_img = show_box_in_tensor.draw_boxes_with_categories(img_batch=self.images_batch,
                                                                           boxes=gtboxes_and_label[:, :-1],
                                                                           labels=gtboxes_and_label[:, -1])
            if cfgs.ADD_BOX_IN_TENSORBOARD:
                detections_in_img = show_box_in_tensor.draw_boxes_with_categories_and_scores(
                    img_batch=self.images_batch,
                    boxes=final_bbox,
                    labels=final_category,
                    scores=final_scores)
                tf.summary.image('Compare/final_detection', detections_in_img)
            tf.summary.image('Compare/gtboxes', gtboxes_in_img)
        else:
            final_bbox, final_scores, final_category = self.fpn(img_batch=self.images_batch,
                                                                        gtboxes_batch=self.gtboxes_batch)
        return final_bbox, final_scores, final_category

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

    def postprocess_rpn_proposals(self, rpn_bbox_pred, rpn_cls_prob, img_shape, anchors, is_training):
        '''

        :param rpn_bbox_pred: [-1, 4]
        :param rpn_cls_prob: [-1, 2]
        :param img_shape:
        :param anchors:[-1, 4]
        :param is_training:
        :return:
        '''
        if is_training:
            pre_nms_topN = cfgs.RPN_TOP_K_NMS_TRAIN
            post_nms_topN = cfgs.RPN_MAXIMUM_PROPOSAL_TARIN
            # pre_nms_topN = cfgs.FPN_TOP_K_PER_LEVEL_TRAIN
            # post_nms_topN = pre_nms_topN
        else:
            pre_nms_topN = cfgs.RPN_TOP_K_NMS_TEST
            post_nms_topN = cfgs.RPN_MAXIMUM_PROPOSAL_TEST
            # pre_nms_topN = cfgs.FPN_TOP_K_PER_LEVEL_TEST
            # post_nms_topN = pre_nms_topN

        nms_thresh = cfgs.RPN_NMS_IOU_THRESHOLD

        cls_prob = rpn_cls_prob[:, 1]

        # 1. decode boxes
        decode_boxes = encode_and_decode.decode_boxes(encoded_boxes=rpn_bbox_pred,
                                                      reference_boxes=anchors,
                                                      scale_factors=cfgs.ANCHOR_SCALE_FACTORS)

        # 2. clip to img boundaries
        decode_boxes = boxes_utils.clip_boxes_to_img_boundaries(decode_boxes=decode_boxes,
                                                                img_shape=img_shape)

        # 3. get top N to NMS
        if pre_nms_topN > 0:
            pre_nms_topN = tf.minimum(pre_nms_topN, tf.shape(decode_boxes)[0], name='avoid_unenough_boxes')
            cls_prob, top_k_indices = tf.nn.top_k(cls_prob, k=pre_nms_topN)
            decode_boxes = tf.gather(decode_boxes, top_k_indices)

        # 4. NMS
        keep = tf.image.non_max_suppression(
            boxes=decode_boxes,
            scores=cls_prob,
            max_output_size=post_nms_topN,
            iou_threshold=nms_thresh)

        final_boxes = tf.gather(decode_boxes, keep)
        final_probs = tf.gather(cls_prob, keep)

        return final_boxes, final_probs

    def assign_levels(self, all_rois, labels=None, bbox_targets=None):
        '''

        :param all_rois:
        :param labels:
        :param bbox_targets:
        :return:
        '''
        with tf.name_scope('assign_levels'):
            # all_rois = tf.Print(all_rois, [tf.shape(all_rois)], summarize=10, message='ALL_ROIS_SHAPE*****')
            xmin, ymin, xmax, ymax = tf.unstack(all_rois, axis=1)

            h = tf.maximum(0., ymax - ymin)
            w = tf.maximum(0., xmax - xmin)

            levels = tf.floor(4. + tf.log(tf.sqrt(w * h + 1e-8) / 224.0) / tf.log(2.))  # 4 + log_2(***)
            # use floor instead of round

            min_level = int(cfgs.LEVLES[0][-1])
            max_level = min(5, int(cfgs.LEVLES[-1][-1]))
            levels = tf.maximum(levels, tf.ones_like(levels) * min_level)  # level minimum is 2
            levels = tf.minimum(levels, tf.ones_like(levels) * max_level)  # level maximum is 5

            levels = tf.stop_gradient(tf.reshape(levels, [-1]))

            def get_rois(levels, level_i, rois, labels, bbox_targets):

                level_i_indices = tf.reshape(tf.where(tf.equal(levels, level_i)), [-1])
                # level_i_indices = tf.Print(level_i_indices, [tf.shape(tf.where(tf.equal(levels, level_i)))[0]], message="SHAPE%d***"%level_i,
                #                            summarize=10)
                tf.summary.scalar('LEVEL/LEVEL_{0}_rois_NUM'.format(level_i), tf.shape(level_i_indices)[0])
                level_i_rois = tf.gather(rois, level_i_indices)

                if self.is_training:
                    if cfgs.CUDA9:
                        # Note: for cuda 9
                        level_i_rois = tf.stop_gradient(level_i_rois)
                        level_i_labels = tf.gather(labels, level_i_indices)

                        level_i_targets = tf.gather(bbox_targets, level_i_indices)
                    else:

                        # Note: for cuda 8
                        level_i_rois = tf.stop_gradient(tf.concat([level_i_rois, [[0, 0, 0., 0.]]], axis=0))
                        # to avoid the num of level i rois is 0.0, which will broken the BP in tf

                        level_i_labels = tf.gather(labels, level_i_indices)
                        level_i_labels = tf.stop_gradient(tf.concat([level_i_labels, [0]], axis=0))

                        level_i_targets = tf.gather(bbox_targets, level_i_indices)
                        level_i_targets = tf.stop_gradient(tf.concat([level_i_targets,
                                                                      tf.zeros(shape=(1, 4 * (cfgs.CLASS_NUM + 1)),
                                                                               dtype=tf.float32)], axis=0))

                    return level_i_rois, level_i_labels, level_i_targets
                else:
                    if not cfgs.CUDA9:
                        # Note: for cuda 8
                        level_i_rois = tf.concat([level_i_rois, [[0, 0, 0., 0.]]], axis=0)
                    return level_i_rois, None, None

            rois_list = []
            labels_list = []
            targets_list = []
            for i in range(min_level, max_level + 1):
                P_i_rois, P_i_labels, P_i_targets = get_rois(levels, level_i=i, rois=all_rois,
                                                             labels=labels,
                                                             bbox_targets=bbox_targets)
                rois_list.append(P_i_rois)
                labels_list.append(P_i_labels)
                targets_list.append(P_i_targets)

            if self.is_training:
                all_labels = tf.concat(labels_list, axis=0)
                all_targets = tf.concat(targets_list, axis=0)
                return rois_list, all_labels, all_targets
            else:
                return rois_list  # [P2_rois, P3_rois, P4_rois, P5_rois] Note: P6 do not assign rois

    def postprocess_fastrcnn(self, rois, bbox_ppred, scores, img_shape):
        '''

        :param rois:[-1, 4]
        :param bbox_ppred: [-1, (cfgs.Class_num+1) * 4]
        :param scores: [-1, cfgs.Class_num + 1]
        :return:
        '''

        with tf.name_scope('postprocess_fastrcnn'):
            rois = tf.stop_gradient(rois)
            scores = tf.stop_gradient(scores)
            bbox_ppred = tf.reshape(bbox_ppred, [-1, cfgs.CLASS_NUM + 1, 4])
            bbox_ppred = tf.stop_gradient(bbox_ppred)

            bbox_pred_list = tf.unstack(bbox_ppred, axis=1)
            score_list = tf.unstack(scores, axis=1)

            allclasses_boxes = []
            allclasses_scores = []
            categories = []
            for i in range(1, cfgs.CLASS_NUM+1):

                # 1. decode boxes in each class
                tmp_encoded_box = bbox_pred_list[i]
                tmp_score = score_list[i]
                tmp_decoded_boxes = encode_and_decode.decode_boxes(encoded_boxes=tmp_encoded_box,
                                                                   reference_boxes=rois,
                                                                   scale_factors=cfgs.ROI_SCALE_FACTORS)
                # 2. clip to img boundaries
                tmp_decoded_boxes = boxes_utils.clip_boxes_to_img_boundaries(decode_boxes=tmp_decoded_boxes,
                                                                             img_shape=img_shape)

                # 3. NMS
                keep = tf.image.non_max_suppression(
                    boxes=tmp_decoded_boxes,
                    scores=tmp_score,
                    max_output_size=cfgs.FAST_RCNN_NMS_MAX_BOXES_PER_CLASS,
                    iou_threshold=cfgs.FAST_RCNN_NMS_IOU_THRESHOLD)

                perclass_boxes = tf.gather(tmp_decoded_boxes, keep)
                perclass_scores = tf.gather(tmp_score, keep)

                allclasses_boxes.append(perclass_boxes)
                allclasses_scores.append(perclass_scores)
                categories.append(tf.ones_like(perclass_scores) * i)

            final_boxes = tf.concat(allclasses_boxes, axis=0)
            final_scores = tf.concat(allclasses_scores, axis=0)
            final_category = tf.concat(categories, axis=0)

            if self.is_training:
                '''
                in training. We should show the detecitons in the tensorboard. So we add this.
                '''
                kept_indices = tf.reshape(tf.where(tf.greater_equal(final_scores, cfgs.SHOW_SCORE_THRSHOLD)), [-1])

                final_boxes = tf.gather(final_boxes, kept_indices)
                final_scores = tf.gather(final_scores, kept_indices)
                final_category = tf.gather(final_category, kept_indices)

        return final_boxes, final_scores, final_category

    def roi_pooling(self, feature_maps, rois, img_shape, scope):
        '''
        Here use roi warping as roi_pooling

        :param featuremaps_dict: feature map to crop
        :param rois: shape is [-1, 4]. [x1, y1, x2, y2]
        :return:
        '''

        with tf.variable_scope('ROI_Warping_'+scope):
            img_h, img_w = tf.cast(img_shape[1], tf.float32), tf.cast(img_shape[2], tf.float32)
            N = tf.shape(rois)[0]
            x1, y1, x2, y2 = tf.unstack(rois, axis=1)

            normalized_x1 = x1 / img_w
            normalized_x2 = x2 / img_w
            normalized_y1 = y1 / img_h
            normalized_y2 = y2 / img_h

            normalized_rois = tf.transpose(
                tf.stack([normalized_y1, normalized_x1, normalized_y2, normalized_x2]), name='get_normalized_rois')

            normalized_rois = tf.stop_gradient(normalized_rois)

            cropped_roi_features = tf.image.crop_and_resize(feature_maps, normalized_rois,
                                                            box_ind=tf.zeros(shape=[N, ],
                                                                             dtype=tf.int32),
                                                            crop_size=[cfgs.ROI_SIZE, cfgs.ROI_SIZE],
                                                            name='CROP_AND_RESIZE'
                                                            )
            roi_features = slim.max_pool2d(cropped_roi_features,
                                           [cfgs.ROI_POOL_KERNEL_SIZE, cfgs.ROI_POOL_KERNEL_SIZE],
                                           stride=cfgs.ROI_POOL_KERNEL_SIZE)

        return roi_features

    def build_fastrcnn(self, P_list, rois_list, img_shape):

        with tf.variable_scope('Fast-RCNN'):
            # 5. ROI Pooling
            with tf.variable_scope('rois_pooling'):
                pooled_features_list = []
                for level_name, p, rois in zip(cfgs.LEVLES, P_list, rois_list):  # exclude P6_rois
                    # p = tf.Print(p, [tf.shape(p)], summarize=10, message=level_name+'SHPAE***')
                    pooled_features = self.roi_pooling(feature_maps=p, rois=rois, img_shape=img_shape,
                                                       scope=level_name)
                    pooled_features_list.append(pooled_features)

                pooled_features = tf.concat(pooled_features_list, axis=0) # [minibatch_size, H, W, C]

            # 6. inferecne rois in Fast-RCNN to obtain fc_flatten features
            if self.base_network_name.startswith('resnet'):
                fc_flatten = self.resnet.restnet_head(inputs=pooled_features,
                                                 is_training=self.is_training,
                                                 scope_name=self.base_network_name)
            else:
                raise NotImplementedError('only support resnet')

            # 7. cls and reg in Fast-RCNN
            with slim.arg_scope([slim.fully_connected], weights_regularizer=slim.l2_regularizer(cfgs.WEIGHT_DECAY)):

                cls_score = slim.fully_connected(fc_flatten,
                                                 num_outputs=cfgs.CLASS_NUM+1,
                                                 weights_initializer=cfgs.INITIALIZER,
                                                 activation_fn=None, trainable=self.is_training,
                                                 scope='cls_fc')

                bbox_pred = slim.fully_connected(fc_flatten,
                                                 num_outputs=(cfgs.CLASS_NUM+1)*4,
                                                 weights_initializer=cfgs.BBOX_INITIALIZER,
                                                 activation_fn=None, trainable=self.is_training,
                                                 scope='reg_fc')
                # for convient. It also produce (cls_num +1) bboxes

                cls_score = tf.reshape(cls_score, [-1, cfgs.CLASS_NUM+1])
                bbox_pred = tf.reshape(bbox_pred, [-1, 4*(cfgs.CLASS_NUM+1)])

                return bbox_pred, cls_score

    def build_loss(self, rpn_box_pred, rpn_bbox_targets, rpn_cls_score, rpn_labels,
                   bbox_pred, bbox_targets, cls_score, labels):
        '''

        :param rpn_box_pred: [-1, 4]
        :param rpn_bbox_targets: [-1, 4]
        :param rpn_cls_score: [-1]
        :param rpn_labels: [-1]
        :param bbox_pred: [-1, 4*(cls_num+1)]
        :param bbox_targets: [-1, 4*(cls_num+1)]
        :param cls_score: [-1, cls_num+1]
        :param labels: [-1]
        :return:
        '''
        with tf.variable_scope('build_loss') as sc:
            with tf.variable_scope('rpn_loss'):

                rpn_bbox_loss = losses.smooth_l1_loss_rpn(bbox_pred=rpn_box_pred,
                                                          bbox_targets=rpn_bbox_targets,
                                                          label=rpn_labels,
                                                          sigma=cfgs.RPN_SIGMA)
                # rpn_cls_loss:
                # rpn_cls_score = tf.reshape(rpn_cls_score, [-1, 2])
                # rpn_labels = tf.reshape(rpn_labels, [-1])
                # ensure rpn_labels shape is [-1]
                rpn_select = tf.reshape(tf.where(tf.not_equal(rpn_labels, -1)), [-1])
                rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score, rpn_select), [-1, 2])
                rpn_labels = tf.reshape(tf.gather(rpn_labels, rpn_select), [-1])
                rpn_cls_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score,
                                                                                             labels=rpn_labels))

                rpn_cls_loss = rpn_cls_loss * cfgs.RPN_CLASSIFICATION_LOSS_WEIGHT
                rpn_bbox_loss = rpn_bbox_loss * cfgs.RPN_LOCATION_LOSS_WEIGHT

            with tf.variable_scope('FastRCNN_loss'):
                if not cfgs.FAST_RCNN_MINIBATCH_SIZE == -1:
                    bbox_loss = losses.smooth_l1_loss_rcnn(bbox_pred=bbox_pred,
                                                           bbox_targets=bbox_targets,
                                                           label=labels,
                                                           num_classes=cfgs.CLASS_NUM + 1,
                                                           sigma=cfgs.FASTRCNN_SIGMA)

                    cls_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=cls_score,
                        labels=labels))  # beacause already sample before
                else:
                    ''' 
                    applying OHEM here
                    '''
                    print("TRAIN WITH OHEM ...")

                    cls_loss, bbox_loss = losses.sum_ohem_loss(cls_score=cls_score,
                                                               label=labels,
                                                               bbox_targets=bbox_targets,
                                                               bbox_pred=bbox_pred,
                                                               num_ohem_samples=256,
                                                               num_classes=cfgs.CLASS_NUM + 1)
                cls_loss = cls_loss * cfgs.FAST_RCNN_CLASSIFICATION_LOSS_WEIGHT
                bbox_loss = bbox_loss * cfgs.FAST_RCNN_LOCATION_LOSS_WEIGHT
            loss_dict = {
                'rpn_cls_loss': rpn_cls_loss,
                'rpn_loc_loss': rpn_bbox_loss,
                'fastrcnn_cls_loss': cls_loss,
                'fastrcnn_loc_loss': bbox_loss
            }
        return loss_dict

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
        with tf.variable_scope('postprocess_FPN'):
            rois, roi_scores = self.postprocess_rpn_proposals(rpn_bbox_pred=fpn_box_pred,
                                                         rpn_cls_prob=fpn_cls_prob,
                                                         img_shape=img_shape,
                                                         anchors=all_anchors,
                                                         is_training=self.is_training)
            if self.is_training:
                rois_in_img = show_box_in_tensor.draw_boxes_with_scores(img_batch=img_batch,
                                                                        boxes=rois,
                                                                        scores=roi_scores)
                tf.summary.image('all_rpn_rois', rois_in_img)

                score_gre_05 = tf.reshape(tf.where(tf.greater_equal(roi_scores, 0.5)), [-1])
                score_gre_05_rois = tf.gather(rois, score_gre_05)
                score_gre_05_score = tf.gather(roi_scores, score_gre_05)
                score_gre_05_in_img = show_box_in_tensor.draw_boxes_with_scores(img_batch=img_batch,
                                                                                boxes=score_gre_05_rois,
                                                                                scores=score_gre_05_score)
                tf.summary.image('score_greater_05_rois', score_gre_05_in_img)
        if self.is_training:
            with tf.variable_scope('sample_anchors_minibatch'):
                fpn_labels, fpn_bbox_targets =tf.py_func(anchor_target_layer,
                                                         [gtboxes_batch, img_shape, all_anchors],
                                                         [tf.float32, tf.float32])
                fpn_bbox_targets = tf.reshape(fpn_bbox_targets, [-1, 4])
                fpn_labels = tf.to_int32(fpn_labels, name='to_int32')
                fpn_labels = tf.reshape(fpn_labels, [-1])
                self.add_anchor_img_smry(img_batch, all_anchors, fpn_labels)

            #------------------------------------------add summary-----------------------------------------------------
            fpn_cls_category = tf.argmax(fpn_cls_prob, axis=1)
            kept_rpppn = tf.reshape(tf.where(tf.not_equal(fpn_labels, -1)), [-1])
            fpn_cls_category = tf.gather(fpn_cls_category, kept_rpppn)
            acc = tf.reduce_mean(tf.to_float(tf.equal(fpn_cls_category,
                                                      tf.to_int64(tf.gather(fpn_labels, kept_rpppn)))))
            tf.summary.scalar('ACC/fpn_accuracy', acc)

            with tf.control_dependencies([fpn_labels]):
                with tf.variable_scope('sample_RCNN_minibatch'):
                    rois, labels, bbox_targets = tf.py_func(proposal_target_layer,
                                                            [rois, gtboxes_batch],
                                                            [tf.float32, tf.float32, tf.float32])
                    rois = tf.reshape(rois, [-1, 4])
                    labels = tf.to_int32(labels)
                    labels = tf.reshape(labels, [-1])
                    bbox_targets = tf.reshape(bbox_targets, [-1, 4 * (cfgs.CLASS_NUM + 1)])
                    self.add_roi_batch_img_smry(img_batch, rois, labels)


        if self.is_training:
            rois_list, labels, bbox_targets = self.assign_levels(all_rois=rois,
                                                                 labels=labels,
                                                                 bbox_targets=bbox_targets)
        else:
            rois_list = self.assign_levels(all_rois=rois)  # rois_list: [P2_rois, P3_rois, P4_rois, P5_rois]

        # -------------------------------------------------------------------------------------------------------------#
        #                                            Fast-RCNN                                                         #
        # -------------------------------------------------------------------------------------------------------------#

        # 5. build Fast-RCNN
        # rois = tf.Print(rois, [tf.shape(rois)], 'rois shape', summarize=10)
        bbox_pred, cls_score = self.build_fastrcnn(P_list=P_list, rois_list=rois_list,
                                                   img_shape=img_shape)
        # bbox_pred shape: [-1, 4*(cls_num+1)].
        # cls_score shape： [-1, cls_num+1]

        cls_prob = slim.softmax(cls_score, 'cls_prob')


        # ----------------------------------------------add smry-------------------------------------------------------
        if self.is_training:
            cls_category = tf.argmax(cls_prob, axis=1)
            fast_acc = tf.reduce_mean(tf.to_float(tf.equal(cls_category, tf.to_int64(labels))))
            tf.summary.scalar('ACC/fast_acc', fast_acc)

        rois = tf.concat(rois_list, axis=0, name='concat_rois')
        #  6. postprocess_fastrcnn
        if not self.is_training:
            return self.postprocess_fastrcnn(rois=rois, bbox_ppred=bbox_pred, scores=cls_prob, img_shape=img_shape)
        else:
            '''
            when train. We need build Loss
            '''
            self.loss_dict = self.build_loss(rpn_box_pred=fpn_box_pred,
                                        rpn_bbox_targets=fpn_bbox_targets,
                                        rpn_cls_score=fpn_cls_score,
                                        rpn_labels=fpn_labels,
                                        bbox_pred=bbox_pred,
                                        bbox_targets=bbox_targets,
                                        cls_score=cls_score,
                                        labels=labels)

            final_bbox, final_scores, final_category = self.postprocess_fastrcnn(rois=rois,
                                                                                 bbox_ppred=bbox_pred,
                                                                                 scores=cls_prob,
                                                                                 img_shape=img_shape)
            return final_bbox, final_scores, final_category

    def add_anchor_img_smry(self, img, anchors, labels):

        positive_anchor_indices = tf.reshape(tf.where(tf.greater_equal(labels, 1)), [-1])
        negative_anchor_indices = tf.reshape(tf.where(tf.equal(labels, 0)), [-1])

        positive_anchor = tf.gather(anchors, positive_anchor_indices)
        negative_anchor = tf.gather(anchors, negative_anchor_indices)

        pos_in_img = show_box_in_tensor.only_draw_boxes(img_batch=img,
                                                        boxes=positive_anchor)
        neg_in_img = show_box_in_tensor.only_draw_boxes(img_batch=img,
                                                        boxes=negative_anchor)

        tf.summary.image('positive_anchor', pos_in_img)
        tf.summary.image('negative_anchors', neg_in_img)

    def add_roi_batch_img_smry(self, img, rois, labels):
        positive_roi_indices = tf.reshape(tf.where(tf.greater_equal(labels, 1)), [-1])

        negative_roi_indices = tf.reshape(tf.where(tf.equal(labels, 0)), [-1])

        pos_roi = tf.gather(rois, positive_roi_indices)
        neg_roi = tf.gather(rois, negative_roi_indices)


        pos_in_img = show_box_in_tensor.only_draw_boxes(img_batch=img,
                                                               boxes=pos_roi)
        neg_in_img = show_box_in_tensor.only_draw_boxes(img_batch=img,
                                                               boxes=neg_roi)
        tf.summary.image('pos_rois', pos_in_img)
        tf.summary.image('neg_rois', neg_in_img)

    def fill_feed_dict(self, image_feed, gtboxes_feed=None):
        """
        generate feed dict
        :param image_feed:
        :param gtboxes_feed:
        :param is_training:
        :return:
        """
        if self.is_training:
            feed_dict = {
                self.images_batch: image_feed,
                self.gtboxes_batch: gtboxes_feed
            }
        else:
            feed_dict = {
                self.images_batch: image_feed
            }
        return feed_dict

    def get_restorer(self):
        """
        restore pretrain weight
        :param pretrain_model_dir:
        :param is_pretrain:
        :return:
        """

        checkpoint_path = tf.train.latest_checkpoint(os.path.join(cfgs.TRAINED_CKPT, cfgs.VERSION))

        if checkpoint_path != None:
            if cfgs.RESTORE_FROM_RPN:
                print('___restore from rpn___')
                model_variables = slim.get_model_variables()
                restore_variables = [var for var in model_variables if not var.name.startswith('FastRCNN_Head')] + \
                                    [slim.get_or_create_global_step()]
                for var in restore_variables:
                    print(var.name)
                restorer = tf.train.Saver(restore_variables)
            else:
                restorer = tf.train.Saver()
            print("model restore from {0}".format(checkpoint_path))
        else:

            model_variables = slim.get_model_variables()
            ckpt_var_dict = {}
            for var in model_variables:
                if var.name.startswith(self.base_network_name):
                    var_name_ckpt = var.op.name
                    ckpt_var_dict[var_name_ckpt] = var
            restore_variables = ckpt_var_dict
            for key, item in restore_variables.items():
                print("var_in_graph: ", item.name)
                print("var_in_ckpt: ", key)

            restorer = tf.compat.v1.train.Saver(restore_variables)
            checkpoint_path = os.path.join(cfgs.PRETRAINED_CKPT, self.base_network_name + '.ckpt')
            print("model restore from {0}".format(checkpoint_path))
            print("restore from pretrained_weighs in IMAGE_NET")

        return restorer, checkpoint_path

    def get_gradients(self, optimizer, loss):
        '''

        :param optimizer:
        :param loss:
        :return:

        return vars and grads that not be fixed
        '''

        return optimizer.compute_gradients(loss)

    def enlarge_gradients_for_bias(self, gradients):

        final_gradients = []
        with tf.variable_scope("Gradient_Mult") as scope:
            for grad, var in gradients:
                scale = 1.0
                if cfgs.MUTILPY_BIAS_GRADIENT and './biases' in var.name:
                    scale = scale * cfgs.MUTILPY_BIAS_GRADIENT
                if not np.allclose(scale, 1.0):
                    grad = tf.multiply(grad, scale)
                final_gradients.append((grad, var))
        return final_gradients

    def losses(self):
        """
        loss operation
        :return:
        """
        # ----------------------------------------------sparse loss---------------------------------------------------
        rpn_location_loss = self.loss_dict['rpn_loc_loss']
        rpn_cls_loss = self.loss_dict['rpn_cls_loss']
        rpn_total_loss = rpn_location_loss + rpn_cls_loss

        fastrcnn_cls_loss = self.loss_dict['fastrcnn_cls_loss']
        fastrcnn_loc_loss = self.loss_dict['fastrcnn_loc_loss']
        fastrcnn_total_loss = fastrcnn_cls_loss + fastrcnn_loc_loss

        weight_decay_loss = tf.add_n(slim.losses.get_regularization_losses())

        total_loss = rpn_total_loss + fastrcnn_total_loss + weight_decay_loss

        tf.summary.scalar('RPN_LOSS/cls_loss', rpn_cls_loss)
        tf.summary.scalar('RPN_LOSS/location_loss', rpn_location_loss)
        tf.summary.scalar('RPN_LOSS/rpn_total_loss', rpn_total_loss)

        tf.summary.scalar('FAST_LOSS/fastrcnn_cls_loss', fastrcnn_cls_loss)
        tf.summary.scalar('FAST_LOSS/fastrcnn_location_loss', fastrcnn_loc_loss)
        tf.summary.scalar('FAST_LOSS/fastrcnn_total_loss', fastrcnn_total_loss)

        tf.summary.scalar('LOSS/total_loss', total_loss)

        return total_loss

    def training(self, total_loss, global_step):
        """
        train operation
        :param loss_dict:
        :param global_step:
        :return:
        """
        # ----------------------------------------------generate optimizer----------------------------------------------
        learning_rate = tf.train.piecewise_constant(global_step,
                                                    boundaries=[np.int64(cfgs.DECAY_STEP[0]),
                                                                np.int64(cfgs.DECAY_STEP[1])],
                                                    values=[cfgs.LR, cfgs.LR / 10., cfgs.LR / 100.])
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=cfgs.MOMENTUM)
        tf.summary.scalar('learning_rate', learning_rate)

        # -----------------------------------------computer gradient----------------------------------------------------
        gradients = self.get_gradients(optimizer, total_loss)

        # enlarge_gradients for bias
        if cfgs.MUTILPY_BIAS_GRADIENT:
            gradients = self.enlarge_gradients_for_bias(gradients)

        if cfgs.GRADIENT_CLIPPING_BY_NORM:
            with tf.name_scope('clip_gradients'):
                gradients = slim.learning.clip_gradient_norms(gradients, cfgs.GRADIENT_CLIPPING_BY_NORM)

        # +++++++++++++++++++++++++++++++++++++++++start train+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # train_op
        train_op = optimizer.apply_gradients(grads_and_vars=gradients,
                                             global_step=global_step)
        return train_op

    def fpn_arg_scope(self):
        with slim.arg_scope([slim.conv2d, slim.conv2d_in_plane, slim.conv2d_transpose, slim.separable_conv2d,
                             slim.fully_connected],
                            weights_regularizer=slim.l2_regularizer((self.weight_decay)),
                            biases_regularizer=tf.no_regularizer,
                            biases_initializer=tf.constant_initializer(0.0)) as sc:
            return sc















