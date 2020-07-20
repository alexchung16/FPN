#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : FPN_train_placeholder.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2019/12/31 PM 3:45
# @ Software   : PyCharm
#-------------------------------------------------------

import os
import time
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from libs.configs import cfgs
from libs.networks import models
from data.pascal.read_tfrecord import dataset_tfrecord
from libs.box_utils import show_box_in_tensor


def train():
    """
    train progress
    :return:
    """
    fpn = models.FPN(base_network_name='resnet_v1_101', is_training=True)
    #-----------------------------------------------read data----------------------------------------------------------
    with tf.name_scope('get_batch'):
        img_name_batch, img_batch, gtboxes_and_label_batch, num_objects_batch = \
            dataset_tfrecord(batch_size=cfgs.BATCH_SIZE,
                            shortside_len=cfgs.IMG_SHORT_SIDE_LEN,
                            length_limitation=cfgs.IMG_MAX_LENGTH,
                            record_file=cfgs.TFRECORD_DIR,
                            is_training=True)
        # gtboxes_and_label_batch = tf.reshape(gtboxes_and_label_batch, [-1, 5])

    final_bbox, final_scores, final_category = fpn.inference()

    rpn_location_loss = fpn.loss_dict['rpn_loc_loss']
    rpn_cls_loss = fpn.loss_dict['rpn_cls_loss']
    rpn_total_loss = rpn_location_loss + rpn_cls_loss

    fastrcnn_cls_loss = fpn.loss_dict['fastrcnn_cls_loss']
    fastrcnn_loc_loss = fpn.loss_dict['fastrcnn_loc_loss']
    fastrcnn_total_loss =  fastrcnn_cls_loss + fastrcnn_loc_loss

    total_loss = rpn_total_loss + fastrcnn_total_loss

    # add final image summary
    gtboxes_in_img = show_box_in_tensor.draw_boxes_with_categories(img_batch=img_batch,
                                                                   boxes=gtboxes_and_label_batch[:, :-1],
                                                                   labels=gtboxes_and_label_batch[:, -1])
    if cfgs.ADD_BOX_IN_TENSORBOARD:
        detections_in_img = show_box_in_tensor.draw_boxes_with_categories_and_scores(img_batch=img_batch,
                                                                                     boxes=final_bbox,
                                                                                     labels=final_category,
                                                                                     scores=final_scores)
        tf.summary.image('Compare/final_detection', detections_in_img)
    tf.summary.image('Compare/gtboxes', gtboxes_in_img)
    #++++++++++++++++++++++++++++++++++++++++++++++++build loss function++++++++++++++++++++++++++++++++++++++++++++++
    global_step = slim.get_or_create_global_step()
    lr = tf.train.piecewise_constant(global_step,
                                     boundaries=[np.int64(cfgs.DECAY_STEP[0]), np.int64(cfgs.DECAY_STEP[1])],
                                     values=[cfgs.LR, cfgs.LR / 10., cfgs.LR / 100.])
    tf.summary.scalar('lr', lr)
    optimizer = tf.train.MomentumOptimizer(lr, momentum=cfgs.MOMENTUM)
    # optimizer = tf.train.AdamOptimizer(lr)

    # ---------------------------------------------------------------------------------------------compute gradients
    gradients = fpn.get_gradients(optimizer, total_loss)

    # enlarge_gradients for bias
    if cfgs.MUTILPY_BIAS_GRADIENT:
        gradients = fpn.enlarge_gradients_for_bias(gradients)

    if cfgs.GRADIENT_CLIPPING_BY_NORM:
        with tf.name_scope('clip_gradients_YJR'):
            gradients = slim.learning.clip_gradient_norms(gradients,
                                                          cfgs.GRADIENT_CLIPPING_BY_NORM)
    # _____________________________________________________________________________________________compute gradients

    # train_op
    train_op = optimizer.apply_gradients(grads_and_vars=gradients,
                                         global_step=global_step)
    summary_op = tf.summary.merge_all()

    restorer, restore_ckpt = fpn.get_restore(pretrain_model_dir=cfgs.PRETRAINED_CKPT)

    saver = tf.train.Saver(max_to_keep=30)

    # support growth train
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    )
    with tf.Session(config=config) as sess:
        sess.run(init_op)

        if not restorer is None:
            restorer.restore(sess, save_path=restore_ckpt)

        model_variables = slim.get_model_variables()
        for var in model_variables:
            print(var.name, var.shape)
        # build summary write
        summary_writer = tf.summary.FileWriter(cfgs.SUMMARY_PATH, graph=sess.graph)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)
        #++++++++++++++++++++++++++++++++++++++++start training+++++++++++++++++++++++++++++++++++++++++++++++++++++
        try:
            if not coord.should_stop():
                for step in range(cfgs.MAX_ITERATION):

                    img_name, image, gtboxes_and_label, num_objects= \
                        sess.run([img_name_batch, img_batch, gtboxes_and_label_batch, num_objects_batch])
                    feed_dict = fpn.fill_feed_dict(image_feed=image,
                                                   gtboxes_feed=gtboxes_and_label)
                    training_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

                    if step % cfgs.SHOW_TRAIN_INFO_INTE != 0 and step % cfgs.SMRY_ITER != 0:
                        _, globalStep = sess.run([train_op, global_step], feed_dict=feed_dict)
                    else:
                        if step % cfgs.SHOW_TRAIN_INFO_INTE == 0 and step % cfgs.SMRY_ITER != 0:
                            start_time = time.time()

                            _, globalStep, rpnLocLoss, rpnClsLoss, rpnTotalLoss, \
                            fastrcnnLocLoss, fastrcnnClsLoss, fastrcnnTotalLoss, totalLoss = \
                                sess.run(
                                    [train_op, global_step, rpn_location_loss, rpn_cls_loss, rpn_total_loss,
                                     fastrcnn_loc_loss, fastrcnn_cls_loss, fastrcnn_total_loss, total_loss],
                                    feed_dict=feed_dict)

                            end_time = time.time()
                            print(""" {}: step{}\t\timage_name:{} |\t rpn_loc_loss:{} |\t rpn_cla_loss:{} |\t rpn_total_loss:{} |
                                      fast_rcnn_loc_loss:{} |\t fast_rcnn_cla_loss:{} |\t fast_rcnn_total_loss:{} |
                                      total_loss:{} |\t per_cost_time:{}s""" \
                                  .format(training_time, globalStep, str(img_name[0]), rpnLocLoss, rpnClsLoss,
                                          rpnTotalLoss, fastrcnnLocLoss, fastrcnnClsLoss, fastrcnnTotalLoss, totalLoss,
                                          (end_time - start_time)))
                        else:
                            if step % cfgs.SMRY_ITER == 0:
                                _, globalStep, summary_str = sess.run([train_op, global_step, summary_op], feed_dict=feed_dict)
                                summary_writer.add_summary(summary_str, globalStep)
                                summary_writer.flush()

                    if (step > 0 and step % cfgs.SAVE_WEIGHTS_INTE == 0) or (step == cfgs.MAX_ITERATION - 1):
                        save_ckpt = os.path.join(cfgs.MODEL_CKPT, 'voc_' + str(globalStep) + 'model.ckpt')
                        saver.save(sess, save_ckpt)
                        print(' weights had been saved')

        except Exception as e:
            # Report exceptions to the coordinator.
            coord.request_stop(e)
        finally:
            coord.request_stop()
            coord.join(threads)
            print('all threads are asked to stop!')


if __name__ == "__main__":

    train()