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
import tensorflow as tf
import tensorflow.contrib.slim as slim

from libs.configs import cfgs
from libs.networks import models
from data.pascal.read_tfrecord import dataset_tfrecord


original_dataset_dir = '/home/alex/Documents/datasets/Pascal_VOC_2012/VOCtrainval/VOCdevkit_test'
tfrecord_dir = os.path.join(original_dataset_dir, 'tfrecords')

pretrain_model_dir = '/home/alex/Documents/pretraing_model/fpn'

model_path = os.path.join(os.getcwd(), 'models')
pb_model_name = os.path.join(model_path, 'faster_rcnn.pb')
summary_path = os.path.join(os.getcwd(), 'logs')


tf.app.flags.DEFINE_string('record_file', tfrecord_dir, 'Directory to put the training data.')
tf.app.flags.DEFINE_bool('restore_from_rpn', False, 'if True, just restore base net and rpn net weights from FPN pretrain model.')
tf.app.flags.DEFINE_bool('is_pretrain', False, 'if True, restore weight from faster rcnn pretrain model, else just restore basenet pretriain model.')

tf.app.flags.DEFINE_string('pretrain_model_dir', pretrain_model_dir, 'pretrain model dir.')
tf.app.flags.DEFINE_string('model_path', model_path, 'path of store model.')
tf.app.flags.DEFINE_string('summary_path', summary_path, 'direct of summary logs.')
tf.app.flags.DEFINE_string('pb_model_name', pb_model_name, 'pb model_name.')
FLAGS = tf.app.flags.FLAGS


def makedir(path):
    """
    create dir
    :param path:
    :return:
    """
    if os.path.exists(path) is False:
        os.makedirs(path)
        print(print('{0} has been created'.format(path)))

makedir(model_path)
makedir(summary_path)


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
                            record_file=FLAGS.record_file,
                            is_training=True)
        # gtboxes_and_label_batch = tf.reshape(gtboxes_and_label_batch, [-1, 5])

    # list as many types of layers as possible, even if they are not used now
    total_loss = fpn.loss
    train_op = fpn.train
    global_step = fpn.global_step
    #++++++++++++++++++++++++++++++++++++++++++++++++build loss function++++++++++++++++++++++++++++++++++++++++++++++
    summary_op = tf.summary.merge_all()

    restorer, restore_ckpt = fpn.get_restore(pretrain_model_dir=FLAGS.pretrain_model_dir,
                                                     restore_from_rpn=FLAGS.restore_from_rpn,
                                                     is_pretrain=FLAGS.is_pretrain)
    saver = tf.train.Saver(max_to_keep=30)

    # support growth train
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    )
    with tf.Session() as sess:
        sess.run(init_op)

        if not restorer is None:
            restorer.restore(sess, save_path=restore_ckpt)

        model_variables = slim.get_model_variables()
        for var in model_variables:
            print(var.name, var.shape)
        # build summary write
        summary_writer = tf.summary.FileWriter(FLAGS.summary_path, graph=sess.graph)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)
        #++++++++++++++++++++++++++++++++++++++++start training+++++++++++++++++++++++++++++++++++++++++++++++++++++
        for step in range(cfgs.MAX_ITERATION):

            img_name, image, gtboxes_and_label, num_objects= \
                sess.run([img_name_batch, img_batch, gtboxes_and_label_batch, num_objects_batch])
            feed_dict = fpn.fill_feed_dict(image_feed=image,
                                           gtboxes_feed=gtboxes_and_label)
            training_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

            if step % cfgs.SHOW_TRAIN_INFO_INTE != 0 and step % cfgs.SMRY_ITER != 0:
                _, global_stepnp = sess.run([train_op, global_step], feed_dict=feed_dict)
            else:
                if step % cfgs.SHOW_TRAIN_INFO_INTE == 0 and step % cfgs.SMRY_ITER != 0:
                    start_time = time.time()

                    _, global_stepnp, totalLoss = \
                        sess.run(
                            [train_op, global_step, total_loss], feed_dict=feed_dict)

                    end_time = time.time()
                    print(""" {}: step{}    image_name:{} |\t total_loss:{} |\t per_cost_time:{}s""" \
                          .format(training_time, global_stepnp, str(img_name[0]), totalLoss, (end_time - start_time)))
                else:
                    if step % cfgs.SMRY_ITER == 0:
                        _, global_stepnp, summary_str = sess.run([train_op, global_step, summary_op], feed_dict=feed_dict)
                        summary_writer.add_summary(summary_str, global_stepnp)
                        summary_writer.flush()

            if (step > 0 and step % cfgs.SAVE_WEIGHTS_INTE == 0) or (step == cfgs.MAX_ITERATION - 1):
                save_ckpt = os.path.join(FLAGS.model_path, 'voc_' + str(global_stepnp) + 'model.ckpt')
                saver.save(sess, save_ckpt)
                print(' weights had been saved')

        coord.request_stop()
        coord.join(threads)



if __name__ == "__main__":

    train()