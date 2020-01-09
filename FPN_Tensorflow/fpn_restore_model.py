#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : fpn_restore_model.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2019/12/31 PM 3:35
# @ Software   : PyCharm
#-------------------------------------------------------


import os
import tensorflow as tf

original_pretrain_model_dir = '/home/alex/Documents/pretraing_model/fpn'
basebone_model_dir = os.path.join(original_pretrain_model_dir, 'resnet_101')
pretrain_model_dir = os.path.join(original_pretrain_model_dir, 'fpn')


if __name__ == "__main__":

    init = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())
    with tf.Session() as sess:
        # images, class_num = sess.run([images, class_num])
        sess.run(init)

        saver = tf.train.import_meta_graph(os.path.join(pretrain_model_dir, 'voc_80000model.ckpt.meta'))
        # restore method 1
        saver.restore(sess, save_path=tf.train.latest_checkpoint(pretrain_model_dir))
        # restore method 2
        # saver.restore(sess, save_path=os.path.join(pretrain_model_dir, 'voc_200000model.ckpt'))

        model_variable = tf.model_variables()
        for var in model_variable:
            print(var.name, var.shape)

        # print(sess.run(['Fast-RCNN/reg_fc/biases:0']))