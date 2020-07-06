#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @ File coco_pascal_tfrecord.py
# @ Description :
# @ Author alexchung
# @ Time 11/12/2019 AM 09:34


import os
import glob
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python_io import tf_record_iterator


# origin_dataset_dir = '/media/alex/AC6A2BDB6A2BA0D6/alex_dataset/pascal_split/val'
tfrecord_dir = '/media/alex/AC6A2BDB6A2BA0D6/alex_dataset/coco_tfrecord'

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94

IMG_SHORT_SIDE_LEN = 600
IMG_MAX_LENGTH = 1000


def read_parse_single_example(serialized_sample, shortside_len, length_limitation, is_training=False):
    """
    parse tensor
    :param image_sample:
    :return:
    """
    # construct feature description
    image_feature_description = {
        'filename': tf.FixedLenFeature([], tf.string),
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'depth': tf.FixedLenFeature([], tf.int64),
        'image': tf.FixedLenFeature([], tf.string),
        'gtboxes_and_label': tf.FixedLenFeature([], tf.string),
        'num_objects': tf.FixedLenFeature([], tf.int64)
    }
    feature = tf.io.parse_single_example(serialized=serialized_sample, features=image_feature_description)

    # parse feature
    image = tf.decode_raw(feature['image'], tf.uint8)
    # shape = tf.cast(feature['shape'], tf.int32)
    height = tf.cast(feature['height'], tf.int32)
    width = tf.cast(feature['width'], tf.int32)
    depth = tf.cast(feature['depth'], tf.int32)
    image = tf.reshape(image, [height, width, depth])
    filename = tf.cast(feature['filename'], tf.string)
    # image augmentation
    # image = augmentation_image(image=image, image_shape=input_shape)
    # parse gtbox
    gtboxes_and_label = tf.decode_raw(feature['gtboxes_and_label'], tf.int32)
    gtboxes_and_label = tf.reshape(gtboxes_and_label, shape=[-1, 5])
    num_objects = tf.cast(feature['num_objects'], tf.int32)

    image, gtboxes_and_label = image_process(image, gtboxes_and_label, shortside_len=shortside_len,
                                             length_limitation=length_limitation, is_training=is_training)
    return image, filename, gtboxes_and_label, num_objects


def image_process(image, gtboxes_and_label, shortside_len, length_limitation, is_training=False):
    """
    image process
    :param image:
    :param gtboxes_and_label:
    :param shortside_len:
    :return:
    """
    img = tf.cast(image, tf.float32)
    if is_training:
        img, gtboxes_and_label = short_side_resize(img_tensor=img, gtboxes_and_label=gtboxes_and_label,
                                                                    target_shortside_len=shortside_len,
                                                                    length_limitation=length_limitation)
        img, gtboxes_and_label = random_flip_left_right(img_tensor=img,
                                                                         gtboxes_and_label=gtboxes_and_label)

    else:
        img, gtboxes_and_label = short_side_resize(img_tensor=img, gtboxes_and_label=gtboxes_and_label,
                                                                    target_shortside_len=shortside_len,
                                                                    length_limitation=length_limitation)
    image = img - tf.constant([_R_MEAN, _G_MEAN, _B_MEAN], dtype=tf.float32)
    # image = image_whitened(img)
    return image, gtboxes_and_label


def image_whitened(image, means=(_R_MEAN, _G_MEAN, _B_MEAN)):
    """Subtracts the given means from each image channel.
    Returns:
        the centered image.
    """
    if image.get_shape().ndims != 3:
        raise ValueError('Input must be of size [height, width, C>0]')
    num_channels = image.get_shape().as_list()[-1]

    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')

    mean = tf.constant(means, dtype=image.dtype)
    image = image - mean
    return image


def max_length_limitation(length, length_limitation):
    return tf.cond(tf.less(length, length_limitation),
                   true_fn=lambda: length,
                   false_fn=lambda: length_limitation)


def short_side_resize(img_tensor, gtboxes_and_label, target_shortside_len, length_limitation=1200):
    '''
    :param img_tensor:[h, w, c], gtboxes_and_label:[-1, 5].  gtboxes: [xmin, ymin, xmax, ymax]
    :param target_shortside_len:
    :param length_limitation: set max length to avoid OUT OF MEMORY
    :return:
    '''
    img_h, img_w = tf.shape(img_tensor)[0], tf.shape(img_tensor)[1]
    new_h, new_w = tf.cond(tf.less(img_h, img_w),
                           true_fn=lambda: (target_shortside_len,
                                            max_length_limitation(target_shortside_len * img_w // img_h, length_limitation)),
                           false_fn=lambda: (max_length_limitation(target_shortside_len * img_h // img_w, length_limitation),
                                             target_shortside_len))
    # expend dimension to 3 for resize
    img_tensor = tf.expand_dims(img_tensor, axis=0)
    img_tensor = tf.image.resize_bilinear(img_tensor, [new_h, new_w])

    xmin, ymin, xmax, ymax, label = tf.unstack(gtboxes_and_label, axis=1)

    new_xmin, new_ymin = xmin * new_w // img_w, ymin * new_h // img_h
    new_xmax, new_ymax = xmax * new_w // img_w, ymax * new_h // img_h
    img_tensor = tf.squeeze(img_tensor, axis=0)  # ensure image tensor rank is 3

    return img_tensor, tf.transpose(tf.stack([new_xmin, new_ymin, new_xmax, new_ymax, label], axis=0))


def flip_left_to_right(img_tensor, gtboxes_and_label):

    h, w = tf.shape(img_tensor)[0], tf.shape(img_tensor)[1]

    img_tensor = tf.image.flip_left_right(img_tensor)

    xmin, ymin, xmax, ymax, label = tf.unstack(gtboxes_and_label, axis=1)
    new_xmax = w - xmin
    new_xmin = w - xmax

    return img_tensor, tf.transpose(tf.stack([new_xmin, ymin, new_xmax, ymax, label], axis=0))


def random_flip_left_right(img_tensor, gtboxes_and_label):
    img_tensor, gtboxes_and_label= tf.cond(tf.less(tf.random_uniform(shape=[], minval=0, maxval=1), 0.5),
                                            lambda: flip_left_to_right(img_tensor, gtboxes_and_label),
                                            lambda: (img_tensor, gtboxes_and_label))
    return img_tensor,  gtboxes_and_label


def dataset_tfrecord(record_file, shortside_len, length_limitation, batch_size=1, epoch=5, shuffle=True, is_training=False):
    """
    construct iterator to read image
    :param record_file:
    :return:
    """
    record_list = []
    # check record file format
    if os.path.isfile(record_file):
        record_list = [record_file]
    else:
        for filename in os.listdir(record_file):
            record_list.append(os.path.join(record_file, filename))
    # # use dataset read record file
    record_dataset = tf.data.TFRecordDataset(record_list)
    # execute parse function to get dataset
    # This transformation applies map_func to each element of this dataset,
    # and returns a new dataset containing the transformed elements, in the
    # same order as they appeared in the input.
    # when parse_example has only one parameter (office recommend)
    # parse_img_dataset = raw_img_dataset.map(parse_example)
    # when parse_example has more than one parameter which used to process data
    parse_img_dataset = record_dataset.map(lambda series_record:
                                            read_parse_single_example(serialized_sample = series_record,
                                                                      shortside_len=shortside_len,
                                                                      length_limitation=length_limitation,
                                                                      is_training=is_training))
    # get dataset batch
    if shuffle:
        shuffle_batch_dataset = parse_img_dataset.shuffle(buffer_size=batch_size*4).repeat(epoch).batch(batch_size=batch_size)
    else:
        shuffle_batch_dataset = parse_img_dataset.repeat(epoch).batch(batch_size=batch_size)
    # make dataset iterator
    image, filename, gtboxes_and_label, num_objects = shuffle_batch_dataset.make_one_shot_iterator().get_next()

    return image, filename, gtboxes_and_label, num_objects


def reader_tfrecord(record_file, shortside_len, length_limitation, batch_size=1, num_threads=2, epoch=5, shuffle=True,
                    is_training=False):
    """
    read and sparse TFRecord
    :param record_file:
    :return:
    """
    record_tensor = []
    # check record file format
    if os.path.isfile(record_file):
        record_tensor = [record_file]
    else:
        for filename in os.listdir(record_file):
            record_tensor.append(os.path.join(record_file, filename))
    # filename_list = tf.train.match_filenames_once(record_file)
    # create input queue
    filename_queue = tf.train.string_input_producer(string_tensor=record_tensor, num_epochs=epoch, shuffle=shuffle)
    # create reader to read TFRecord sample instant
    reader = tf.TFRecordReader()
    # read one sample instant
    _, serialized_sample = reader.read(filename_queue)
    # parse sample
    # image, label, filename = read_parse_single_example(serialized_sample, input_shape=input_shape, class_depth=class_depth)
    image, filename, gtboxes_and_label, num_objects = read_parse_single_example(serialized_sample,
                                                                                shortside_len=shortside_len,
                                                                                length_limitation=length_limitation,
                                                                                is_training=is_training)

    image, filename, gtboxes_and_label, num_objects = tf.train.batch([image, filename, gtboxes_and_label, num_objects],
                                            batch_size=batch_size,
                                            capacity=batch_size,
                                            num_threads=num_threads,
                                            dynamic_pad=True
                                            )
    # dataset = tf.data.Dataset.shuffle(buffer_size=batch_size*4)
    return image, filename, gtboxes_and_label, num_objects


def get_num_samples(record_dir):
    """
    get tfrecord numbers
    :param record_file:
    :return:
    """


    # check record file format

    record_list = glob.glob(os.path.join(record_dir, '*.record'))


    num_samples = 0
    for record_file in record_list:
        for record in tf_record_iterator(record_file):
            num_samples += 1
    return num_samples


if __name__ == "__main__":

    print('number samples: {0}'.format(get_num_samples(tfrecord_dir)))
    # create local and global variables initializer group
    # image, filename, gtboxes_and_label, num_objects = reader_tfrecord(record_file=tfrecord_dir,
    #                                                                   shortside_len=IMG_SHORT_SIDE_LEN,
    #                                                                   is_training=True)
    image_batch, filename_batch, gtboxes_and_label_batch, num_objects_batch = dataset_tfrecord(record_file=tfrecord_dir,
                                                                       shortside_len=IMG_SHORT_SIDE_LEN,
                                                                       length_limitation=IMG_MAX_LENGTH,
                                                                       is_training=True)
    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    )
    with tf.Session() as sess:
        sess.run(init_op)
        # create Coordinator to manage the life period of multiple thread
        coord = tf.train.Coordinator()
        # Starts all queue runners collected in the graph to execute input queue operation
        # the step contain two operation:filename to filename queue and sample to sample queue
        threads = tf.train.start_queue_runners(coord=coord)
        try:
            if not coord.should_stop():
                image, filename, gtboxes_and_label = sess.run([ image_batch, filename_batch, gtboxes_and_label_batch])

                plt.imshow(image[0])
                print(filename[0])
                print(gtboxes_and_label[0])
                plt.show()
        except Exception as e:
            print(e)
        finally:
            # request to stop all background threads
            coord.request_stop()
        # waiting all threads safely exit
        coord.join(threads)
        sess.close()

