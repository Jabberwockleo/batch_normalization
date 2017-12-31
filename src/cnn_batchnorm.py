#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) 2017 Wan Li. All Rights Reserved
#
########################################################################

"""
File: cnn.py
Author: Wan Li
Date: 2017/11/27 10:41:01
"""
import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.examples.tutorials.mnist as mnist
import mock_util

# environment config
GRAPH_MODE = "train" # train/predict
MODEL_PATH = "model.cnn_batchnorm.ckpt"
LOGS_PATH = "logs/cnn_batchnorm/"

# specific hyper parameters

# graph config
HIDDEN_UNIT_NUM = 10
HIDDEN_LAYERS_NUM = 2
FEATURE_DIM = 1
OUT_DIM = 1

# general hyper parameters
LEARNING_RATE = 0.01
TRAINING_EPOCH_NUM = 1000
BATCH_SIZE = 10

# debug flags
DEBUG_LOG = False

# auxiliary storage
accumulative_var_name_list = []
accumulative_var_list = []

# dataset parameters
DATA_DIR = "../data/mnist"
IMG_WIDTH = 28
IMG_HEIGHT = 28
PIXEL_NUM = 28 * 28
CLASS_NUM = 10

# mock config
param_x = -5
param_y = 5
param_bias = 50 # works for higer bias without preprocessing

def clear_accumulative_ref():
    """Clear the reference list

    The list is for tf.Saver to restore ExponentialMeanAverage
    values to predict graph
    """
    accumulative_var_name_list[:] = []
    accumulative_var_list[:] = []

def build_graph(mode="train"):
    """Build compute graph

    Train graph and predict graph is different
    for batch normalization

    Args:
        mode: train/predict
    Returns:
        X, Y, y, ema, train_op, cost, accuracy
    """
    with tf.name_scope("input"):
        X = tf.placeholder(tf.float32, [None, PIXEL_NUM]) # flat image 0/1 pixel
        Y = tf.placeholder(tf.float32, [None, CLASS_NUM]) # one hot

    with tf.name_scope("cnn"):
        # remember mean/var in each iteration and compute
        # aggregates through time
        ema = tf.train.ExponentialMovingAverage(decay=0.99)

        def add_batch_norm(layer):
            """Add batch normalization

            Args:
                layer: building layer
            Returns:
                built graph
            """
            scale = tf.Variable(tf.ones([HIDDEN_UNIT_NUM]), name="scale")
            shift = tf.Variable(tf.zeros([HIDDEN_UNIT_NUM]), name="shift")
            mean = tf.Variable(tf.zeros([HIDDEN_UNIT_NUM]), name="mean")
            var = tf.Variable(tf.zeros([HIDDEN_UNIT_NUM]), name="variance")
            accumulative_var_list.append(mean)
            accumulative_var_list.append(var)

            def accumulate(layer):
                """Accumulate mean and var

                The accumulative average will be used in predict mode
                """
                # ensure mean and var are updated
                ema_apply_op = ema.apply([mean, var])
                accumulative_var_name_list.append(ema.average_name(mean))
                accumulative_var_name_list.append(ema.average_name(var))
                with tf.control_dependencies([ema_apply_op]):
                    # use an identity op wrap to ensure that ema_apply_op is executed
                    return tf.identity(mean), tf.identity(var)

            if mode == "train":
                mean_, var_ = tf.nn.moments(layer, axes=[0])
                # assign value instead of replacing variable referance
                # must force op dependencies
                with tf.control_dependencies([tf.assign(mean, mean_), tf.assign(var, var_)]):
                    mean, var = accumulate(layer)
            elif mode == "predict":
                # do nothing, mean and var is assigned by Saver
                pass

            # batch norm
            layer = (layer - mean) / tf.sqrt(var + 0.001)
            layer = layer * scale + shift
            return layer

        # default format "NHWC", the data is stored in the order of: [batch, height, width, channels].
        layer = tf.reshape(X, shape=[-1, IMG_WIDTH, IMG_HEIGHT, 1])
        # first convolution, output shape is [-1, 28, 28, 32]
        layer = tf.nn.conv2d(input=layer,
                # 5x5 filter of 1 channel in 32 channels out
                filter=tf.Variable(tf.random_normal([5, 5, 1, 32])),
                strides=[1, 1, 1, 1],
                padding="SAME")
        layer = tf.nn.bias_add(layer, tf.Variable(tf.random_normal([32])))
        # maxpooling with 2x2, strides 2x2, output [-1, 14, 14, 32]
        layer = tf.nn.max_pool(layer,
                ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1],
                padding="SAME")
        # second convolution, output shape is [-1, 14, 14, 64]
        layer = tf.nn.conv2d(input=layer,
                filter=tf.Variable(tf.random_normal([5, 5, 32, 64])),
                strides=[1, 1, 1, 1],
                padding="SAME")
        layer = tf.nn.bias_add(layer, tf.Variable(tf.random_normal([64])))
        # maxpooling with 2x2, strides 2x2, output [-1, 7, 7, 64]
        layer = tf.nn.max_pool(layer,
                ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1],
                padding="SAME")

        # flatten channels
        layer = tf.reshape(layer, [-1, 7 * 7 * 64])
        # transform image to 1024 features
        layer = tf.add(tf.matmul(layer,
            tf.Variable(tf.random_normal([7 * 7 * 64, 1024]))),
            tf.Variable([tf.random_normal([1024])]))
        # apply dropout for regularization
        layer = tf.nn.dropout(layer, D)
        # finally each image outputs
        layer = tf.add(tf.matmul(layer,
            tf.Variable(tf.random_normal([1024, CLASS_NUM]))),
            tf.Variable(tf.random_normal([CLASS_NUM])))

    with tf.name_scope("op"):
        cost = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    logits=layer, labels=Y))
        train_op = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)
        correct_flag_tensor = tf.equal(tf.argmax(layer, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_flag_tensor, tf.float32))
    return X, Y, y, ema, train_op, cost, accuracy

if __name__ == "__main__":
    GRAPH_MODE = sys.argv[1]
    mnist_data = mnist.input_data.read_data_sets(DATA_DIR, one_hot=True)
    clear_accumulative_ref()
    if GRAPH_MODE == "train":
        with tf.Session() as sess:
            X, Y, y, ema, train_op, cost, accuracy = build_graph(GRAPH_MODE)
            sess.run(tf.global_variables_initializer())
            summary_writer = tf.summary.FileWriter(LOGS_PATH, graph=tf.get_default_graph())
            for epoch in xrange(TRAINING_EPOCH_NUM):
                # the bias term severely affects training with no norm
                x_data, y_data = mnist_data.train.next_batch(BATCH_SIZE)
                sess.run([train_op],
                         feed_dict={X:x_data, Y:y_data})
                if epoch % 100 == 0:
                    cost_v, accuracy_v = sess.run([cost, accuracy],
                         feed_dict={X:x_data, Y:y_data})
                    print "== epoch [%d], cost [%f], accuracy [%f]" % (epoch, cost_v, accuracy)
            # accumulative variables
            print "accumulative variables:"
            all_vars = ema.variables_to_restore(moving_avg_variables=accumulative_var_list)
            all_vars_inverse = dict((v, k) for k, v in all_vars.iteritems())
            accumulative_var_dict = dict((all_vars_inverse[v], v) for v in accumulative_var_list)
            print accumulative_var_dict
            # save model
            saver = tf.train.Saver()
            save_path = saver.save(sess, MODEL_PATH)
            print "Model saved in file: %s" % save_path


    if GRAPH_MODE == "predict":
        with tf.Session() as sess:
            X, Y, y, ema, train_op, cost = build_graph(GRAPH_MODE)

            sess.run(tf.global_variables_initializer())

            # restore model
            restorer = tf.train.Saver()
            restorer.restore(sess, MODEL_PATH)
            # restore accumulative variables to save
            print "accumulative variables:"
            all_vars = ema.variables_to_restore(moving_avg_variables=accumulative_var_list)
            all_vars_inverse = dict((v, k) for k, v in all_vars.iteritems())
            accumulative_var_dict = dict((all_vars_inverse[v], v) for v in accumulative_var_list)
            print accumulative_var_dict
            # save model
            restorer = tf.train.Saver(accumulative_var_dict)
            restorer.restore(sess, MODEL_PATH)

            summary_writer = tf.summary.FileWriter(LOGS_PATH, graph=tf.get_default_graph())
            print "plotting..."
            x_data, y_data = mock_util.x2makeup(
                param_x, param_y, param_bias, 1000)
            y_v = sess.run([y],
                feed_dict={X:x_data, Y:y_data})
            plt.scatter(x_data, y_data, c='b', label='truth')
            plt.scatter(x_data, y_v, c='r', label='predict')
            plt.legend(loc='upper left')
            plt.show()

    print("Run the command line:\n" \
          "--> tensorboard --logdir=" + LOGS_PATH + "" \
          "\nThen open http://0.0.0.0:6006/ into your web browser")
