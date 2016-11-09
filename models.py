""" This file defines models for face detection. Note that all the
    models are defined to be fully convolutional."""

import tensorflow as tf
import utils

def fcn_12_detect(imgs, labels, dropout=False, activation=tf.nn.relu):
    
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    conv1 = utils.conv2d(x=imgs, n_output=16, k_w=3, k_h=3, d_w=1, d_h=1, name="conv1")
    conv1 = activation(conv1)
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool1")
    ip1 = utils.conv2d(x=pool1, n_output=16, k_w=6, k_h=6, d_w=1, d_h=1, name="ip1")
    ip1 = activation(ip1)
    if dropout:
        ip1 = tf.nn.dropout(ip1, keep_prob)
    ip2 = utils.conv2d(x=ip1, n_output=2, k_w=1, k_h=1, d_w=1, d_h=1, name="ip2")

    pred = utils.flatten(ip2)
    target = utils.flatten(labels)

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(pred, target)
    cost = tf.reduce_mean(loss)

    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.cast(target, tf.int32))
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return {'cost': cost, 'pred': pred, 'accuracy': acc, 'keep_prob': keep_prob}




