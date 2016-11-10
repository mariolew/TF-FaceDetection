""" This file defines models for face detection. Note that all the
    models are defined to be fully convolutional."""

import tensorflow as tf
from libs import utils
from libs.batch_norm import batch_norm

def fcn_12_detect(imgs, labels, threshold, dropout=False, activation=tf.nn.relu):
    
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    conv1 = utils.conv2d(x=imgs, n_output=16, k_w=3, k_h=3, d_w=1, d_h=1, name="conv1")
    conv1 = activation(conv1)
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool1")
    ip1 = utils.conv2d(x=pool1, n_output=16, k_w=6, k_h=6, d_w=1, d_h=1, name="ip1")
    ip1 = activation(ip1)
    if dropout:
        ip1 = tf.nn.dropout(ip1, keep_prob)
    ip2 = utils.conv2d(x=ip1, n_output=1, k_w=1, k_h=1, d_w=1, d_h=1, padding="VALID", name="ip2")

    pred = tf.nn.sigmoid(utils.flatten(ip2))
    target = utils.flatten(labels)

    loss = -tf.reduce_sum( (  (target*tf.log(pred + 1e-9)) + ((1-target) * tf.log(1 - pred + 1e-9))), 1  , name='xentropy' )
    cost = tf.reduce_mean(loss)

    correct_prediction = tf.equal(tf.cast(tf.greater(pred, threshold), tf.int32), tf.cast(target, tf.int32))
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return {'cost': cost, 'pred': pred, 'accuracy': acc, 'keep_prob': keep_prob, 'features': ip1}

def fcn_24_detect(imgs, labels, threshold, dropout=False, activation=tf.nn.relu):
    
    net_12 = fcn_12_detect(imgs, labels, dropout, activation)
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    conv1 = utils.conv2d(x=imgs, n_output=64, k_w=5, k_h=5, d_w=1, d_h=1, name="conv1")
    conv1 = activation(conv1)
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool1")
    ip1 = utils.conv2d(x=pool1, n_output=128, k_w=12, k_h=12, d_w=1, d_h=1, name="ip1")
    ip1 = activation(ip1)
    net_12_ip1 = net_12['features']
    concat = tf.concat(3, [ip1, net_12_ip1])
    if dropout:
        concat = tf.nn.dropout(concat, keep_prob)
    ip2 = utils.conv2d(x=concat, n_output=1, k_w=1, k_h=1, d_w=1, d_h=1, padding="VALID", name="ip2")

    pred = tf.nn.sigmoid(utils.flatten(ip2))
    target = utils.flatten(labels)

    loss = -tf.reduce_sum( (  (target*tf.log(pred + 1e-9)) + ((1-target) * tf.log(1 - pred + 1e-9))), 1  , name='xentropy' )
    cost = tf.reduce_mean(loss)

    correct_prediction = tf.equal(tf.cast(tf.greater(pred, threshold), tf.int32), tf.cast(target, tf.int32))
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return {'cost': cost, 'pred': pred, 'accuracy': acc, 'keep_prob': keep_prob, 'features': concat}


def fcn_48_detect(imgs, labels, threshold, dropout=False, activation=tf.nn.relu):
    
    net_24 = fcn_24_detect(imgs, labels, dropout, activation)
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    conv1 = utils.conv2d(x=imgs, n_output=64, k_w=5, k_h=5, d_w=1, d_h=1, name="conv1")
    conv1 = activation(conv1)
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool1")
    norm1 = batch_norm(pool1)
    conv2 = utils.conv2d(x=norm1, n_output=64, k_w=5, k_h=5, d_w=1, d_h=1, name="conv2")
    conv2 = activation(batch_norm(conv2))
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool2")

    ip1 = utils.conv2d(x=pool2, n_output=256, k_w=12, k_h=12, d_w=1, d_h=1, name="ip1")
    ip1 = activation(ip1)
    net_24_ip1 = net_24['features']
    concat = tf.concat(3, [ip1, net_24_ip1])
    if dropout:
        concat = tf.nn.dropout(concat, keep_prob)
    ip2 = utils.conv2d(x=concat, n_output=1, k_w=1, k_h=1, d_w=1, d_h=1, padding="VALID", name="ip2")

    pred = tf.nn.sigmoid(utils.flatten(ip2))
    target = utils.flatten(labels)

    loss = -tf.reduce_sum( (  (target*tf.log(pred + 1e-9)) + ((1-target) * tf.log(1 - pred + 1e-9))), 1  , name='xentropy' )
    cost = tf.reduce_mean(loss)

    correct_prediction = tf.equal(tf.cast(tf.greater(pred, threshold), tf.int32), tf.cast(target, tf.int32))
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return {'cost': cost, 'pred': pred, 'accuracy': acc, 'keep_prob': keep_prob, 'features': concat}




