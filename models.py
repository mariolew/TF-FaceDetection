""" This file defines models for face detection. Note that all the
    models are defined to be fully convolutional."""

import tensorflow as tf
from libs import utils
from libs.batch_norm import batch_norm

def fcn_12_detect(threshold, dropout=False, activation=tf.nn.relu):
    
    imgs = tf.placeholder(tf.float32, [None, 12, 12, 3])
    labels = tf.placeholder(tf.float32, [None, 1])
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    with tf.variable_scope('net_12'):
        conv1,_ = utils.conv2d(x=imgs, n_output=16, k_w=3, k_h=3, d_w=1, d_h=1, name="conv1")
        conv1 = activation(conv1)
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool1")
        ip1,W1 = utils.conv2d(x=pool1, n_output=16, k_w=6, k_h=6, d_w=1, d_h=1, padding="VALID", name="ip1")
        ip1 = activation(ip1)
        if dropout:
            ip1 = tf.nn.dropout(ip1, keep_prob)
        ip2,W2 = utils.conv2d(x=ip1, n_output=1, k_w=1, k_h=1, d_w=1, d_h=1, name="ip2")

        pred = tf.nn.sigmoid(utils.flatten(ip2))
        target = utils.flatten(labels)

        regularizer = 8e-3 * (tf.nn.l2_loss(W1)+100*tf.nn.l2_loss(W2))

        loss = tf.reduce_mean(tf.div(tf.add(-tf.reduce_sum(target * tf.log(pred + 1e-9),1), -tf.reduce_sum((1-target) * tf.log(1-pred + 1e-9),1)),2)) + regularizer
        cost = tf.reduce_mean(loss)

        thresholding_12 = tf.cast(tf.greater(pred, threshold), "float")
        recall_12 = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(thresholding_12, tf.constant([1.0])), tf.equal(target, tf.constant([1.0]))), "float")) / tf.reduce_sum(target)

        correct_prediction = tf.equal(tf.cast(tf.greater(pred, threshold), tf.int32), tf.cast(target, tf.int32))
        acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return {'imgs': imgs, 'labels': labels, 'keep_prob': keep_prob,
            'cost': cost, 'pred': pred, 'accuracy': acc, 'features': ip1,
            'recall': recall_12, 'thresholding': thresholding_12}

def fcn_24_detect(threshold, dropout=False, activation=tf.nn.relu):
    
    imgs = tf.placeholder(tf.float32, [None, 24, 24, 3])
    labels = tf.placeholder(tf.float32, [None, 1])
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    
    net_12 = fcn_12_detect(0.5, activation)
    with tf.variable_scope('net_24'):
        conv1, _ = utils.conv2d(x=imgs, n_output=64, k_w=5, k_h=5, d_w=1, d_h=1, name="conv1")
        conv1 = activation(conv1)
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool1")
        ip1, W1 = utils.conv2d(x=pool1, n_output=128, k_w=12, k_h=12, d_w=1, d_h=1, padding="VALID", name="ip1")
        ip1 = activation(ip1)
        net_12_ip1 = net_12['features']
        concat = tf.concat(3, [ip1, net_12_ip1])
        if dropout:
            concat = tf.nn.dropout(concat, keep_prob)
        ip2, W2 = utils.conv2d(x=concat, n_output=1, k_w=1, k_h=1, d_w=1, d_h=1, name="ip2")

        pred = tf.nn.sigmoid(utils.flatten(ip2))
        target = utils.flatten(labels)

        regularizer = 8e-3 * (tf.nn.l2_loss(W1)+100*tf.nn.l2_loss(W2))

        loss = tf.reduce_mean(tf.div(tf.add(-tf.reduce_sum(target * tf.log(pred + 1e-9),1), -tf.reduce_sum((1-target) * tf.log(1-pred + 1e-9),1)),2)) + regularizer
        cost = tf.reduce_mean(loss)

        thresholding_24 = tf.cast(tf.greater(pred, threshold), "float")
        recall_24 = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(thresholding_24, tf.constant([1.0])), tf.equal(target, tf.constant([1.0]))), "float")) / tf.reduce_sum(target)

        correct_prediction = tf.equal(tf.cast(tf.greater(pred, threshold), tf.int32), tf.cast(target, tf.int32))
        acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return {'imgs': imgs, 'labels': labels,
            'imgs_12': net_12['imgs'], 'labels_12': net_12['labels'],
            'keep_prob': keep_prob, 'keep_prob_12': net_12['keep_prob'],
            'cost': cost, 'pred': pred, 'accuracy': acc, 'features': concat,
            'recall': recall_24, 'thresholding': thresholding_24}


def fcn_48_detect(threshold, dropout=False, activation=tf.nn.relu):
    
    imgs = tf.placeholder(tf.float32, [None, 48, 48, 3])
    labels = tf.placeholder(tf.float32, [None, 1])
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    
    net_24 = fcn_24_detect(0.5, activation)
    with tf.variable_scope('net_48'):
        conv1, _ = utils.conv2d(x=imgs, n_output=64, k_w=5, k_h=5, d_w=1, d_h=1, name="conv1")
        conv1 = activation(conv1)
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool1")
        conv2, _ = utils.conv2d(x=pool1, n_output=64, k_w=5, k_h=5, d_w=1, d_h=1, name="conv2")
        conv2 = activation(conv2)
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool2")

        ip1, W1 = utils.conv2d(x=pool2, n_output=256, k_w=12, k_h=12, d_w=1, d_h=1, padding="VALID", name="ip1")
        ip1 = activation(ip1)
        net_24_concat = net_24['features']
        concat = tf.concat(3, [ip1, net_24_concat])
        if dropout:
            concat = tf.nn.dropout(concat, keep_prob)
        ip2, W2 = utils.conv2d(x=concat, n_output=1, k_w=1, k_h=1, d_w=1, d_h=1, name="ip2")

        pred = tf.nn.sigmoid(utils.flatten(ip2))
        target = utils.flatten(labels)

        regularizer = 8e-3 * (tf.nn.l2_loss(W1)+100*tf.nn.l2_loss(W2))

        loss = tf.reduce_mean(tf.div(tf.add(-tf.reduce_sum(target * tf.log(pred + 1e-9),1), -tf.reduce_sum((1-target) * tf.log(1-pred + 1e-9),1)),2)) + regularizer
        cost = tf.reduce_mean(loss)

        thresholding_48 = tf.cast(tf.greater(pred, threshold), "float")
        recall_48 = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(thresholding_48, tf.constant([1.0])), tf.equal(target, tf.constant([1.0]))), "float")) / tf.reduce_sum(target)

        correct_prediction = tf.equal(tf.cast(tf.greater(pred, threshold), tf.int32), tf.cast(target, tf.int32))
        acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return {'imgs': imgs, 'labels': labels,
            'imgs_24': net_24['imgs'], 'labels_24': net_24['labels'],
            'imgs_12': net_24['imgs_12'], 'labels_12': net_24['labels_12'],
            'keep_prob': keep_prob, 'keep_prob_24': net_24['keep_prob'], 'keep_prob_12': net_24['keep_prob_12'],
            'cost': cost, 'pred': pred, 'accuracy': acc,
            'recall': recall_48, 'thresholding': thresholding_48}


def fcn_12_cal(dropout=False, activation=tf.nn.relu):

    imgs = tf.placeholder(tf.float32, [None, 12, 12, 3])
    labels = tf.placeholder(tf.float32, [None])

    with tf.variable_scope('cal_12'):
        conv1,_ = utils.conv2d(x=imgs, n_output=16, k_w=3, k_h=3, d_w=1, d_h=1, name="conv1")
        conv1 = activation(conv1)
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool1")
        ip1,W1 = utils.conv2d(x=pool1, n_output=128, k_w=6, k_h=6, d_w=1, d_h=1, padding="VALID", name="ip1")
        ip1 = activation(ip1)
        if dropout:
            ip1 = tf.nn.dropout(ip1, keep_prob)
        ip2,W2 = utils.conv2d(x=ip1, n_output=45, k_w=1, k_h=1, d_w=1, d_h=1, name="ip2")

        pred = utils.flatten(ip2)
        # target = utils.flatten(labels)
        # label_shape = labels.get_shape().as_list()
        # target = tf.reshape(labels,[label_shape[0]])
        target = labels

        cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(pred, tf.cast(target, tf.int64)))
        regularizer = 8e-3 * (tf.nn.l2_loss(W1)+100*tf.nn.l2_loss(W2))

        loss = cross_entropy + regularizer

        correct_prediction = tf.equal(tf.argmax(pred,1), tf.cast(target,tf.int64))
        acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return {'cost': loss, 'pred': pred, 'accuracy': acc, 'target': target, 'imgs': imgs, 'labels': labels}


def fcn_24_cal(dropout=False, activation=tf.nn.relu):

    imgs = tf.placeholder(tf.float32, [None, 24, 24, 3])
    labels = tf.placeholder(tf.float32, [None])

    with tf.variable_scope('cal_24'):
        conv1,_ = utils.conv2d(x=imgs, n_output=32, k_w=5, k_h=5, d_w=1, d_h=1, name="conv1")
        conv1 = activation(conv1)
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool1")
        ip1,W1 = utils.conv2d(x=pool1, n_output=64, k_w=12, k_h=12, d_w=1, d_h=1, padding="VALID", name="ip1")
        ip1 = activation(ip1)
        if dropout:
            ip1 = tf.nn.dropout(ip1, keep_prob)
        ip2,W2 = utils.conv2d(x=ip1, n_output=45, k_w=1, k_h=1, d_w=1, d_h=1, name="ip2")

        pred = utils.flatten(ip2)
        # target = utils.flatten(labels)
        # label_shape = labels.get_shape().as_list()
        # target = tf.reshape(labels,[label_shape[0]])
        target = labels

        cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(pred, tf.cast(target, tf.int64)))
        regularizer = 8e-3 * (tf.nn.l2_loss(W1)+100*tf.nn.l2_loss(W2))

        loss = cross_entropy + regularizer

        correct_prediction = tf.equal(tf.argmax(pred,1), tf.cast(target,tf.int64))
        acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return {'cost': loss, 'pred': pred, 'accuracy': acc, 'target': target, 'imgs': imgs, 'labels': labels}


def fcn_48_cal(dropout=False, activation=tf.nn.relu):

    imgs = tf.placeholder(tf.float32, [None, 48, 48, 3])
    labels = tf.placeholder(tf.float32, [None])

    with tf.variable_scope('cal_48'):
        conv1,_ = utils.conv2d(x=imgs, n_output=64, k_w=5, k_h=5, d_w=1, d_h=1, name="conv1")
        conv1 = activation(conv1)
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool1")
        conv2,_ = utils.conv2d(x=pool1, n_output=64, k_w=5, k_h=5, d_w=1, d_h=1, name="conv2")
        ip1,W1 = utils.conv2d(x=conv2, n_output=256, k_w=24, k_h=24, d_w=1, d_h=1, padding="VALID", name="ip1")
        ip1 = activation(ip1)
        if dropout:
            ip1 = tf.nn.dropout(ip1, keep_prob)
        ip2,W2 = utils.conv2d(x=ip1, n_output=45, k_w=1, k_h=1, d_w=1, d_h=1, name="ip2")

        pred = utils.flatten(ip2)
        # target = utils.flatten(labels)
        # label_shape = labels.get_shape().as_list()
        # target = tf.reshape(labels,[label_shape[0]])
        target = labels

        cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(pred, tf.cast(target, tf.int64)))
        regularizer = 8e-3 * (tf.nn.l2_loss(W1)+100*tf.nn.l2_loss(W2))

        loss = cross_entropy + regularizer

        correct_prediction = tf.equal(tf.argmax(pred,1), tf.cast(target,tf.int64))
        acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return {'cost': loss, 'pred': pred, 'accuracy': acc, 'target': target, 'imgs': imgs, 'labels': labels}