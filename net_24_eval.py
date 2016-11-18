import tensorflow as tf
import numpy as np
from image_inputs import inputs_for_test
from models import fcn_24_detect
import h5py
import random

def test():

    model_path = 'model/model_net_24-210000'

    f = h5py.File('net_24_neg_for_eval.hdf5','r')
    imgs_neg = f['imgs'][:]
    neg_len = len(imgs_neg)

    lists = ['net_pos_for_eval.txt']
    images, labels = inputs_for_test(lists, [24, 24, 3], 20)

    net_output = fcn_24_detect(0.025)

    is_train = tf.placeholder(tf.bool)

    saver = tf.train.Saver(tf.trainable_variables())

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    saver.restore(sess, model_path)

    sum_acc=0
    sum_recall=0
    for i in range(10):
        imgs_pos = sess.run(images)
        imgs = np.vstack([imgs_pos,imgs_neg[i*80:i*80+80]])
        imgs_12 = []
        for img in imgs:
            im = img.copy()
            im.resize((12, 12, 3))
            imgs_12.append(im)
        imgs_12 = np.array(imgs_12)
        # print(imgs_12.shape)

        labels = np.vstack([np.ones((20,1)), np.zeros((80,1))])

        sn_shf = np.array(random.sample(range(100), 100))
        imgs = imgs[sn_shf]
        imgs_12 = imgs_12[sn_shf]
        labels = labels[sn_shf]

        feed_dict = {
            net_output['imgs']: imgs,
            net_output['labels']: labels,
            net_output['imgs_12']: imgs_12,
            net_output['labels_12']: labels,
            net_output['keep_prob']: 1.0,
            net_output['keep_prob_12']: 1.0
        }
        acc, recall = sess.run([net_output['accuracy'], net_output['recall']], feed_dict=feed_dict)
        print('iter %d, acc: %f, recall: %f'%(i, acc, recall))
        sum_acc += acc
        sum_recall += recall
    print('mean_acc: %f, mean_recall: %f'%(sum_acc/10, sum_recall/10))

    coord.request_stop()
    coord.join(threads)

    sess.close()

if __name__ == '__main__':

    test()