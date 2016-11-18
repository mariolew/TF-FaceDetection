import tensorflow as tf
import numpy as np
from image_inputs import inputs_for_test
from models import fcn_48_detect
import h5py
import random

def test():

    model_path = 'model/model_net_48-7853'

    f = h5py.File('net_48_neg_for_eval.hdf5','r')
    neg_list = f['imgs']
    neg_len = neg_list.len()

    lists = ['net_pos_for_eval.txt']
    # lists = ['net_pos_list.txt','net_pos_flip_list.txt']
    images, labels = inputs_for_test(lists, [48, 48, 3], 20)

    net_output = fcn_48_detect(0.025)

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
        imgs = np.vstack([imgs_pos, neg_list[i*80:i*80+80]])
        imgs_24 = []
        imgs_12 = []
        for img in imgs:
            im_24 = img.copy()
            im_12 = img.copy()
            im_24.resize((24, 24, 3))
            im_12.resize((12, 12, 3))
            imgs_24.append(im_24)
            imgs_12.append(im_12)
        imgs_24 = np.array(imgs_24)
        imgs_12 = np.array(imgs_12)

        labels = np.vstack([np.ones((20,1)), np.zeros((80,1))])

        feed_dict = {
            net_output['imgs']: imgs,
            net_output['labels']: labels,
            net_output['imgs_24']: imgs_24,
            net_output['labels_24']: labels,
            net_output['imgs_12']: imgs_12,
            net_output['labels_12']: labels,
            net_output['keep_prob']: 1.0,
            net_output['keep_prob_24']: 1.0,
            net_output['keep_prob_12']: 1.0
        }
        acc, recall = sess.run([net_output['accuracy'], net_output['recall']], feed_dict=feed_dict)
        print('iter %d, acc: %f, recall: %f'%(i, acc, recall))
        # print(pred)
        sum_acc += acc
        sum_recall += recall
    print('mean_acc: %f, mean_recall: %f'%(sum_acc/10, sum_recall/10))

    coord.request_stop()
    coord.join(threads)

    sess.close()

if __name__ == '__main__':

    test()