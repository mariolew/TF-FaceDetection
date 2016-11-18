import tensorflow as tf
from models import fcn_12_detect
from image_inputs import *

def test(model_path, lists, batch_size, test_interval):

    images, labels = inputs_for_test(lists, [12, 12, 3], batch_size)
    is_train = tf.placeholder(tf.bool)

    net_output = fcn_12_detect(0.16)

    saver = tf.train.Saver()

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    saver.restore(sess, model_path)

    sum_acc=0
    sum_recall=0
    for i in range(test_interval):
        imgs, lbls = sess.run([images, labels])
        acc, recall = sess.run([net_output['accuracy'], net_output['recall']], feed_dict={net_output['imgs']: imgs, net_output['labels']: lbls, is_train: False})
        print('iter %d, acc: %f, recall: %f'%(i, acc, recall))
        sum_acc += acc
        sum_recall += recall
    print('mean_acc: %f, mean_recall: %f'%(sum_acc/test_interval, sum_recall/test_interval))

    coord.request_stop()
    coord.join(threads)

    sess.close()


if __name__ == '__main__':

    test(model_path='model/model_net_12-400000', lists=['net_12_validation.txt'], batch_size=100, test_interval=10)