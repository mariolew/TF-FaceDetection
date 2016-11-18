import tensorflow as tf
import numpy as np
from image_inputs import inputs
from models import fcn_24_detect
import h5py

def train():

    f = h5py.File('net_24_neg_for_train.hdf5','r')
    imgs_neg = f['imgs'][:]
    neg_len = len(imgs_neg)

    lists = ['net_pos_list.txt','net_pos_flip_list.txt']
    image_train, label_train = inputs(lists, [24, 24, 3], 12)

    net_output = fcn_24_detect(0.2)

    global_step = tf.Variable(0, tf.int32)
    starter_learning_rate = 0.005
    learning_rate = tf.train.exponential_decay(
        learning_rate=starter_learning_rate,
        global_step=global_step,
        decay_steps=1000,
        decay_rate=0.99,
        staircase=True,
        name=None)
    opt_vars_24 = [v for v in tf.trainable_variables() if v.name.startswith('net_24')]
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(net_output['cost'], var_list=opt_vars_24, global_step=global_step)


    sess = tf.Session()
    opt_vars_12 = [v for v in tf.trainable_variables() if v.name.startswith('net_12')]
    saver_12 = tf.train.Saver(opt_vars_12)
    saver_24 = tf.train.Saver(tf.trainable_variables())
    # import pdb; pdb.set_trace()
    sess.run(tf.initialize_all_variables())
    coord = tf.train.Coordinator()

    # tf.get_default_graph().finalize()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    saver_12.restore(sess, 'model/model_net_12-400000')
    try:
        for i in range(400000):

            sn_rand = np.random.random_integers(0, neg_len-1, 50)
            imgs_pos = sess.run(image_train)
            imgs = np.vstack([imgs_pos,imgs_neg[sn_rand]])
            imgs_12 = []
            for img in imgs:
                im = img.copy()
                im.resize((12, 12, 3))
                imgs_12.append(im)
            imgs_12 = np.array(imgs_12)

            labels = np.vstack([np.ones((12,1)), np.zeros((50,1))])

            sn_shf = np.array(random.sample(range(62), 62))
            imgs = imgs[sn_shf]
            imgs_12 = imgs_12[sn_shf]
            labels = labels[sn_shf]

            if i%10000==0 and i!=0:

                saver_24.save(sess, 'model/model_net_24', global_step=global_step, write_meta_graph=False)
                
            if i%1==0:

                feed_dict = {
                    net_output['imgs']: imgs,
                    net_output['labels']: labels,
                    net_output['imgs_12']: imgs_12,
                    net_output['labels_12']: labels,
                    net_output['keep_prob']: 1.0,
                    net_output['keep_prob_12']: 1.0
                }

                cost, accuracy, recall, lr = sess.run(
                    [net_output['cost'], net_output['accuracy'], net_output['recall'], learning_rate],
                    feed_dict=feed_dict)

                print("Step %d, cost: %f, acc: %f, recall: %f, lr: %f"%(i, cost, accuracy, recall, lr))
                # print("target: ", target)
                # print("pred: ", pred)

            # train
            feed_dict = {
                net_output['imgs']: imgs,
                net_output['labels']: labels,
                net_output['imgs_12']: imgs_12,
                net_output['labels_12']: labels,
                net_output['keep_prob']: 0.8,
                net_output['keep_prob_12']: 0.8
            }
            sess.run(train_step, feed_dict=feed_dict)

        coord.request_stop()

    except Exception as e:

        coord.request_stop(e)

    finally:

        print('Done training.')
        saver_24.save(sess, 'model/model_net_24', global_step=global_step, write_meta_graph=False)

    coord.request_stop()

    coord.join(threads)
    sess.close()



if __name__ == '__main__':

    train()