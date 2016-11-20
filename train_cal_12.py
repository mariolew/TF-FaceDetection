import tensorflow as tf
from image_inputs import inputs_without_crop
from models import fcn_12_cal


def train():

    lists = ['cal_12_train.txt']
    image_train, label_train = inputs_without_crop(lists, [12, 12, 3], 64)

    net_output = fcn_12_cal(image_train, label_train)

    global_step = tf.Variable(0, tf.int32)
    starter_learning_rate = 0.001
    learning_rate = tf.train.exponential_decay(
        learning_rate=starter_learning_rate,
        global_step=global_step,
        decay_steps=1000,
        decay_rate=0.99,
        staircase=True,
        name=None)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(net_output['cost'], global_step=global_step)


    sess = tf.Session()
    saver = tf.train.Saver(tf.trainable_variables())
    # import pdb; pdb.set_trace()
    sess.run(tf.initialize_all_variables())
    coord = tf.train.Coordinator()

    # tf.get_default_graph().finalize()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    # saver.restore(sess, 'model/model_cal_12-23395')
    try:
        for i in range(400000):

            imgs, lbls = sess.run([image_train, label_train])
            lbls = lbls.reshape((lbls.shape[0]))

            if i%10000==0 and i!=0:

                saver.save(sess, 'model/model_cal_12', global_step=global_step, write_meta_graph=False)
                
            if i%1==0:

                cost, accuracy, lr = sess.run(
                    [net_output['cost'], net_output['accuracy'], learning_rate],
                    feed_dict={net_output['imgs']: imgs, net_output['labels']: lbls})

                print("Step %d, cost: %f, acc: %f, lr: %f"%(i, cost, accuracy, lr))
                # print("target: ", target)
                # print("pred: ", pred)

            # train
            sess.run(train_step, feed_dict={net_output['imgs']: imgs, net_output['labels']: lbls})

        coord.request_stop()

    except Exception as e:

        coord.request_stop(e)

    finally:

        print('Done training.')
        saver.save(sess, 'model/model_cal_12', global_step=global_step, write_meta_graph=False)

    coord.request_stop()

    coord.join(threads)
    sess.close()



if __name__ == '__main__':

    train()
