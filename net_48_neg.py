import tensorflow as tf
import numpy as np
from models import fcn_24_detect
from PIL import Image
import glob
import h5py


def image_preprocess(img):

    m = img.mean()
    s = img.std()
    min_s = 1.0/(np.sqrt(img.shape[0]*img.shape[1]*img.shape[2]))
    std = max(min_s, s)

    return (img-m)/std


def slide_window(image_path, F, window_size, stride):
    
    window_list = []

    img = Image.open(image_path)
    w, h = img.size
    w_re = int(float(w)*window_size/F)
    h_re = int(float(h)*window_size/F)
    if w_re<=window_size+stride or h_re<=window_size+stride or w_re>=20*window_size or h_re>=20*window_size:
        return None
    img = img.resize((w_re, h_re))
    img = np.array(img)
    if len(img.shape)!=3:
        return None

    for i in range(int((w_re-window_size)/stride)):
        for j in range(int((h_re-window_size)/stride)):
            patch = img[j*stride:j*stride+window_size, i*stride:i*stride+window_size, :]
            patch = image_preprocess(patch)

            window_list.append(patch)

    return np.asarray(window_list)


def eval_save(model_path, image_list):

    net_output = fcn_24_detect(0.025)

    saver = tf.train.Saver()

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    saver.restore(sess, model_path)

    f1 = h5py.File('net_48_neg_for_train.hdf5', 'w')
    dataset_for_train = f1.create_dataset('imgs', (1, 48, 48, 3), maxshape=(None, 48, 48, 3))
    neg_num = 0
    for idx, image_path in enumerate(image_list):
        print('iter %d'%(idx))
        batch_window1 = slide_window(image_path, 40, 24, 8)
        batch_window2 = slide_window(image_path, 40, 48, 16)
        
        if batch_window1 is None:
            continue
        batch_window1_12 = []
        for window in batch_window1:
            im = window.copy()
            im.resize((12, 12, 3))
            batch_window1_12.append(im)
        batch_window1_12 = np.array(batch_window1_12)
        # print(batch_window1_12.shape)

        feed_dict = {
            net_output['imgs']: batch_window1,
            net_output['imgs_12']: batch_window1_12,
            net_output['keep_prob']: 1.0,
            net_output['keep_prob_12']: 1.0
        }
        thresholding = sess.run(net_output['thresholding'], feed_dict=feed_dict)
        sn_wrong = np.where(thresholding==1.0)[0]
        neg_list = batch_window2[sn_wrong]

        neg_num += len(sn_wrong)
        dataset_for_train.resize((neg_num, 48, 48, 3))
        dataset_for_train[neg_num-len(sn_wrong):neg_num] = neg_list
        print(dataset_for_train.len())
        print(neg_num)
    
    f2 = h5py.File('net_48_neg_for_eval.hdf5', 'w')
    dataset_for_eval = f2.create_dataset('imgs', (1000, 48, 48, 3))
    dataset_for_eval[:] = dataset_for_train[-1000:]
    dataset_for_train.resize((neg_num-1000, 48, 48, 3))

    coord.request_stop()
    coord.join(threads)

    sess.close()



if __name__ == '__main__':

    image_list = glob.glob('imagenet_selected/*.jpg')
    eval_save('model/model_net_24-210000', image_list)