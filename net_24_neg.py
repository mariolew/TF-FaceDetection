import tensorflow as tf
import numpy as np
from models import fcn_12_detect
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
    if w_re<=window_size+stride or h_re<=window_size+stride or w_re>=10*window_size or h_re>=10*window_size:
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

    net_output = fcn_12_detect(0.16)

    saver = tf.train.Saver()

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    saver.restore(sess, model_path)

    neg_list = []
    for idx, image_path in enumerate(image_list):
        print('iter %d'%(idx))
        batch_window1 = slide_window(image_path, 40, 12, 4)
        batch_window2 = slide_window(image_path, 40, 24, 8)
        if batch_window1 is None:
            continue

        thresholding = sess.run(net_output['thresholding'], feed_dict={net_output['imgs']: batch_window1, net_output['keep_prob']: 1.0})
        sn_wrong = np.where(thresholding==1.0)[0]
        neg_list.extend(batch_window2[sn_wrong])
        print(len(neg_list))

    f = h5py.File('net_24_neg_for_train.hdf5', 'w')
    dataset_for_train = f.create_dataset('imgs', data=neg_list[0:-800])
    f = h5py.File('net_24_neg_for_eval.hdf5', 'w')
    dataset_for_eval = f.create_dataset('imgs', data=neg_list[-800:])

    coord.request_stop()
    coord.join(threads)

    sess.close()


    
if __name__ == '__main__':

    image_list = glob.glob('imagenet_selected/*.jpg')
    eval_save('model/model_net_12-400000', image_list)