import tensorflow as tf
import numpy as np
import sys
from models import *
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFile
from skimage.transform import pyramid_gaussian
from skimage.transform import resize
from matplotlib import pyplot
ImageFile.LOAD_TRUNCATED_IMAGES = True


def NMS_fast(detected_list, overlap_rate=0.5, included_rate=0.9):

    if len(detected_list) == 0:
        return []
    
    pick = []
    lt_x = np.array([detected_list[i][0] for i in range(len(detected_list))],np.float32)
    lt_y = np.array([detected_list[i][1] for i in range(len(detected_list))],np.float32)
    rb_x = np.array([detected_list[i][2] for i in range(len(detected_list))],np.float32)
    rb_y = np.array([detected_list[i][3] for i in range(len(detected_list))],np.float32)

    area = (rb_x-lt_x)*(rb_y-lt_y)
    idxs = np.array(range(len(detected_list)))

    while len(idxs) > 0:
        i = idxs[0]
        pick.append(i)

        xx1 = np.maximum(lt_x[i],lt_x[idxs[1:]])
        yy1 = np.maximum(lt_y[i],lt_y[idxs[1:]])
        xx2 = np.minimum(rb_x[i],rb_x[idxs[1:]])
        yy2 = np.minimum(rb_y[i],rb_y[idxs[1:]])

        w = np.maximum(xx2-xx1,0)
        h = np.maximum(yy2-yy1,0)

        overlap = (w*h)/(area[idxs[1:]] + area[i] - w*h)
        included = w*h / (0.9 * area[idxs[1:]])

        idxs = np.delete(idxs, np.concatenate(([0],np.where(((overlap >= overlap_rate) & (overlap < 1)) |(included > included_rate)  )[0]+1)))
    
    return pick


def image_preprocess(img):

    m = img.mean()
    s = img.std()
    min_s = 1.0/(np.sqrt(img.shape[0]*img.shape[1]*img.shape[2]))
    std = max(min_s, s)

    return (img-m)/std


def slide_window(img, F, window_size, stride):
    
    window_list = []

    w = img.shape[1]
    h = img.shape[0]
    w_re = int(float(w)*window_size/F)
    h_re = int(float(h)*window_size/F)
    if w_re<=window_size+stride or h_re<=window_size+stride:
        return None
    img = resize(img, (h_re, w_re, 3))
    img = image_preprocess(img)
    if len(img.shape)!=3:
        return None

    for i in range(int((w_re-window_size)/stride)):
        for j in range(int((h_re-window_size)/stride)):
            box = [j*stride, i*stride, j*stride+window_size, i*stride+window_size]

            window_list.append(box)

    return img, np.asarray(window_list)


if __name__ == '__main__':

    dic = []
    s_set = (0.83, 0.91, 1.0, 1.10, 1.21)
    x_set = (-0.17, 0, 0.17)
    y_set = (-0.17, 0, 0.17)
    for x in x_set:
        for y in y_set:
            for s in s_set:
                dic.append([s, x, y])
    dic = np.array(dic)

    image = Image.open('test6.jpg')
    w, h = image.size

    cal_12 = fcn_12_cal()
    cal_24 = fcn_24_cal()
    net_48 = fcn_48_detect(0.01)
    cal_48 = fcn_48_cal()

    net_24 = net_48['net_24']
    net_12 = net_24['net_12']

    net_12_vars = [v for v in tf.trainable_variables() if v.name.startswith('net_12')]
    saver_net_12 = tf.train.Saver(net_12_vars)
    cal_12_vars = [v for v in tf.trainable_variables() if v.name.startswith('cal_12')]
    saver_cal_12 = tf.train.Saver(cal_12_vars)
    net_24_vars = [v for v in tf.trainable_variables() if v.name.startswith('net_24')]
    saver_net_24 = tf.train.Saver(net_24_vars)
    cal_24_vars = [v for v in tf.trainable_variables() if v.name.startswith('cal_24')]
    saver_cal_24 = tf.train.Saver(cal_24_vars)
    net_48_vars = [v for v in tf.trainable_variables() if v.name.startswith('net_48')]
    saver_net_48 = tf.train.Saver(net_48_vars)
    cal_48_vars = [v for v in tf.trainable_variables() if v.name.startswith('cal_48')]
    saver_cal_48 = tf.train.Saver(cal_48_vars)

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    saver_net_12.restore(sess, 'model/model_net_12-400000')
    saver_cal_12.restore(sess, 'model/model_cal_12-20000')
    saver_net_24.restore(sess, 'model/model_net_24-210000')
    saver_cal_24.restore(sess, 'model/model_cal_24-80000')
    saver_net_48.restore(sess, 'model/model_net_48-40000')
    saver_cal_48.restore(sess, 'model/model_cal_48-10000')


    # net-12
    pyramid = tuple(pyramid_gaussian(np.array(image), downscale=1.2))
    image_array = pyramid[0]
    window_after_24 = []
    for i, img in enumerate(pyramid):
        slide_return = slide_window(img, 40, 12, 4)
        if slide_return is None:
            break
        img_12 = slide_return[0]
        window_net_12 = slide_return[1]
        w_12 = img_12.shape[1]
        h_12 = img_12.shape[0]

        patch_net_12 = []
        for box in window_net_12:
            patch = img_12[box[0]:box[2], box[1]:box[3], :]
            patch_net_12.append(patch)
        patch_net_12 = np.array(patch_net_12)

        pred_net_12, thresholding = sess.run([net_12['pred'], net_12['thresholding']], feed_dict={net_12['imgs']: patch_net_12})
        thresholding = np.reshape(thresholding, [len(patch_net_12)])
        pred_net_12 = np.reshape(pred_net_12, [len(patch_net_12)])
        sn = np.where(thresholding>0.5)[0]
        if len(sn)==0:
            sn = [np.argmax(thresholding)]
        indices = np.argsort(-pred_net_12[sn])
        window_net = window_net_12[sn]
        window_net = window_net[indices]

        # cal-12
        pred_cal_12 = sess.run(cal_12['pred'], feed_dict={cal_12['imgs']:patch_net_12[sn][indices]})
        for i, pred in enumerate(pred_cal_12):
            s = np.where(pred>0.3)[0]
            if len(s)==0:
                continue
            trans = np.mean(dic[s],axis=0)
            window_net[i][0] = window_net[i][0] + trans[1]*12/trans[0]
            window_net[i][1] = window_net[i][1] + trans[2]*12/trans[0]
            window_net[i][2] = window_net[i][0] + 12*trans[0]
            window_net[i][3] = window_net[i][1] + 12*trans[0]
        detected_list = NMS_fast(window_net, overlap_rate=0.5, included_rate=0.5)
        window_net = window_net[detected_list]

        # net-24
        patch_net_24 = []
        patch_net_12 = []
        for box in window_net:
            if len(box)==0 or len(np.where(box>=0)[0])!=len(box):
                continue
            lt_x = int(float(box[0])*h/h_12)
            lt_y = int(float(box[1])*w/w_12)
            rb_x = int(float(box[2])*h/h_12)
            rb_y = int(float(box[3])*w/w_12)
            patch = image_array[lt_x:rb_x, lt_y:rb_y]
            patch = resize(patch, (24, 24, 3))
            patch_net_24.append(patch)
            patch = resize(patch, (12, 12, 3))
            patch_net_12.append(patch)
        if(len(patch_net_24)==0):
            continue
        patch_net_24 = np.array(patch_net_24)
        patch_net_12 = np.array(patch_net_12)

        feed_dict = {
            net_24['imgs']: patch_net_24,
            net_24['imgs_12']: patch_net_12
        }
        pred_net_24, thresholding = sess.run([net_24['pred'], net_24['thresholding']], feed_dict=feed_dict)
        thresholding = np.reshape(thresholding, [len(patch_net_24)])
        pred_net_24 = np.reshape(pred_net_24, [len(patch_net_24)])
        sn = np.where(thresholding>0.5)[0]
        if len(sn)==0:
            sn = [np.argmax(thresholding)]
        indices = np.argsort(-pred_net_24[sn])
        window_net = window_net[sn]
        window_net = window_net[indices]

        # cal-24
        pred_cal_24 = sess.run(cal_24['pred'], feed_dict={cal_24['imgs']: patch_net_24[sn][indices]})
        for i, pred in enumerate(pred_cal_24):
            s = np.where(pred>0.3)[0]
            if len(s)==0:
                continue
            trans = np.mean(dic[s],axis=0)
            window_net[i][0] = window_net[i][0] + trans[1]*12/trans[0]
            window_net[i][1] = window_net[i][1] + trans[2]*12/trans[0]
            window_net[i][2] = window_net[i][0] + 12*trans[0]
            window_net[i][3] = window_net[i][1] + 12*trans[0]
        detected_list = NMS_fast(window_net, overlap_rate=0.5, included_rate=0.5)
        window_net = window_net[detected_list]

        for box in window_net:
            lt_x = int(float(box[0])*h/h_12)
            lt_y = int(float(box[1])*w/w_12)
            rb_x = int(float(box[2])*h/h_12)
            rb_y = int(float(box[3])*w/w_12)
            window_after_24.append([lt_x, lt_y, rb_x, rb_y])
        # break

    window_after_24 = np.vstack(window_after_24)

    # net-48
    patch_net_48 = []
    patch_net_24 = []
    patch_net_12 = []
    w_h_48 = []
    for box in window_after_24:
        if len(np.where(box>=0)[0])!=len(box):
            continue
        patch = image_array[box[0]:box[2], box[1]:box[3]]
        patch = resize(patch, (48, 48, 3))
        patch_net_48.append(patch)
        patch = resize(patch, (24, 24, 3))
        patch_net_24.append(patch)
        patch = resize(patch, (12, 12, 3))
        patch_net_12.append(patch)
        w_h_48.append([box[3]-box[1], box[2]-box[0]])
    patch_net_48 = np.array(patch_net_48)
    patch_net_24 = np.array(patch_net_24)
    patch_net_12 = np.array(patch_net_12)
    w_h_48 = np.array(w_h_48)

    feed_dict = {
        net_48['imgs']: patch_net_48,
        net_48['imgs_24']: patch_net_24,
        net_48['imgs_12']: patch_net_12
    }
    pred_net_48, thresholding = sess.run([net_48['pred'], net_48['thresholding']], feed_dict=feed_dict)
    thresholding = np.reshape(thresholding, [len(patch_net_48)])
    pred_net_48 = np.reshape(pred_net_48, [len(patch_net_48)])
    sn = np.where(thresholding>0.5)[0]
    indices = np.argsort(-pred_net_48[sn])
    window_net = window_after_24[sn]
    window_net = window_net[indices]
    w_h_48 = w_h_48[sn][indices]

    if window_net.shape[0]==0:
        print('No Faces Detected.')
        coord.request_stop()
        exit()

    # cal-48
    pred_cal_48 = sess.run(cal_48['pred'], feed_dict={cal_48['imgs']:patch_net_48[sn][indices]})
    for i, pred in enumerate(pred_cal_12):
        s = np.where(pred>0.6)[0]
        if len(s)==0:
            continue
        trans = np.mean(dic[s],axis=0)
        window_net[i][0] = window_net[i][0] + trans[1]*w_h_48[i][1]/trans[0]
        window_net[i][1] = window_net[i][1] + trans[2]*w_h_48[i][0]/trans[0]
        window_net[i][2] = window_net[i][0] + w_h_48[i][1]*trans[0]
        window_net[i][3] = window_net[i][1] + w_h_48[i][0]*trans[0]
    detected_list = NMS_fast(window_net, overlap_rate=0.1, included_rate=0.1)
    window_net = window_net[detected_list]

    print(window_net.shape)
    for box in window_net:
        ImageDraw.Draw(image).rectangle((box[1], box[0], box[3], box[2]), outline = "red")
    image.show()
        

    coord.request_stop()
    coord.join(threads)

    sess.close()
