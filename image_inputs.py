import tensorflow as tf


def read_my_file_format(filename):

    record_defaults = [[""]] + [[0]]

    components = tf.decode_csv(filename, record_defaults=record_defaults, field_delim=" ")
    imgName = components[0]
    label = components[1:]
    img_contents = tf.read_file(imgName)
    img = tf.image.decode_jpeg(img_contents, channels=3)

    return img, label


def inputs(lists, image_shape, batch_size):

    filename_queue = tf.train.string_input_producer(lists, shuffle=True)
    reader = tf.TextLineReader()
    _, value = reader.read(filename_queue)
    image, label = read_my_file_format(value)
    image = tf.image.resize_images(image, [image_shape[0]+3, image_shape[1]+3])
    image = tf.random_crop(image, image_shape)
    label = tf.cast(label, tf.float32)

    image.set_shape(image_shape)
    # image = tf.image.random_flip_left_right(image)
    float_image = tf.image.per_image_whitening(image)

    min_after_dequeue = 1000
    capacity = min_after_dequeue+(2+1)*batch_size

    image_batch, label_batch = tf.train.shuffle_batch([float_image, label],
                                                    batch_size=batch_size,
                                                    capacity=capacity,
                                                    min_after_dequeue=min_after_dequeue)

    return image_batch, label_batch


def inputs_without_crop(lists, image_shape, batch_size):

    filename_queue = tf.train.string_input_producer(lists, shuffle=True)
    reader = tf.TextLineReader()
    _, value = reader.read(filename_queue)
    image, label = read_my_file_format(value)
    image = tf.image.resize_images(image, [image_shape[0], image_shape[1]])
    # image = tf.random_crop(image, image_shape)
    label = tf.cast(label, tf.float32)

    image.set_shape(image_shape)
    # image = tf.image.random_flip_left_right(image)
    float_image = tf.image.per_image_whitening(image)

    min_after_dequeue = 1000
    capacity = min_after_dequeue+(2+1)*batch_size

    # image_batch, label_batch = tf.train.shuffle_batch([float_image, label],
    #                                                 batch_size=batch_size,
    #                                                 capacity=capacity,
    #                                                 min_after_dequeue=min_after_dequeue)
    image_batch, label_batch = tf.train.batch([float_image, label],
                                            batch_size=batch_size,
                                            capacity=128)

    return image_batch, label_batch


def inputs_for_test(lists, image_shape, batch_size):

    filename_queue = tf.train.string_input_producer(lists, shuffle=True)
    reader = tf.TextLineReader()
    _, value = reader.read(filename_queue)
    image, label = read_my_file_format(value)
    image = tf.image.resize_images(image, [image_shape[0], image_shape[1]])
    # image = tf.random_crop(image, image_shape)
    label = tf.cast(label, tf.float32)

    image.set_shape(image_shape)
    # image = tf.image.random_flip_left_right(image)
    float_image = tf.image.per_image_whitening(image)

    min_after_dequeue = 1000
    capacity = min_after_dequeue+(2+1)*batch_size

    image_batch, label_batch = tf.train.batch([float_image, label],
                                            batch_size=batch_size)

    return image_batch, label_batch


if __name__ == '__main__':

    image_batch, label_batch = inputs(['net_12_list.txt'], [12, 12, 3], 10)

    sess = tf.Session()

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for i in range(100):
        ibatch, lbatch = sess.run([image_batch, label_batch])
        print(lbatch)


    coord.request_stop()

    coord.join(threads)

    sess.close()