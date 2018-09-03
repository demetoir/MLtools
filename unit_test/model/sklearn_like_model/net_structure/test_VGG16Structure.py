from script.model.sklearn_like_model.net_structure.VGG16Structure import VGG16Structure
import tensorflow as tf
import numpy as np

from script.util.deco import deco_timeit


@deco_timeit
def test_VGG16Structure():
    x = tf.placeholder(tf.float32, [None, 300, 300, 3])
    n_classes = 2
    net = VGG16Structure(x, n_classes)
    net.build()

    h = net.h
    x_np = np.random.normal(0, 1, [10, 300, 300, 3])
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        forward = sess.run(h, feed_dict={x: x_np})
        print(forward.shape)
