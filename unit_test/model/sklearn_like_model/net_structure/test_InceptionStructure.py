from pprint import pprint

import numpy as np
import tensorflow as tf

from script.model.sklearn_like_model.net_structure.InceptionSructure.InceptionV4Structure import InceptionV4Structure
from script.model.sklearn_like_model.net_structure.InceptionSructure.InceptionV2Structure import InceptionV2Structure
from script.model.sklearn_like_model.net_structure.InceptionSructure.InceptionV1Structure import InceptionV1Structure
from script.util.deco import deco_timeit
from script.util.elapse_time import elapse_time


@deco_timeit
def test_InceptionV1Structure():
    n_classes = 10
    # model_types = [1, 2, 3, 4]
    model_types = [1]
    for model_type in model_types:
        print(f'test_model_type= {model_type}')
        x = tf.placeholder(tf.float32, [None, 128, 128, 3])
        net = InceptionV1Structure(x, n_classes, model_type=model_type)
        with elapse_time('build'):
            net.build()

        x_np = np.random.normal(0, 1, [10, 128, 128, 3])
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            with elapse_time('net proba forward time'):
                forward = sess.run(net.proba, feed_dict={x: x_np})
            print(forward.shape)
            with elapse_time('aux0 proba forward time'):
                forward = sess.run(net.aux0_proba, feed_dict={x: x_np})
            print(forward.shape)
            with elapse_time('aux1 proba forward time'):
                forward = sess.run(net.aux1_proba, feed_dict={x: x_np})

            print(forward.shape)

            # show_summary()
            #
            # pprint(net.vars)


@deco_timeit
def test_InceptionV2Structure():
    n_classes = 10
    # model_types = [1, 2, 3, 4]
    model_type = 2
    x = tf.placeholder(tf.float32, [None, 128, 128, 3])
    net = InceptionV2Structure(x, n_classes, model_type=model_type)
    with elapse_time('build'):
        net.build()

    x_np = np.random.normal(0, 1, [10, 128, 128, 3])
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        with elapse_time('net proba forward time'):
            forward = sess.run(net.proba, feed_dict={x: x_np})
        print(forward.shape)

        with elapse_time('aux proba forward time'):
            forward = sess.run(net.aux_proba, feed_dict={x: x_np})
            print(forward.shape)


@deco_timeit
def test_InceptionV4Structure():
    n_classes = 10
    # model_types = [1, 2, 3, 4]
    model_type = 4
    x = tf.placeholder(tf.float32, [None, 128, 128, 3])
    net = InceptionV4Structure(x, n_classes, model_type=model_type)
    with elapse_time('build'):
        net.build()

    x_np = np.random.normal(0, 1, [10, 128, 128, 3])
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        with elapse_time('net proba forward time'):
            forward = sess.run(net.proba, feed_dict={x: x_np})
        print(forward.shape)

        with elapse_time('aux proba forward time'):
            forward = sess.run(net.aux_proba, feed_dict={x: x_np})
        print(forward.shape)

        # show_summary()
        #
        # pprint(net.vars)
