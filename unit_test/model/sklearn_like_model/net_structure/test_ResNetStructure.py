import numpy as np
import tensorflow as tf

from script.model.sklearn_like_model.net_structure.ResNetStructure.ResNet101Structure import ResNet101Structure
from script.model.sklearn_like_model.net_structure.ResNetStructure.ResNet152Structure import ResNet152Structure
from script.model.sklearn_like_model.net_structure.ResNetStructure.ResNet18Structure import ResNet18Structure
from script.model.sklearn_like_model.net_structure.ResNetStructure.ResNet50Structure import ResNet50Structure
from script.util.deco import deco_timeit
from script.util.elapse_time import elapse_time


@deco_timeit
def test_ResNet18Structure():
    x = tf.placeholder(tf.float32, [None, 128, 128, 3])
    n_classes = 10

    net = ResNet18Structure(x, n_classes)
    with elapse_time('build'):
        net.build()

    h = net.proba
    x_np = np.random.normal(0, 1, [10, 128, 128, 3])
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        with elapse_time('forward time'):
            forward = sess.run(h, feed_dict={x: x_np})

        print(forward.shape)

        # show_summary()
        #
        # pprint(net.vars)


@deco_timeit
def test_ResNet34Structure():
    x = tf.placeholder(tf.float32, [None, 128, 128, 3])
    n_classes = 10

    net = ResNet50Structure(x, n_classes)
    with elapse_time('build'):
        net.build()

    h = net.proba
    x_np = np.random.normal(0, 1, [10, 128, 128, 3])
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        with elapse_time('forward time'):
            forward = sess.run(h, feed_dict={x: x_np})

        print(forward.shape)

        # show_summary()
        #
        # pprint(net.vars)


@deco_timeit
def test_ResNet50Structure():
    x = tf.placeholder(tf.float32, [None, 128, 128, 3])
    n_classes = 10

    net = ResNet50Structure(x, n_classes)
    with elapse_time('build'):
        net.build()

    h = net.proba
    x_np = np.random.normal(0, 1, [10, 128, 128, 3])
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        with elapse_time('forward time'):
            forward = sess.run(h, feed_dict={x: x_np})

        print(forward.shape)

        # show_summary()
        #
        # pprint(net.vars)


@deco_timeit
def test_ResNet101Structure():
    x = tf.placeholder(tf.float32, [None, 128, 128, 3])
    n_classes = 10

    net = ResNet101Structure(x, n_classes)
    with elapse_time('build'):
        net.build()

    h = net.proba
    x_np = np.random.normal(0, 1, [10, 128, 128, 3])
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        with elapse_time('forward time'):
            forward = sess.run(h, feed_dict={x: x_np})

        print(forward.shape)

        # show_summary()
        #
        # pprint(net.vars)


@deco_timeit
def test_ResNet152Structure():
    x = tf.placeholder(tf.float32, [None, 128, 128, 3])
    n_classes = 10

    net = ResNet152Structure(x, n_classes)
    with elapse_time('build'):
        net.build()

    h = net.proba
    x_np = np.random.normal(0, 1, [10, 128, 128, 3])
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        with elapse_time('forward time'):
            forward = sess.run(h, feed_dict={x: x_np})

        print(forward.shape)

        # show_summary()
        #
        # pprint(net.vars)


def test_ResNetStructure_full():
    test_ResNet18Structure()
    test_ResNet34Structure()
    test_ResNet50Structure()
    test_ResNet101Structure()
    test_ResNet152Structure()
