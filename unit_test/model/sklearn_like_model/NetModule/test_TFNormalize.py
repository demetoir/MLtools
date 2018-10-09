from script.model.sklearn_like_model.NetModule.TFNormalize import TFL1Normalize, TFL2Normalize
from script.util.Stacker import Stacker
from script.util.tensor_ops import *


def test_TFL1Normalize():
    import numpy as np
    x = np.random.normal(size=[100, 10])
    y = np.random.normal(size=[100, 1])
    x_ph = placeholder(tf.float32, [-1, 10], name='ph_x')

    with tf.variable_scope('net'):
        stack = Stacker(x_ph)
        stack.linear_block(100, relu)
        stack.linear_block(100, relu)
        logit = stack.linear(1)
        proba = stack.softmax()

        loss = (proba - y) ** 2

    var_list = collect_vars('net')
    l1_norm = TFL1Normalize(var_list)
    l1_norm.build()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        val = sess.run(l1_norm.penalty, feed_dict={x_ph: x})
        rate = sess.run(l1_norm.rate_var)
        print(f'l1_norm = {val}, rate = {rate}')

        l1_norm.update_rate(sess, 0.5)
        val = sess.run(l1_norm.penalty, feed_dict={x_ph: x})
        rate = sess.run(l1_norm.rate_var)
        print(f'l1_norm = {val}, rate = {rate}')

        l1_norm.update_rate(sess, 0.1)
        val = sess.run(l1_norm.penalty, feed_dict={x_ph: x})
        rate = sess.run(l1_norm.rate_var)
        print(f'l1_norm = {val}, rate = {rate}')


def test_TFL2Normalize():
    import numpy as np
    x = np.random.normal(size=[100, 10])
    y = np.random.normal(size=[100, 1])
    x_ph = placeholder(tf.float32, [-1, 10], name='ph_x')

    with tf.variable_scope('net'):
        stack = Stacker(x_ph)
        stack.linear_block(100, relu)
        stack.linear_block(100, relu)
        logit = stack.linear(1)
        proba = stack.softmax()

        loss = (proba - y) ** 2

    var_list = collect_vars('net')
    l2_norm = TFL2Normalize(var_list)
    l2_norm.build()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        val = sess.run(l2_norm.penalty, feed_dict={x_ph: x})
        rate = sess.run(l2_norm.rate_var)
        print(f'l2_norm = {val}, rate = {rate}')

        l2_norm.update_rate(sess, 0.5)
        val = sess.run(l2_norm.penalty, feed_dict={x_ph: x})
        rate = sess.run(l2_norm.rate_var)
        print(f'l2_norm = {val}, rate = {rate}')

        l2_norm.update_rate(sess, 0.1)
        val = sess.run(l2_norm.penalty, feed_dict={x_ph: x})
        rate = sess.run(l2_norm.rate_var)
        print(f'l2_norm = {val}, rate = {rate}')
