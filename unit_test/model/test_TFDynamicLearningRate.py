from script.model.sklearn_like_model.TFDynamicLearningRate import TFDynamicLearningRate
from script.util.Stacker import Stacker
from script.util.tensor_ops import *


def test_TFDynamicLearningRate():
    import numpy as np
    x = np.random.normal(size=[100, 10])
    y = np.random.normal(size=[100, 1])
    x_ph = placeholder(tf.float32, [-1, 10], name='ph_x')

    stack = Stacker(x_ph)
    stack.linear_block(100, relu)
    stack.linear_block(100, relu)
    logit = stack.linear(1)
    proba = stack.softmax()

    loss = (proba - y) ** 2
    dlr = TFDynamicLearningRate(0.01)
    dlr.build()

    lr_var = dlr.learning_rate
    var_list = None
    train_op = tf.train.AdamOptimizer(learning_rate=lr_var, beta1=0.9).minimize(loss, var_list=var_list)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(f'lr = {dlr.lr_tensor(sess)}')

        dlr.update(sess, 0.1)
        print(f'lr = {dlr.lr_tensor(sess)}')
        sess.run(train_op, feed_dict={x_ph: x})
        print(f'loss = {np.mean(sess.run(loss, feed_dict={x_ph: x}))}')

        dlr.update(sess, 0.05)
        print(f'lr = {dlr.lr_tensor(sess)}')
        sess.run(train_op, feed_dict={x_ph: x})
        print(f'loss = {np.mean(sess.run(loss, feed_dict={x_ph: x}))}')

        dlr.update(sess, 0.02)
        print(f'lr = {dlr.lr_tensor(sess)}')
        sess.run(train_op, feed_dict={x_ph: x})
        print(f'loss = {np.mean(sess.run(loss, feed_dict={x_ph: x}))}')
