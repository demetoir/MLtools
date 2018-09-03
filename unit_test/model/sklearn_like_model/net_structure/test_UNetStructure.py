from tensorflow.contrib import slim

from script.model.sklearn_like_model.net_structure.UNetStructure import UNetStructure
import tensorflow as tf
import numpy as np

from script.util.deco import deco_timeit


def show_summary():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


@deco_timeit
def test_UNetStructure():
    x = tf.placeholder(tf.float32, [None, 128, 128, 3])
    net = UNetStructure(x, level=3)
    net.build()

    h = net.proba
    x_np = np.random.normal(0, 1, [10, 128, 128, 3])
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # show_summary()

        forward = sess.run(h, feed_dict={x: x_np})
        print(forward.shape)
