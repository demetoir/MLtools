import time

from tensorflow.contrib import slim
from script.model.sklearn_like_model.net_structure.ResNetStructure import ResNetStructure
import tensorflow as tf
import numpy as np
from script.util.deco import deco_timeit


# TODO refactoring move
def show_summary():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


# TODO refactoring move
class elapse_time:
    def __init__(self, title=None):
        self.start_time = time.time()
        self.title = title

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.title:
            print(self.title)
        print(f"time {time.time() - self.start_time:.4f}'s elapsed")
        return True

    def __enter__(self):
        return None


@deco_timeit
def test_ResNetStructure():
    x = tf.placeholder(tf.float32, [None, 128, 128, 3])
    n_classes = 10
    model_type = 18
    # model_type = 34
    # model_type = 50
    # model_type = 101
    # model_type = 152
    net = ResNetStructure(x, n_classes, model_type=model_type)
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
