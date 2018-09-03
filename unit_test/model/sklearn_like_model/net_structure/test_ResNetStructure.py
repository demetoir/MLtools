import numpy as np
import tensorflow as tf
from script.model.sklearn_like_model.net_structure.ResNetStructure import ResNetStructure
from script.util.deco import deco_timeit
from script.util.elapse_time import elapse_time


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
