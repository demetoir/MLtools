from script.util.Stacker import Stacker
from script.util.tensor_ops import *


def test_add_stack():
    inner_stack = Stacker(name='inner')
    inner_stack.linear_block(10, relu)
    inner_stack.linear_block(10, relu)
    inner_stack.linear_block(10, relu)

    x = placeholder(tf.float32, [-1, 100], 'x')
    stack = Stacker(x, name='outer')
    stack.linear_block(10, relu)
    stack.linear_block(10, relu)
    stack.add_stacker(inner_stack)
    stack.linear_block(10, relu)

    import numpy as np
    x_np = np.zeros([10, 10])

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        a = sess.run(stack.last_layer, {stack.last_layer: x_np})
        print(a)
