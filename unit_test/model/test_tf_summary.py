import tensorflow as tf

from script.model.sklearn_like_model.TFSummary import TFSummary


def test_tf_summary():
    xs = [i for i in range(10)]

    for i in range(4):
        tf.reset_default_graph()

        with tf.Session() as sess:
            summary_a = TFSummary(f'./tf_summary/inside{i}', 'a')
            summary_b = TFSummary(sess, f'./tf_summary/inside{i}', 'b')
            for x in xs:
                summary_a.update(sess, x * (i + 1))
                summary_b.update(sess, x * (i + 1 + .3))
