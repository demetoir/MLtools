import tensorflow as tf
from script.model.sklearn_like_model.TFSummary import TFSummaryScalar, TFSummaryParams


def test_TFSummaryParam():
    param1 = {
        'a': 1,
        'b': 'str'
    }
    param2 = {
        'a': 1,
        'b': 'str'
    }

    for i in range(1):
        tf.reset_default_graph()

        with tf.Session() as sess:
            summary_a = TFSummaryParams(f'./test_tf_summary/Params/{i}', 'param1')
            summary_b = TFSummaryParams(f'./test_tf_summary/params/{i}', 'param2')

            summary_a.update(sess, param1)
            summary_b.update(sess, param2)


def test_TFSummaryScalar():
    xs = [i for i in range(10)]

    for i in range(4):
        tf.reset_default_graph()

        with tf.Session() as sess:
            summary_a = TFSummaryScalar(f'./test_tf_summary/Scalar/{i}', 'a')
            summary_b = TFSummaryScalar(f'./test_tf_summary/Scalar/{i}', 'b')
            for x in xs:
                summary_a.update(sess, x * (i + 1))
                summary_b.update(sess, x * (i + 1 + .3))
