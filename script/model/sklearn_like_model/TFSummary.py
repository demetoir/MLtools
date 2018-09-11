import os
import tensorflow as tf
import numpy as np
from script.util.MixIn import LoggerMixIn
from script.util.misc_util import setup_directory, error_trace


class TFSummaryBuildError(BaseException):
    pass


class TFSummary(LoggerMixIn):
    def __init__(self, logdir, name, device='/cpu:0', epoch=0, verbose=0):
        super().__init__(verbose)
        self.logdir = logdir
        setup_directory(self.logdir)
        self.name = name
        self.device = device
        self.writer = None
        self.epoch = epoch
        self.sess = None
        self.is_build = False
        self.x_shape = None

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return self.__class__.__name__

    def build_graph(self, name):
        raise NotImplementedError

    def init_shape(self, x):
        raise NotImplementedError

    def build(self):
        with tf.device(self.device):
            try:
                self.ph, self.summary_op = self.build_graph(self.name)

                self.writer_path = os.path.join(self.logdir, self.name)
                self.writer = tf.summary.FileWriter(self.writer_path, self.sess.graph)
                self.is_build = True

                self.log.info(f'build summary tensor={self.name}, writer_path={self.writer_path}')
            except BaseException as e:
                self.log.error(error_trace(e))
                raise TFSummaryBuildError(e)

    def update(self, sess, x, epoch=None):
        if not self.is_build:
            self.x_shape = self.init_shape(x)
            self.sess = sess
            self.build()

        if epoch is None:
            epoch = self.epoch
            self.epoch += 1

        summary = self.sess.run(self.summary_op, feed_dict={self.ph: x})
        self.writer.add_summary(summary, epoch)

    def flush(self):
        self.writer.flush()

    def close(self):
        self.writer.close()


class TFSummaryScalar(TFSummary):
    def build_graph(self, name):
        ph = tf.placeholder(tf.float32, name=f'ph_{self.name}')
        summary_op = tf.summary.scalar(name, ph)
        return ph, summary_op

    def init_shape(self, x):
        return []


class TFSummaryParams(TFSummary):
    @staticmethod
    def dict_to_np(d):
        k = list(d.keys())
        v = list(d.values())
        return np.stack([np.array(k), np.array(v)]).transpose()

    def update(self, sess, x, epoch=None):
        if type(x) is dict:
            x = self.dict_to_np(x)
        super().update(sess, x, epoch)

    def build_graph(self, name):
        ph = tf.placeholder(tf.string, name=f'ph_{self.name}')
        summary_op = tf.summary.text(name, ph)
        return ph, summary_op

    def init_shape(self, x):
        return []
