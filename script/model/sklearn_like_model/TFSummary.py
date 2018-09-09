import os
import tensorflow as tf
from script.util.MixIn import LoggerMixIn
from script.util.misc_util import setup_directory, error_trace


class TFSummaryBuildError(BaseException):
    pass


class TFSummary(LoggerMixIn):

    def __init__(self, logdir, name, summary_type='scalar', device='/cpu:0',
                 epoch=0, verbose=0):
        super().__init__(verbose)
        self.logdir = logdir
        setup_directory(self.logdir)
        self.name = name
        self.summary_type = summary_type
        self.device = device
        self.writer = None
        self.epoch = epoch
        self.sess = None
        self.is_build = False
        self.x_shape = None

        self.summary_types = {
            'scalar': None,
            'image': None,
            'audio': None,
            'text': None,
            'histogram': None
        }
        self.init_shape_funcs_by_type = {
            'scalar': self._init_shape_scalar,
            'image': self._init_shape_image,
            'audio': self._init_shape_audio,
            'text': self._init_shape_text,
            'histogram': self._init_shape_histogram,
        }
        self.build_funcs_by_type = {
            'scalar': self._build_scalar,
            'image': self._build_image,
            'audio': self._build_audio,
            'text': self._build_text,
            'histogram': self._build_histogram,
        }

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return self.__class__.__name__

    def _build_scalar(self, name):
        ph = tf.placeholder(tf.float32, name=f'ph_{self.name}')
        summary_op = tf.summary.scalar(name, ph)
        return ph, summary_op

    def _build_image(self):
        # TODO
        raise NotImplementedError
        pass

    def _build_audio(self):
        # TODO
        raise NotImplementedError

    def _build_text(self):
        # TODO
        raise NotImplementedError

    def _build_histogram(self):
        # TODO
        raise NotImplementedError

    def build(self):
        with tf.device(self.device):
            try:
                self.ph, self.summary_op = self.build_funcs_by_type[self.summary_type](self.name)

                self.writer_path = os.path.join(self.logdir, self.name)
                self.writer = tf.summary.FileWriter(self.writer_path, self.sess.graph)
                self.is_build = True

                self.log.info(f'build summary tensor={self.name}, writer_path={self.writer_path}')
            except BaseException as e:
                self.log.error(error_trace(e))
                raise TFSummaryBuildError(e)

    def _init_shape_scalar(self, x):
        return []

    def _init_shape_image(self, x):
        # TODO
        raise NotImplementedError

    def _init_shape_audio(self, x):
        # TODO
        raise NotImplementedError

    def _init_shape_text(self, x):
        # TODO
        raise NotImplementedError

    def _init_shape_histogram(self, x):
        # TODO
        raise NotImplementedError

    def update(self, sess, x, epoch=None):
        if not self.is_build:
            self.x_shape = self.init_shape_funcs_by_type[self.summary_type](x)
            self.sess = sess
            self.build()

        if epoch is None:
            epoch = self.epoch
            self.epoch += 1

        summary = self.sess.run(self.summary_op, feed_dict={self.ph: x})
        self.writer.add_summary(summary, epoch)

    def close(self):
        self.writer.close()
