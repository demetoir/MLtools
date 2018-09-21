from pprint import pprint

from script.model.sklearn_like_model.MLPClassifier import MLPClassifier
from script.util.MixIn import LoggerMixIn
from script.util.misc_util import path_join, temp_directory, error_trace, time_stamp
from script.util.tensor_ops import *


class TransferLearning(LoggerMixIn):
    def __init__(self, source_model, source_save_path, source_scope, verbose=0):
        super().__init__(verbose=verbose)

        if not source_model.is_built:
            raise RuntimeError(f'transfer fail, source model must be built')

        self.source_model = source_model
        self.source_save_path = source_save_path
        self.source_scope = source_scope

        self.temp_path = f'./temp_transfer_{time_stamp()}'
        self.transfer_path = path_join(self.temp_path, 'transfer')
        self.target_path = path_join(self.temp_path, 'target')

    @staticmethod
    def build_transfer_dict(source_var_list, source_scope, target_scope):
        var_dict = {}
        for var in source_var_list:
            var_name = str(var.name).split(':')[0]
            var_name = var_name.replace(source_scope, target_scope)
            var_dict[var_name] = var

        return var_dict

    def to(self, target, target_scope):
        if not target.is_built:
            raise RuntimeError(f'transfer fail, target model must be built')

        with temp_directory(self.temp_path):
            try:
                source_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.source_scope)
                target_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=target_scope)
                self.log.info(f'collect var_list')

                transfer_dict = self.build_transfer_dict(source_var_list, self.source_scope, target_scope)
                self.log.info(f'build transfer_dict')

                # save transfer
                saver = tf.train.Saver(transfer_dict, name='transfer_saver')
                saver.save(self.source_model.sess, self.temp_path)
                self.log.info(f'save transfer')

                # load transfer
                saver = tf.train.Saver(target_var_list)
                saver.restore(target.sess, self.temp_path)
                self.log.info(f'load transfer')
            except BaseException as e:
                self.log.error(error_trace(e))

        return target


def test_transfer_weight():
    import numpy as np
    x = np.random.normal(size=[10, 10])
    y = np.ones([10, 2])

    source_path = './test_instance/source'
    source = MLPClassifier()
    source.train(x, y, epoch=1)
    source.save(source_path)
    del source

    source = MLPClassifier()
    source_var_list = []
    target = MLPClassifier()
    target_var_list = []
    target.build(x=x, y=y)

    transfer = TransferLearning(source, source_path, source_var_list)
    target = transfer.to(target, target_var_list)
    target.train(x, y, epoch=1)


def test_save_rename_var():
    import tensorflow as tf
    import numpy as np

    x_np = np.ones([5, 5])

    class model:
        def __init__(self, name):
            with tf.variable_scope(name):
                self.x_ph = tf.placeholder(tf.float32, [None, 5], name='ph')
                self.var_1 = tf.get_variable(name='var_1', initializer=2.)
                self.var_2 = tf.get_variable(name='var_2', initializer=3.)
                self.update = tf.assign_add(self.var_1, 7)
                self.result = self.x_ph * self.var_1 * self.var_2

                print(self.x_ph)
                print(self.var_1)
                print(self.var_2)
                print(self.result)
                # need to refactoring tensor ops collect vars
                self.vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
                print(self.vars)
                # a = tf.get_variable()
                # print(a)

    def scope_transform(vars, old=None, new=None):
        transformed = {}
        print('transform')
        for var in vars:
            name = var.name.split(':')[0]

            new_name = []
            for token in name.split('/'):
                if token == old:
                    new_name += [new]
                else:
                    new_name += [token]

            new_name = '/'.join(new_name)
            print(var, name, new_name)
            transformed[new_name] = var

        return transformed

    path = './test_instance/instance.pkl'
    import os
    if os.path.exists(path):
        os.remove(path)

    a = model('a')

    with tf.Session() as sess:
        save_vars = scope_transform(a.vars, 'a', 'b')
        pprint(save_vars)
        saver = tf.train.Saver(save_vars)

        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        sess.run(a.update)
        val = sess.run(a.result, feed_dict={a.x_ph: x_np})
        print(val)
        saver.save(sess, path)

    print('reset graph')
    tf.reset_default_graph()
    print()

    b = model('b')
    c = model('c')
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        saver = tf.train.Saver(b.vars)
        # saver = tf.train.Saver()

        saver.restore(sess, path)
        val = sess.run(b.result, feed_dict={b.x_ph: x_np})
        print(val)

        val = sess.run(c.result, feed_dict={c.x_ph: x_np})
        print(val)

        a = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='a')
        print(a)

        a = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='b')
        print(a)

        a = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='c')
        print(a)

        a = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='d')
        print(a)

    sess1 = tf.Session()
    sess2 = tf.Session()
