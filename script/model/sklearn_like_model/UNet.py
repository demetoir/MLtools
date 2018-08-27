from tqdm import trange
from script.model.sklearn_like_model.BaseModel import BaseModel
from script.model.sklearn_like_model.Mixin import Xs_MixIn, Ys_MixIn, supervised_trainMethodMixIn, predictMethodMixIn, \
    predict_probaMethodMixIn, scoreMethodMixIn, supervised_metricMethodMixIn
from script.util.Stacker import Stacker
from script.util.tensor_ops import *


class segmentation_loss_mixIn:

    def __init__(self):
        # todo
        # iou
        # dice
        # pixel wise softmax

        cls = self.__class__

        self.loss_builder_func = {
            'iou': cls._iou,
            'dice_soft': cls._dice_soft,
            'dice_hard': cls._dice_hard,
            'pixel_wise_softmax': cls._pixel_wise_softmax
        }

        # out_seg = net.outputs
        # dice_loss = 1 - tl.cost.dice_coe(out_seg, t_seg, axis=[0, 1, 2, 3])  # , 'jaccard', epsilon=1e-5)
        # iou_loss = tl.cost.iou_coe(out_seg, t_seg, axis=[0, 1, 2, 3])
        # dice_hard = tl.cost.dice_hard_coe(out_seg, t_seg, axis=[0, 1, 2, 3])
        # loss = dice_loss

    def _build_loss(self, loss_type, label, logit):
        return self.loss_builder_func[loss_type](label, logit)

    def _iou(self, label, logit):
        logits = tf.reshape(logit, [-1])
        trn_labels = tf.reshape(label, [-1])

        '''
        Eq. (1) The intersection part - tf.mul is element-wise, 
        if logits were also binary then tf.reduce_sum would be like a bitcount here.
        '''

        inter = tf.reduce_sum(logits * trn_labels)

        '''
        Eq. (2) The union part - element-wise sum and multiplication, then vector sum
        '''
        union = tf.reduce_sum(logits + trn_labels - inter * trn_labels)
        loss = 1 - (inter / union)
        # Eq. (4)
        # loss = tf.sub(tf.constant(1.0, dtype=tf.float32), tf.div(inter, union))

        return loss

    def _dice_soft(self, label, logit):
        return None

    def _dice_hard(self, label, logit):
        logits = tf.reshape(logit, [-1])
        trn_labels = tf.reshape(label, [-1])

        '''
        Eq. (1) The intersection part - tf.mul is element-wise, 
        if logits were also binary then tf.reduce_sum would be like a bitcount here.
        '''

        inter = tf.reduce_sum(logits * trn_labels)

        '''
        Eq. (2) The union part - element-wise sum and multiplication, then vector sum
        '''
        union = tf.reduce_sum(logits + trn_labels - inter * trn_labels)
        loss = 1 - (2 * inter / union)
        # Eq. (4)
        # loss = tf.sub(tf.constant(1.0, dtype=tf.float32), tf.div(inter, union))
        return loss

    def _pixel_wise_softmax(self, label, logit, n_classes):
        logits = tf.reshape(logit, (-1, n_classes))
        label = tf.reshape(label, [-1])
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, label, name='loss')
        # loss = tf.reduce_mean(cross_entropy, name='x_ent_mean')
        return cross_entropy


class UNet(
    BaseModel,
    Xs_MixIn,
    Ys_MixIn,
    supervised_trainMethodMixIn,
    predictMethodMixIn,
    predict_probaMethodMixIn,
    scoreMethodMixIn,
    supervised_metricMethodMixIn,
    segmentation_loss_mixIn
):

    def __init__(self, verbose=10, net_shapes=(), learning_rate=0.001, learning_rate_decay_rate=0.99,
                 learning_rate_decay_method=None, beta1=0.01, batch_size=100, stage=4, loss_type=None, n_classes=1,
                 **kwargs):
        BaseModel.__init__(self, verbose, **kwargs)
        Xs_MixIn.__init__(self)
        Ys_MixIn.__init__(self)
        supervised_trainMethodMixIn.__init__(self, None)
        predictMethodMixIn.__init__(self)
        predict_probaMethodMixIn.__init__(self)
        scoreMethodMixIn.__init__(self)
        supervised_metricMethodMixIn.__init__(self)
        segmentation_loss_mixIn.__init__(self)

        self.net_shapes = net_shapes
        self.learning_rate = learning_rate
        self.learning_rate_decay_method = learning_rate_decay_method
        self.learning_rate_decay_rate = learning_rate_decay_rate
        self.beta1 = beta1
        self.batch_size = batch_size
        self.stage = stage
        self.loss_type = loss_type
        self.n_classes = n_classes

    def _build_input_shapes(self, shapes):
        ret = {}
        ret.update(self._build_Xs_input_shape(shapes))
        ret.update(self._build_Ys_input_shape(shapes))

    def Unet_recursion(self, Xs, level=4, n_classes=1, reuse=False, name='Unet'):

        def _Unet_recursion(stacker, channel_size, level):
            if level == 0:
                stacker.conv_block(channel_size, CONV_FILTER_3311, relu)
                stacker.conv_block(channel_size, CONV_FILTER_3311, relu)
            else:
                # encode
                stacker.conv_block(64, CONV_FILTER_3311, relu)
                stacker.conv_block(64, CONV_FILTER_3311, relu)
                concat = stacker.last_layer
                stacker.max_pooling(CONV_FILTER_2211)

                stacker = _Unet_recursion(stacker, channel_size * 2, level - 1)

                # decode
                stacker.upscale_2x_block(64, CONV_FILTER_2211, relu)
                stacker.concat(concat, axis=3)
                stacker.conv_block(64, CONV_FILTER_3311, relu)
                stacker.conv_block(64, CONV_FILTER_3311, relu)

            return stacker

        with tf.variable_scope(name, reuse=reuse):
            stack = Stacker(Xs)
            stack = _Unet_recursion(stack, channel_size=64, level=level)
            stack.conv_block(n_classes, CONV_FILTER_3311, relu)
            h = stack.last_layer
        return h

    def Unet_hard_coding(self, Xs, reuse=False, name='Unet'):
        with tf.variable_scope(name, reuse=reuse):
            stack = Stacker(Xs)
            # encoder
            # 128 * 128 * 64
            stack.conv_block(64, CONV_FILTER_3311, relu)
            stack.conv_block(64, CONV_FILTER_3311, relu)
            stage_1_concat = stack.last_layer
            stack.max_pooling(CONV_FILTER_2211)

            # 64 * 64 * 128
            stack.conv_block(128, CONV_FILTER_3311, relu)
            stack.conv_block(128, CONV_FILTER_3311, relu)
            stage_2_concat = stack.last_layer
            stack.max_pooling(CONV_FILTER_2211)

            # 32 * 32 * 256
            stack.conv_block(256, CONV_FILTER_3311, relu)
            stack.conv_block(256, CONV_FILTER_3311, relu)
            stage_3_concat = stack.last_layer
            stack.max_pooling(CONV_FILTER_2211)

            # 16 * 16 * 512
            stack.conv_block(512, CONV_FILTER_3311, relu)
            stack.conv_block(512, CONV_FILTER_3311, relu)
            stage_4_concat = stack.last_layer
            stack.max_pooling(CONV_FILTER_2211)

            # 8 * 8 * 1024
            # low stage
            stack.conv_block(1024, CONV_FILTER_3311, relu)
            stack.conv_block(1024, CONV_FILTER_3311, relu)

            # decoder
            # 16 * 16 512
            stack.upscale_2x_block(512, CONV_FILTER_2211, relu)
            stack.concat(stage_4_concat, axis=3)
            stack.conv_block(512, CONV_FILTER_3311, relu)
            stack.conv_block(512, CONV_FILTER_3311, relu)

            # 32 * 32 256
            stack.upscale_2x_block(256, CONV_FILTER_2211, relu)
            stack.concat(stage_3_concat, axis=3)
            stack.conv_block(256, CONV_FILTER_3311, relu)
            stack.conv_block(256, CONV_FILTER_3311, relu)

            # 64 * 64 128
            stack.upscale_2x_block(128, CONV_FILTER_2211, relu)
            stack.concat(stage_2_concat, axis=3)
            stack.conv_block(128, CONV_FILTER_3311, relu)
            stack.conv_block(128, CONV_FILTER_3311, relu)

            # 128 * 128
            stack.upscale_2x_block(64, CONV_FILTER_2211, relu)
            stack.concat(stage_1_concat, axis=3)
            stack.conv_block(64, CONV_FILTER_3311, relu)
            stack.conv_block(64, CONV_FILTER_3311, relu)
            stack.conv_block(1, CONV_FILTER_3311, relu)

            h = stack.last_layer
        return h

    def pixel_wise_softmax(self, output_map, name='pixel_wise_softmax', reuse=False):
        with tf.variable_scope(name, reuse=reuse):
            max_axis = tf.reduce_max(output_map, axis=3, keepdims=True)
            exponential_map = tf.exp(output_map - max_axis)
            normalize = tf.reduce_sum(exponential_map, axis=3, keepdims=True)
            return exponential_map / normalize

    def _build_main_graph(self):
        self.Xs = tf.placeholder(tf.float32, self.Xs_shape, name='Xs')
        self.Ys = tf.placeholder(tf.float32, self.Ys_shape, name='Ys')
        self.lr_ph = tf.placeholder(tf.float32, 1, name='lr_ph')

        self._proba = self.Unet_recursion(self.Xs, level=self.stage)
        # self.h = self.Unet_hard_coding(self.Xs)

        prediction = self.pixel_wise_softmax(self._proba)

        # For inference/visualization, prediction is argmax across output 'channels'
        # prediction = tf.argmax(tf.reshape(tf.nn.softmax(logits), tf.shape(vgg.up)), dimension=3)

        # For inference/visualization iou
        # valid_prediction = tf.reshape(logits, tf.shape(vgg.up))

        self._predict_ops = prediction
        self.Unet_vars = collect_vars(join_scope(get_scope(), 'generator'))

    def _build_loss_function(self):
        # TODO

        label = None
        logit = None
        self.loss = self._build_loss(self.loss, label, logit)

    def _build_train_ops(self):
        self._train_ops = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
            .minimize(self.loss, var_list=self.Unet_vars)

    @property
    def train_ops(self):
        return self._train_ops

    @property
    def predict_ops(self):
        return self._predict_ops

    @property
    def predict_proba_ops(self):
        return self._proba

    @property
    def score_ops(self):
        return self.metric_ops

    @property
    def metric_ops(self):
        return self.loss

    def train(self, Xs, Ys, epoch=1, save_interval=None, batch_size=None):
        self._prepare_train(Xs=Xs, Ys=Ys)
        dataset = self.to_dummyDataset(Xs=Xs, Ys=Ys)

        if batch_size is None:
            batch_size = self.batch_size

        iter_num = 0
        iter_per_epoch = dataset.size
        for e in trange(epoch):
            for i in range(iter_per_epoch):
                iter_num += 1

                Xs, Ys = dataset.next_batch(batch_size, batch_keys=['Xs', 'Ys'])
                self.sess.run(self.train_ops, feed_dict={self._Xs: Xs, self._Ys: Ys})

            Xs, Ys = dataset.next_batch(batch_size, batch_keys=['Xs', 'Ys'], update_cursor=False)
            loss = self.sess.run(self.metric_ops, feed_dict={self._Xs: Xs, self._Ys: Ys})
            self.log.info(f"e:{e}, i:{iter_num} loss : {loss}")

            if save_interval is not None and e % save_interval == 0:
                self.save()
