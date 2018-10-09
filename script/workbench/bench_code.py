# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd

from script.util.Logger import pprint_logger, Logger
from script.util.PlotTools import PlotTools
from script.util.deco import deco_timeit
from script.util.misc_util import load_pickle
from script.workbench.TGS_salt.SS_Inference import SS_baseline
from script.workbench.TGS_salt.TGS_salt_inference import Metrics
from script.workbench.TGS_salt.U_net_with_simple_Resnet_Blocks_v2 import train_kernel_main
from script.workbench.TGS_salt.is_empty_mask_inference import is_emtpy_mask_clf_pipeline
from script.workbench.TGS_salt.post_process_AE import post_process_AE
from slackbot.SlackBot import deco_slackbot

bprint = print
logger = Logger('bench_code', level='INFO', )
print = logger.info
pprint = pprint_logger(print)
NpArr = np.array
DF = pd.DataFrame
Series = pd.Series
plot = PlotTools(save=True, show=False)


class is_empty_mask_baseline:
    def top_baseline(self):
        # top base line but suspicious 0.94 test
        path = f'./instance/TGS_salt/empty_mask_clf/inceptionv1'
        pipe = is_emtpy_mask_clf_pipeline()
        params = pipe.params(
            capacity=4,
            net_type='InceptionV1',
            batch_size=64,
            learning_rate=0.01,
            dropout_rate=0.5,
            fc_capacity=1024,
            fc_depth=2,
            comment='inception v1',
        )
        pipe.train(params, n_epoch=100, augmentation=False, path=path)

    def inceptionv1(self):
        # top base line but suspicious 0.94 test
        path = f'./instance/TGS_salt/empty_mask_clf/inceptionv1'
        pipe = is_emtpy_mask_clf_pipeline()
        params = pipe.params(
            capacity=4,
            net_type='InceptionV1',
            batch_size=64,
            learning_rate=0.01,
            dropout_rate=0.5,
            fc_capacity=1024,
            fc_depth=2,
            comment='inception v1',
        )
        pipe.train(params, n_epoch=100, augmentation=False, path=path)

    def inceptionv2(self):
        path = f'./instance/TGS_salt/empty_mask_clf/inceptionv2'
        pipe = is_emtpy_mask_clf_pipeline()
        params = pipe.params(
            capacity=4,
            net_type='InceptionV2',
            # net_type='ResNet18',
            batch_size=32,
            learning_rate=0.01,
            dropout_rate=0.5,
            fc_capacity=1024,
            fc_depth=2,
            comment='inception v1',
        )
        pipe.train(params, n_epoch=100, augmentation=False, path=path)

    def Resnet34(self):
        path = f'./instance/TGS_salt/empty_mask_clf/resnet34'
        pipe = is_emtpy_mask_clf_pipeline()
        params = pipe.params(
            capacity=4,
            net_type='ResNet34',
            batch_size=32,
            learning_rate=0.01,
            dropout_rate=0.5,
            fc_capacity=1024,
            fc_depth=2,
            comment='inception v1',
        )
        pipe.train(params, n_epoch=100, augmentation=False, path=path)

    def Resnet50(self):
        path = f'./instance/TGS_salt/empty_mask_clf/resnet50'
        pipe = is_emtpy_mask_clf_pipeline()
        params = pipe.params(
            capacity=4,
            net_type='ResNet50',
            batch_size=32,
            learning_rate=0.01,
            dropout_rate=0.5,
            fc_capacity=1024,
            fc_depth=2,
            comment='inception v1',
        )
        pipe.train(params, n_epoch=100, augmentation=False, path=path)


class post_processing:

    def fill_hole(self):
        # suck way
        import cv2
        import numpy as np

        # Read image
        im_in = cv2.imread("nickel.jpg", cv2.IMREAD_GRAYSCALE)

        # Threshold.
        # Set values equal to or above 220 to 0.
        # Set values below 220 to 255.

        th, im_th = cv2.threshold(im_in, 220, 255, cv2.THRESH_BINARY_INV)

        # Copy the thresholded image.
        im_floodfill = im_th.copy()

        # Mask used to flood filling.
        # Notice the size needs to be 2 pixels than the image.
        h, w = im_th.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)

        # Floodfill from point (0, 0)
        cv2.floodFill(im_floodfill, mask, (0, 0), 255)

        # Invert floodfilled image
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)

        # Combine the two images to get the foreground.
        im_out = im_th | im_floodfill_inv

        # Display images.
        cv2.imshow("Thresholded Image", im_th)
        cv2.imshow("Floodfilled Image", im_floodfill)
        cv2.imshow("Inverted Floodfilled Image", im_floodfill_inv)
        cv2.imshow("Foreground", im_out)
        cv2.waitKey(0)

    def fill_hole2(self):
        # suck way
        train_predict, valid_predict = self.load_SS_predict()

        def fill_hole(xs):
            #
            def batch(x):
                import numpy as np
                import skimage.morphology, skimage.data

                labels = skimage.morphology.label(x)
                labelCount = np.bincount(labels.ravel())
                background = np.argmax(labelCount)
                x[labels != background] = 255
                return x

            return np.concatenate(
                [
                    batch(x)
                    for x in xs
                ],
                axis=0
            )

        a = fill_hole(train_predict)
        a = a.reshape([-1, 101, 101, 1])
        print(a.shape)
        print(a[:10])
        plot.plot_image_tile(a, path=f"./matplot/fill_hole_train.png")

    def AE(self):
        ae = post_process_AE(capacity=32, batch_size=32)
        pprint(ae)
        # set_dict = data_dict().post_AE_mask_only()
        set_dict = None
        train_x = set_dict['train_x']
        valid_x = set_dict['valid_x']
        # ae.load(f"./instance/post_process_AE")
        ae.build(x=train_x, y=train_x)

        # ae.restore(f"./instance/post_process_AE")

        def to_scale(x):
            return x / 255.

        def from_scale(x):
            return x * 255.

        train_x_scale = to_scale(train_x)
        valid_x_scale = to_scale(valid_x)

        train_metric = ae.metric(train_x_scale, train_x_scale)
        valid_metric = ae.metric(valid_x_scale, valid_x_scale)

        print(f'train metric = {train_metric}')
        print(f'test metric = {valid_metric}')

        # print(train_x_scale)
        for i in range(1):
            ae.train(train_x_scale, train_x_scale, epoch=1)
            train_metric = ae.metric(train_x_scale, train_x_scale)
            valid_metric = ae.metric(valid_x_scale, valid_x_scale)

            print(f'train metric = {train_metric}')
            print(f'test metric = {valid_metric}')

            recon = ae.recon(train_x_scale)
            recon = from_scale(recon)
            # print(recon)
            # recon = recon.reshape([-1, 101, 101, 1])
            plot.plot_image_tile(recon, path=f'./matplot/train/recon_{str(i*10)}.png')

            recon = ae.recon(valid_x_scale)
            recon = from_scale(recon)
            # print(recon)
            # recon = recon.reshape([-1, 101, 101, 1])
            plot.plot_image_tile(recon, path=f'./matplot/test/recon_{str(i*10)}.png')

        # ae.save(f"./instance/post_process_AE")

    @property
    def ae(self):
        if not getattr(self, '_ae', None):
            ae = post_process_AE(
                capacity=8, batch_size=64, learning_rate=0.01
            )

            ae.build(x=(101, 101, 1), y=(101, 101, 1))
            # ae.restore(f"./instance/post_process_AE")
            self._ae = ae

        return self._ae

    def post_processing(self, predict):
        ae = self.ae

        def to_scale(x):
            return x / 255.

        def from_scale(x):
            return x * 255.

        predict = to_scale(predict)
        pp_predict = ae.recon(predict)
        pp_predict = from_scale(pp_predict)
        return pp_predict

    def ae_score(self, x, y):

        def to_scale(x):
            return x / 255.

        def from_scale(x):
            return x * 255.

        x = to_scale(x)
        y = to_scale(y)
        x = self.ae.metric(x, y)
        return x

    @staticmethod
    def scramble_column(*args, size=10):
        ret = []
        for i in range(0, len(args[0]), size):
            for j in range(len(args)):
                ret += [args[j][i:i + size]]

        return np.concatenate(ret, axis=0)

    def log_post_process(self, y, predict, post_predict):

        before_tgs_score = Metrics.TGS_salt_score(y, predict)
        before_miou_score = Metrics.miou(y, predict)

        after_tgs_score = Metrics.TGS_salt_score(y, post_predict)
        after_miou_score = Metrics.miou(y, post_predict)

        print(
            f"before TGS score = {before_tgs_score}\n"
            f"before miou score = {before_miou_score}\n"
            f"after TGS score = {after_tgs_score}\n"
            f"after miou score = {after_miou_score}\n"
            f"diff = {after_tgs_score - before_tgs_score}\n"
            f"\n"
        )

    def load_SS_predict(self):
        # set_dict = data_dict().non_empty_with_depth()

        # train_x = set_dict['train_x']
        # train_y = set_dict['train_y']
        # valid_x = set_dict['valid_x']
        # valid_y = set_dict['valid_y']

        train_predict = load_pickle('./train_predict')
        valid_predict = load_pickle('./valid_predict')

        # SS_pipe = SS_baseline().load_baseline()
        # ss = SS_pipe.model
        # train_predict = ss.predict(train_x)
        # train_predict = mask_label_encoder.from_label(train_predict)
        # valid_predict = ss.predict(valid_x)
        # valid_predict = mask_label_encoder.from_label(valid_predict)
        # dump_pickle(train_predict, './train_predict')
        # dump_pickle(valid_predict, './valid_predict')
        # return

        # print(train_predict[0])
        # print(train_y[0])

        return train_predict, valid_predict

    def apply_pp(self):
        set_dict = None

        train_x = set_dict['train_x']
        train_y = set_dict['train_y']
        valid_x = set_dict['valid_x']
        valid_y = set_dict['valid_y']

        train_predict, valid_predict = self.load_SS_predict()

        for i in range(100):
            def to_scale(x):
                return x / 255.

            self.ae.train(to_scale(train_predict), to_scale(train_y))

            def normalize(x):
                threash_hold = 255 * 0.5
                new_x = np.zeros_like(x)
                new_x[x > threash_hold] = 255
                new_x[x <= threash_hold] = 0
                return new_x

            pp_train_predict = self.post_processing(to_scale(train_predict))
            pp_valid_predict = self.post_processing(to_scale(valid_predict))
            pre_normalize = pp_valid_predict
            pp_train_predict = normalize(pp_train_predict)
            pp_valid_predict = normalize(pp_valid_predict)

            # train_tile = self.scramble_column(train_y[:30], train_predict[:30], pp_train_predict[:30])
            valid_tile = self.scramble_column(
                valid_y[i * 30:i * 30 + 30],
                valid_predict[i * 30:i * 30 + 30],
                pp_valid_predict[i * 30:i * 30 + 30],
                pre_normalize[i * 30:i * 30 + 30])
            # plot.plot_image_tile(train_tile, path=f'./matplot/train_tile_{i}.png')
            plot.plot_image_tile(valid_tile, path=f'./matplot/valid_tile_{i}.png')

            print('train')
            self.log_post_process(train_y, train_predict, pp_train_predict)

            print('valid')
            self.log_post_process(valid_y, valid_predict, pp_valid_predict)

    def TTA(self, predict):
        pass


@deco_timeit
@deco_slackbot('./slackbot/tokens/ml_bot_token', 'mltool_bot')
def main():
    baseline = SS_baseline()
    baseline.new_model()
    baseline.train()
    #
    # train_kernel_main()
