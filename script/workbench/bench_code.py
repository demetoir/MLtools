# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd

from script.data_handler.TGS_salt import mask_label_encoder
from script.util.Logger import pprint_logger, Logger
from script.util.MixIn import LoggerMixIn
from script.util.PlotTools import PlotTools
from script.util.deco import deco_timeit
# print(built-in function) is not good for logging
from script.util.misc_util import path_join
from script.workbench.TGS_salt.SS_Inference import SemanticSegmentation_pipeline
from script.workbench.TGS_salt.TGS_salt_inference import masks_rate, TGS_salt_DataHelper, Metrics
from script.workbench.TGS_salt.is_empty_mask_inference import is_emtpy_mask_clf_pipeline
from slackbot.SlackBot import deco_slackbot

bprint = print
logger = Logger('bench_code', level='INFO', )
print = logger.info
pprint = pprint_logger(print)
NpArr = np.array
DF = pd.DataFrame
Series = pd.Series
plot = PlotTools(save=True, show=False)


def SS_test():
    pipe = SemanticSegmentation_pipeline()
    param = pipe.params(
        stage=4,
        net_type='FusionNet',
        batch_size=32,
        dropout_rate=0.5,
        learning_rate=0.01,
        depth=2,
        loss_type='BCE+dice_soft',
        comment='change dropout location'
    )
    print(param)
    pipe.new_model(param)
    pipe.train(n_epoch=100, augmentation=False)


class SS_baseline:
    def train_baseline(self):
        pipe = self.load_baseline()
        pipe.train(n_epoch=100, augmentation=False)

    def load_baseline(self):
        path = './instance/TGS_salt/SS/baseline'
        pipe = SemanticSegmentation_pipeline()
        set_dict = data_dict().upper_1p_with_depth()
        pipe.set_dict(set_dict)
        pipe.load_model(path)

        return pipe

    def train_target(self):

        path = './instance/TGS_salt/SS/target'
        "./instance/TGS_salt/SS/2018-10-04_02-02-55/test_score/top_1"
        pipe = SemanticSegmentation_pipeline()
        set_dict = data_dict().upper_1p_with_depth()
        pipe.set_dict(set_dict)
        pipe.load_model(path)
        pipe.train(n_epoch=100, augmentation=False)

        return pipe

    @staticmethod
    def scramble_column(*args, size=10):
        ret = []
        for i in range(0, len(args[0]), size):
            for j in range(len(args)):
                ret += [args[j][i:i + size]]

        return np.concatenate(ret, axis=0)

    def run(self):

        pipe = self.load_baseline()
        baseline = pipe.model

        train_predict = baseline.predict(pipe.train_x)
        train_predict = mask_label_encoder.from_label(train_predict)

        test_predict = baseline.predict(pipe.valid_x)
        test_predict = mask_label_encoder.from_label(test_predict)

        train_loss = baseline.metric(pipe.train_x, pipe.train_y_encode)
        test_loss = baseline.metric(pipe.valid_x, pipe.valid_y_encode)

        train_proba = baseline.predict_proba(pipe.train_x)
        test_proba = baseline.predict_proba(pipe.valid_x)

        train_score = Metrics.TGS_salt_score(pipe.train_y, train_predict)
        test_score = Metrics.TGS_salt_score(pipe.valid_y, test_predict)

        train_mask_rate = masks_rate(pipe.train_y)
        test_mask_rate = masks_rate(pipe.valid_y)

        train_miou = Metrics.miou(pipe.train_y, train_predict)
        test_miou = Metrics.miou(pipe.valid_y, test_predict)

        train_ious = Metrics.ious(pipe.train_y, train_predict)
        test_ious = Metrics.ious(pipe.valid_y, test_predict)

        print(
            f'train TGS score = {train_score}\n'
            f'test TGS score = {test_score}\n'
            f'train miou = {train_miou}\n'
            f'test miou = {test_miou}\n'
            f'train loss = {train_loss}\n'
            f'test loss = {test_loss}\n'
        )

    def log_score(self):

        pipe = self.load_baseline()
        baseline = pipe.model

        train_predict = baseline.predict(pipe.train_x)
        train_predict = mask_label_encoder.from_label(train_predict)

        test_predict = baseline.predict(pipe.valid_x)
        test_predict = mask_label_encoder.from_label(test_predict)

        train_loss = baseline.metric(pipe.train_x, pipe.train_y_encode)
        test_loss = baseline.metric(pipe.valid_x, pipe.valid_y_encode)

        train_proba = baseline.predict_proba(pipe.train_x)
        test_proba = baseline.predict_proba(pipe.valid_x)

        train_score = Metrics.TGS_salt_score(pipe.train_y, train_predict)
        test_score = Metrics.TGS_salt_score(pipe.valid_y, test_predict)

        train_mask_rate = masks_rate(pipe.train_y)
        test_mask_rate = masks_rate(pipe.valid_y)

        train_miou = Metrics.miou(pipe.train_y, train_predict)
        test_miou = Metrics.miou(pipe.valid_y, test_predict)

        train_ious = Metrics.ious(pipe.train_y, train_predict)
        test_ious = Metrics.ious(pipe.valid_y, test_predict)

        print(
            f'train TGS score = {train_score}\n'
            f'test TGS score = {test_score}\n'
            f'train miou = {train_miou}\n'
            f'test miou = {test_miou}\n'
            f'train loss = {train_loss}\n'
            f'test loss = {test_loss}\n'
        )

    def plot_test_set_sorted_by_iou(self):
        pipe = self.load_baseline()
        baseline = pipe.model

        train_predict = baseline.predict(pipe.train_x)
        train_predict = mask_label_encoder.from_label(train_predict)

        test_predict = baseline.predict(pipe.valid_x)
        test_predict = mask_label_encoder.from_label(test_predict)

        train_loss = baseline.metric(pipe.train_x, pipe.train_y_encode)
        test_loss = baseline.metric(pipe.valid_x, pipe.valid_y_encode)

        train_proba = baseline.predict_proba(pipe.train_x)
        test_proba = baseline.predict_proba(pipe.valid_x)

        train_score = Metrics.TGS_salt_score(pipe.train_y, train_predict)
        test_score = Metrics.TGS_salt_score(pipe.valid_y, test_predict)

        train_mask_rate = masks_rate(pipe.train_y)
        test_mask_rate = masks_rate(pipe.valid_y)

        train_miou = Metrics.miou(pipe.train_y, train_predict)
        test_miou = Metrics.miou(pipe.valid_y, test_predict)

        train_ious = Metrics.ious(pipe.train_y, train_predict)
        test_ious = Metrics.ious(pipe.valid_y, test_predict)

        test_x = pipe.valid_x
        test_y = pipe.valid_y
        test_predict = test_predict
        test_mask_rate = test_mask_rate

        zipped = zip(test_x, test_y, test_predict, test_ious)
        sort = list(sorted(zipped, key=lambda a: a[3]))
        test_x, test_y, test_predict, test_ious = zip(*sort)
        print(test_ious)
        test_x = np.array(test_x)
        test_y = np.array(test_y)
        test_predict = np.array(test_predict)
        test_x_image = test_x[:, :, :, 0]
        test_x_image = test_x_image.reshape([-1, 101, 101, 1])
        test_y = test_y.reshape([-1, 101, 101, 1])
        test_predict = test_predict.reshape([-1, 101, 101, 1])

        tile_image = self.scramble_column(test_x_image, test_y, test_predict)
        plot.plot_image_tile(tile_image, title='all', path=f'./matplot/baseline/test_set_sorted_by_iou.png')

    def test_set_sorted_by_mask_rate(self):
        pipe = self.load_baseline()
        baseline = pipe.model

        train_predict = baseline.predict(pipe.train_x)
        train_predict = mask_label_encoder.from_label(train_predict)

        test_predict = baseline.predict(pipe.valid_x)
        test_predict = mask_label_encoder.from_label(test_predict)

        train_loss = baseline.metric(pipe.train_x, pipe.train_y_encode)
        test_loss = baseline.metric(pipe.valid_x, pipe.valid_y_encode)

        train_proba = baseline.predict_proba(pipe.train_x)
        test_proba = baseline.predict_proba(pipe.valid_x)

        train_score = Metrics.TGS_salt_score(pipe.train_y, train_predict)
        test_score = Metrics.TGS_salt_score(pipe.valid_y, test_predict)

        train_mask_rate = masks_rate(pipe.train_y)
        test_mask_rate = masks_rate(pipe.valid_y)

        train_miou = Metrics.miou(pipe.train_y, train_predict)
        test_miou = Metrics.miou(pipe.valid_y, test_predict)

        train_ious = Metrics.ious(pipe.train_y, train_predict)
        test_ious = Metrics.ious(pipe.valid_y, test_predict)

        test_x = pipe.valid_x
        test_y = pipe.valid_y
        test_predict = test_predict
        test_mask_rate = test_mask_rate

        zipped = zip(test_x, test_y, test_predict, test_mask_rate)
        sort = list(sorted(zipped, key=lambda a: a[3]))
        test_x, test_y, test_predict, test_mask_rate = zip(*sort)
        test_x = np.array(test_x)
        test_y = np.array(test_y)
        test_predict = np.array(test_predict)
        test_mask_rate = np.array(test_mask_rate)
        test_x_image = test_x[:, :, :, 0]
        test_x_image = test_x_image.reshape([-1, 101, 101, 1])
        test_y = test_y.reshape([-1, 101, 101, 1])
        test_predict = test_predict.reshape([-1, 101, 101, 1])

        # plot_path = f'./matplot/baseline'
        tile_image = self.scramble_column(test_x_image, test_y, test_predict)
        plot.plot_image_tile(tile_image, title='all', path=f'./matplot/baseline/test_set_sorted_by_mask_rate.png')

    def snapshot_var(self):
        class snapshot_variable(LoggerMixIn):
            def __init__(self, model, scope=None, verbose=0):
                super().__init__(verbose=verbose)
                self.model = model
                self.scope = scope

            def snapshot(self, x, y, path):
                for var in self.model.main_graph_var_list:
                    var_path = path_join(path, var.name + '.png')
                    var_path = var_path.replace(':', '_')

                sess = self.model.sess
                ops = sess.graph.get_operations()
                for op in ops[:50]:
                    print(op)

        pipe = self.load_baseline()
        baseline = pipe.model
        snapshot = snapshot_variable(baseline)
        snapshot.snapshot(None, None, './snapshot')

    def log_split_mask_rate(self):

        pipe = self.load_baseline()
        baseline = pipe.model

        helper = pipe.data_helper
        train_set = helper.train_set_non_empty_mask_with_depth_image_under_1p
        train_set.x_keys = ['x_with_depth']
        x, y = train_set.full_batch()
        predict = baseline.predict(x)
        score_under_1 = Metrics.TGS_salt_score(y, mask_label_encoder.from_label(predict))
        iou_score_under_1 = Metrics.miou(y, mask_label_encoder.from_label(predict))

        train_set = helper.train_set_non_empty_mask_with_depth_image_under_5p
        train_set.x_keys = ['x_with_depth']
        x, y = train_set.full_batch()
        predict = baseline.predict(x)
        score_under_5 = Metrics.TGS_salt_score(y, mask_label_encoder.from_label(predict))
        iou_score_under_5 = Metrics.miou(y, mask_label_encoder.from_label(predict))

        train_set = helper.train_set_non_empty_mask_with_depth_image_under_10p
        train_set.x_keys = ['x_with_depth']
        x, y = train_set.full_batch()
        predict = baseline.predict(x)
        score_under_10 = Metrics.TGS_salt_score(y, mask_label_encoder.from_label(predict))
        iou_score_under_10 = Metrics.miou(y, mask_label_encoder.from_label(predict))

        train_set = helper.train_set_non_empty_mask_with_depth_image_under_20p
        train_set.x_keys = ['x_with_depth']
        x, y = train_set.full_batch()
        predict = baseline.predict(x)
        score_under_20 = Metrics.TGS_salt_score(y, mask_label_encoder.from_label(predict))
        iou_score_under_20 = Metrics.miou(y, mask_label_encoder.from_label(predict))

        train_set = helper.train_set_non_empty_mask_with_depth_image_upper_1p
        train_set.x_keys = ['x_with_depth']
        x, y = train_set.full_batch()
        predict = baseline.predict(x)
        score_upper_1 = Metrics.TGS_salt_score(y, mask_label_encoder.from_label(predict))
        iou_score_upper_1 = Metrics.miou(y, mask_label_encoder.from_label(predict))

        train_set = helper.train_set_non_empty_mask_with_depth_image_upper_5p
        train_set.x_keys = ['x_with_depth']
        x, y = train_set.full_batch()
        predict = baseline.predict(x)
        score_upper_5 = Metrics.TGS_salt_score(y, mask_label_encoder.from_label(predict))
        iou_score_upper_5 = Metrics.miou(y, mask_label_encoder.from_label(predict))

        train_set = helper.train_set_non_empty_mask_with_depth_image_upper_10p
        train_set.x_keys = ['x_with_depth']
        x, y = train_set.full_batch()
        predict = baseline.predict(x)
        score_upper_10 = Metrics.TGS_salt_score(y, mask_label_encoder.from_label(predict))
        iou_score_upper_10 = Metrics.miou(y, mask_label_encoder.from_label(predict))

        train_set = helper.train_set_non_empty_mask_with_depth_image_upper_20p
        train_set.x_keys = ['x_with_depth']
        x, y = train_set.full_batch()
        predict = baseline.predict(x)
        score_upper_20 = Metrics.TGS_salt_score(y, mask_label_encoder.from_label(predict))
        iou_score_upper_20 = Metrics.miou(y, mask_label_encoder.from_label(predict))

        print(
            f'\n'
            f' TGS score\n'
            f'under 1 {score_under_1}\n'
            f'under 5 {score_under_5}\n'
            f'under 10 {score_under_10}\n'
            f'under 20 {score_under_20}\n'
            f'upper 1 {score_upper_1}\n'
            f'upper 5 {score_upper_5}\n'
            f'upper 10 {score_upper_10}\n'
            f'upper 20 {score_upper_20}\n'

            f'\n'
            f'\n'
            f' iou score\n'
            f'under 1 {iou_score_under_1}\n'
            f'under 5 {iou_score_under_5}\n'
            f'under 10 {iou_score_under_10}\n'
            f'under 20 {iou_score_under_20}\n'
            f'upper 1 {iou_score_upper_1}\n'
            f'upper 5 {iou_score_upper_5}\n'
            f'upper 10 {iou_score_upper_10}\n'
            f'upper 20 {iou_score_upper_20}\n'
            f'\n'
        )


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


class data_dict:
    @staticmethod
    def split_10P_set(random_sate=1234):
        data_helper = TGS_salt_DataHelper()

        train_set = data_helper.train_set_with_depth_image
        train_set.y_keys = ['mask_rate']

        train_x, train_y = train_set.full_batch()

        y = np.zeros([len(train_y)])
        y[train_y >= 0.1] = 0
        y[0 < train_y < 0.1] = 1
        y[train_y == 0] = 2
        train_y = y
        train_y = train_y.reshape([-1, 1])

        from sklearn.model_selection import train_test_split
        train_x, test_x, train_y, test_y = train_test_split(
            train_x, train_y, test_size=0.33, random_state=random_sate)

        from sklearn.preprocessing import OneHotEncoder
        encoder = OneHotEncoder()
        encoder.fit(train_y)
        train_y_onehot = encoder.transform(train_y).toarray()
        test_y_onehot = encoder.transform(test_y).toarray()

        return {
            'train_set': train_set,
            'train_x': train_x,
            'train_y': train_y,
            'test_x': test_x,
            'test_y': test_y,
            'test_y_onehot': test_y_onehot,
            'train_y_onehot': train_y_onehot,
            'encoder': encoder
        }

    @staticmethod
    def upper_1p_with_depth(random_sate=1234):
        data_helper = TGS_salt_DataHelper()
        train_set = data_helper.train_set_non_empty_mask_with_depth_image
        # train_set = data_helper.train_set_non_empty_mask_with_depth_image
        train_set.x_keys = ['x_with_depth']

        train_set, holdout_set = train_set.split((9, 1), shuffle=True, random_state=random_sate)

        kfold_sets = train_set.k_fold_split(5)

        fold_1_train_set, fold_1_test_set = kfold_sets[0]
        fold_1_train_x, fold_1_train_y = fold_1_train_set.full_batch()
        fold_1_test_x, fold_1_test_y = fold_1_test_set.full_batch()
        holdout_x, holdout_y = holdout_set.full_batch()

        return {
            'train_set': train_set,
            'holdout_set': holdout_set,
            'kfold_sets ': kfold_sets,

            'train_x': fold_1_train_x,
            'train_y': fold_1_train_y,
            'valid_x': fold_1_test_x,
            'valid_y': fold_1_test_y,

            'holdout_x': holdout_x,
            'holdout_y': holdout_y,
        }


class post_processing:

    def fill_hole(self):
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
        import numpy as np
        import matplotlib.pyplot as plt
        import skimage.morphology, skimage.data

        img = skimage.data.imread('j1ESv.png', 1)
        labels = skimage.morphology.label(img)
        labelCount = np.bincount(labels.ravel())
        background = np.argmax(labelCount)
        img[labels != background] = 255
        plt.imshow(img, cmap=plt.cm.gray)
        plt.show()


@deco_timeit
@deco_slackbot('./slackbot/tokens/ml_bot_token', 'mltool_bot')
def main():
    data_dict().upper_1p_with_depth()

    SS_baseline().train_baseline()

    # set_dict = data_dict().split_10P_set()
