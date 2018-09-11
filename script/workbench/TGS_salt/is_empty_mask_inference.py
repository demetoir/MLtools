import tensorflow as tf
from pprint import pprint
from script.model.sklearn_like_model.BaseModel import BaseEpochCallback
from script.model.sklearn_like_model.ImageClf import ImageClf
from script.model.sklearn_like_model.TFSummary import TFSummaryParams, TFSummaryScalar
from script.model.sklearn_like_model.Top_k_save import Top_k_save
from script.util.misc_util import time_stamp, path_join
from script.util.numpy_utils import *
from script.workbench.TGS_salt.TGS_salt_inference import data_helper, plot, to_dict


class cnn_EpochCallback(BaseEpochCallback):
    def __init__(self, model, train_x, train_y, test_x, test_y, params):
        super().__init__()
        self.train_x = train_x
        self.train_y = train_y
        self.model = model
        self.test_x = test_x
        self.test_y = test_y
        self.params = params

        self.k = 5
        self.top_k = [np.Inf for _ in range(self.k)]

        self.run_id = self.params['run_id']
        run_id = self.run_id
        self.base_path = path_join(f'./tf_summary/TGS_salt/empty_mask_clf', run_id)
        self.top_k_save = Top_k_save(path_join(self.base_path, 'top_k'))
        self.summary_train_loss = TFSummaryScalar(path_join(self.base_path, 'train'), 'train_loss')
        self.summary_train_acc = TFSummaryScalar(path_join(self.base_path, 'train'), 'train_acc')
        self.summary_test_acc = TFSummaryScalar(path_join(self.base_path, 'test'), 'test_acc')

    def log_score(self, epoch, log):
        from sklearn.metrics import confusion_matrix
        from sklearn.metrics import accuracy_score
        self.train_loss = self.model.metric(self.train_x, self.train_y)

        test_predict = self.model.predict(self.test_x)
        test_y_decode = np_onehot_to_index(self.test_y)
        self.test_score = accuracy_score(test_y_decode, test_predict)
        test_confusion = confusion_matrix(test_y_decode, test_predict)

        train_predict = self.model.predict(self.train_x)
        train_y_decode = np_onehot_to_index(self.train_y)
        self.train_score = accuracy_score(train_y_decode, train_predict)
        train_confusion = confusion_matrix(train_y_decode, train_predict)

        log(f'e={epoch}, '
            f'train_score = {self.train_score},\n'
            f'train_confusion = {train_confusion},\n'
            f'test_score = {self.test_score},\n'
            f'test_confusion ={test_confusion}\n')

    def update_summary(self, sess, epoch):
        self.summary_train_loss.update(sess, self.train_loss, epoch)
        self.summary_train_acc.update(sess, self.train_score, epoch)
        self.summary_test_acc.update(sess, self.test_score, epoch)

    def __call__(self, sess, dataset, epoch, log=None):
        self.log_score(epoch, log)
        self.update_summary(sess, epoch)
        self.top_k_save(self.test_score, self.model)


class is_emtpy_mask_clf_pipeline:
    def __init__(self):
        self.data_helper = data_helper()
        self.plot = plot
        self.data_helper.train_set.y_keys = ['empty_mask']

        self.init_dataset()
        self.base_path = f'./tf_summary/TGS_salt/empty_mask_clf'

    def init_dataset(self):
        train_set = self.data_helper.train_set
        train_x, train_y = train_set.full_batch()
        train_x = train_x.reshape([-1, 101, 101, 1])
        train_y = train_y.reshape([-1, 1])
        from sklearn.preprocessing import OneHotEncoder
        enc = OneHotEncoder()
        enc.fit(train_y)

        from sklearn.model_selection import train_test_split
        train_x, test_x, train_y, test_y = train_test_split(
            train_x, train_y, test_size=0.33)
        self.enc = enc
        train_y_onehot = enc.transform(train_y).toarray()
        test_y_onehot = enc.transform(test_y).toarray()

        print(np.mean(train_y))

        sample_size = 100
        sample_x = train_x[:sample_size]
        sample_y = train_y[:sample_size]
        sample_y_onehot = train_y_onehot[:sample_size]

        self.train_set = train_set
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.test_y_onehot = test_y_onehot
        self.train_y_onehot = train_y_onehot
        self.sample_x = sample_x
        self.sample_y = sample_y
        self.sample_y_onehot = sample_y_onehot

    def params(self, run_id=None, learning_rate=0.01, learning_rate_decay_rate=0.99,
               learning_rate_decay_method=None, beta1=0.9, batch_size=128, net_type='InceptionV1',
               n_classes=None, capacity=4, ):
        # net_type = 'InceptionV1'
        net_type = 'InceptionV2'
        # net_type = 'InceptionV4'
        # net_type = 'ResNet18'
        # net_type = 'ResNet34'
        # net_type = 'ResNet50'
        # net_type = 'ResNet101'
        # net_type = 'ResNet152'

        if run_id is None:
            run_id = time_stamp()

        return to_dict(
            run_id=run_id,
            batch_size=batch_size,
            net_type=net_type,
            capacity=capacity,
            learning_rate=learning_rate,
            beta1=beta1,
            n_classes=2,
        )

    @staticmethod
    def save_tf_summary_params(params):
        with tf.Session() as sess:
            run_id = params['run_id']
            summary_params = TFSummaryParams(f'./tf_summary/TGS_salt/empty_mask_clf/{run_id}', 'params')
            summary_params.update(sess, params)

    def train(self, params, n_epoch, augmentation=False, early_stop=True, patience=20):
        self.save_tf_summary_params(params)
        pprint(f'train {params}')

        clf = ImageClf(**params)
        epoch_callback = cnn_EpochCallback(
            clf,
            self.train_x, self.train_y_onehot,
            self.test_x, self.test_y_onehot,
            params,
        )

        clf.train(self.train_x, self.train_y_onehot, epoch=n_epoch, epoch_callback=epoch_callback,
                  iter_pbar=True, dataset_callback=None, early_stop=early_stop, patience=patience)

        score = clf.score(self.sample_x, self.sample_y_onehot)
        test_score = clf.score(self.test_x, self.test_y_onehot)
        print(f'score = {score}, test= {test_score}')
