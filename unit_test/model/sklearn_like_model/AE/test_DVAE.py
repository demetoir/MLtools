from script.data_handler.DatasetPackLoader import DatasetPackLoader
from script.model.sklearn_like_model.AE.DVAE import DVAE
import numpy as np


class Test_DVAE:
    class_ = DVAE

    def test_mnist(self):
        class_ = self.class_
        dataset = DatasetPackLoader().load_dataset("MNIST")
        dataset = dataset['train']

        model = class_(dataset.input_shapes)
        model.build()

        Xs = dataset.full_batch(['Xs'])
        model.train(Xs, epoch=1)

        sample_X = Xs[:2]
        code = model.code(sample_X)
        print("code {code}".format(code=code))

        recon = model.recon(sample_X)
        print("recon {recon}".format(recon=recon))

        loss = model.metric(Xs)
        loss = np.mean(loss)
        print("loss {:.4}".format(loss))

        path = model.save()

        model = class_()
        model.load(path)
        print('model reloaded')

        sample_X = Xs[:2]
        code = model.code(sample_X)
        print("code {code}".format(code=code))

        recon = model.recon(sample_X)
        print("recon {recon}".format(recon=recon))

        loss = model.metric(Xs)
        loss = np.mean(loss)
        print("loss {:.4}".format(loss))

    def test_titanic(self):
        class_ = self.class_
        dataset = DatasetPackLoader().load_dataset("titanic")
        dataset = dataset['train']

        model = class_(dataset.input_shapes)
        model.build()

        Xs = dataset.full_batch(['Xs'])
        model.train(Xs, epoch=1)

        sample_X = Xs[:2]
        code = model.code(sample_X)
        print("code {code}".format(code=code))

        recon = model.recon(sample_X)
        print("recon {recon}".format(recon=recon))

        loss = model.metric(Xs)
        loss = np.mean(loss)
        print("loss {:.4}".format(loss))

        path = model.save()

        model = class_()
        model.load(path)
        print('model reloaded')

        sample_X = Xs[:2]
        code = model.code(sample_X)
        print("code {code}".format(code=code))

        recon = model.recon(sample_X)
        print("recon {recon}".format(recon=recon))

        loss = model.metric(Xs)
        loss = np.mean(loss)
        print("loss {:.4}".format(loss))
