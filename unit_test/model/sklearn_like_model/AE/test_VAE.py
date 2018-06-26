from script.data_handler.DatasetPackLoader import DatasetPackLoader
from script.model.sklearn_like_model.AE.VAE import VAE
import numpy as np


class Test_VAE:
    def test_mnist(self):
        dataset = DatasetPackLoader().load_dataset("MNIST")
        dataset = dataset['train']

        model = VAE(dataset.input_shapes)
        model._build()

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

        model = VAE()
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
        dataset = DatasetPackLoader().load_dataset("titanic")
        dataset = dataset['train']

        model = VAE(dataset.input_shapes)
        model._build()

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

        model = VAE()
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
