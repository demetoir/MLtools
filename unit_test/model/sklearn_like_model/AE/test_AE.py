import numpy as np
from script.data_handler.DatasetPackLoader import DatasetPackLoader
from script.model.sklearn_like_model.AE.AutoEncoder import AutoEncoder


class Test_AE:

    def test_mnist(self):
        data_pack = DatasetPackLoader().load_dataset("MNIST")
        dataset = data_pack['train']

        model = AutoEncoder(dataset.input_shapes)
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

        model = AutoEncoder()
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

        model = AutoEncoder(dataset.input_shapes)
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

        model = AutoEncoder()
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
