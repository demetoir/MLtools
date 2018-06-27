import numpy as np
from script.data_handler.DatasetPackLoader import DatasetPackLoader
from script.model.sklearn_like_model.AE.CVAE import CVAE


class_ = CVAE

def test_mnist():
    class_ = CVAE
    data_pack = DatasetPackLoader().load_dataset("MNIST")
    dataset = data_pack['train']
    Xs, Ys = dataset.full_batch(['Xs', 'Ys'])
    sample_X = Xs[:2]
    sample_Y = Ys[:2]

    model = class_(dataset.input_shapes)
    model._build()
    model.train(Xs, Ys, epoch=1)

    code = model.code(sample_X, sample_Y)
    print("code {code}".format(code=code))

    recon = model.recon(sample_X, sample_Y)
    print("recon {recon}".format(recon=recon))

    loss = model.metric(sample_X, sample_Y)
    loss = np.mean(loss)
    print("loss {:.4}".format(loss))

    path = model.save()

    model = class_()
    model.load(path)
    print('model reloaded')

    code = model.code(sample_X, sample_Y)
    print("code {code}".format(code=code))

    recon = model.recon(sample_X, sample_Y)
    print("recon {recon}".format(recon=recon))

    loss = model.metric(sample_X, sample_Y)
    loss = np.mean(loss)
    print("loss {:.4}".format(loss))

