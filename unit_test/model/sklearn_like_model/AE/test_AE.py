from script.data_handler.DatasetPackLoader import DatasetPackLoader
from script.model.sklearn_like_model.AE.AE import AE

data_pack = DatasetPackLoader().load_dataset("MNIST")
dataset = data_pack['train']
Xs, Ys = dataset.full_batch(['Xs', 'Ys'])
sample_X = Xs[:2]
sample_Y = Ys[:2]


def AE_total_execute(model):
    model.train(Xs, epoch=1)

    metric = model.metric(sample_X)
    print(metric)

    code = model.code(sample_X)
    print(code)

    recon = model.recon(sample_X)
    print(recon)

    path = model.save()

    class_ = model.__class__
    del model

    model = class_().load(path)
    print('model reloaded')

    for i in range(2):
        model.train(Xs, epoch=1)

    metric = model.metric(sample_X)
    print(metric)

    metric = model.metric(sample_X)
    print(metric)

    code = model.code(sample_X)
    print(code)

    recon = model.recon(sample_X)
    print(recon)

    model.save()


def test_AE():
    model = AE()
    AE_total_execute(model)


def test_AE_with_noise():
    model = AE(with_noise=True)
    AE_total_execute(model)
