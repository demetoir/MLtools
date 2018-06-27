from script.data_handler.DatasetPackLoader import DatasetPackLoader
from script.model.sklearn_like_model.GAN.GAN import GAN


def test_GAN():
    class_ = GAN
    data_pack = DatasetPackLoader().load_dataset("titanic")
    dataset = data_pack['train']
    Xs, Ys = dataset.full_batch(['Xs', 'Ys'])
    sample_X = Xs[:2]
    # sample_Y = Ys[:2]

    model = class_()
    model.train(Xs, epoch=1)

    metric = model.metric(sample_X)
    # print(metric)

    gen = model.generate(size=2)
    # print(gen)

    path = model.save()

    model = class_()
    model.load(path)
    # print('model reloaded')

    for i in range(1):
        model.train(Xs, epoch=1)

    metric = model.metric(sample_X)
    # print(metric)

    model.save()
