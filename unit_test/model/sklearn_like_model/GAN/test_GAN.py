from script.data_handler.DatasetPackLoader import DatasetPackLoader
from script.model.sklearn_like_model.GAN.GAN import GAN


class Test_GAN:
    class_ = GAN

    def test_mnist(self):
        class_ = self.class_
        data_pack = DatasetPackLoader().load_dataset("MNIST")
        dataset = data_pack['train']
        Xs, Ys = dataset.full_batch(['Xs', 'Ys'])
        sample_X = Xs[:2]
        sample_Y = Ys[:2]

        model = class_(dataset.input_shapes)
        model.build()

        model.train(Xs, epoch=1)

        metric = model.metric(sample_X)
        print(metric)

        gen = model.generate(size=2)
        print(gen)

        path = model.save()

        model = class_()
        model.load(path)
        print('model reloaded')

        metric = model.metric(sample_X)
        print(metric)

        gen = model.generate(size=2)
        print(gen)

    def test_titanic(self):
        class_ = self.class_
        data_pack = DatasetPackLoader().load_dataset("titanic")
        dataset = data_pack['train']
        Xs, Ys = dataset.full_batch(['Xs', 'Ys'])
        sample_X = Xs[:2]
        sample_Y = Ys[:2]

        model = class_(dataset.input_shapes)
        model.build()

        model.train(Xs, epoch=1)

        metric = model.metric(sample_X)
        print(metric)

        gen = model.generate(size=2)
        print(gen)

        path = model.save()

        model = class_()
        model.load(path)
        print('model reloaded')

        metric = model.metric(sample_X)
        print(metric)

        gen = model.generate(size=2)
        print(gen)
