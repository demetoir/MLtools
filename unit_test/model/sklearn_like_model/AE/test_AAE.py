from script.data_handler.DatasetPackLoader import DatasetPackLoader
from script.model.sklearn_like_model.AE.AAE import AAE


class test_AAE:
    class_ = AAE

    def __init__(self):
        self.test_mnist()
        self.test_titanic()

    def test_mnist(self):
        class_ = self.class_
        dataset = DatasetPackLoader().load_dataset("MNIST")
        dataset = dataset.train_set
        Xs, Ys = dataset.full_batch(['Xs', 'Ys'])
        sample_X = Xs[:2]
        sample_Y = Ys[:2]

        model = class_(dataset.input_shapes)
        model.build()

        model.train(Xs, Ys, epoch=1)

        code = model.code(sample_X)
        print("code {code}".format(code=code))

        recon = model.recon(sample_X, sample_Y)
        print("recon {recon}".format(recon=recon))

        loss = model.metric(sample_X, sample_Y)
        print("loss {}".format(loss))

        # generate(self, zs, Ys)

        proba = model.proba(sample_X)
        print("proba {}".format(proba))

        predict = model.predict(sample_X)
        print("predict {}".format(predict))

        score = model.score(sample_X, sample_Y)
        print("score {}".format(score))

        path = model.save()

        model = class_()
        model.load(path)
        print('model reloaded')

        code = model.code(sample_X)
        print("code {code}".format(code=code))

        recon = model.recon(sample_X, sample_Y)
        print("recon {recon}".format(recon=recon))

        loss = model.metric(sample_X, sample_Y)
        print("loss {}".format(loss))

        # generate(self, zs, Ys)

        proba = model.proba(sample_X)
        print("proba {}".format(proba))

        predict = model.predict(sample_X)
        print("predict {}".format(predict))

        score = model.score(sample_X, sample_Y)
        print("score {}".format(score))

    def test_titanic(self):
        class_ = self.class_
        dataset = DatasetPackLoader().load_dataset("titanic")
        dataset = dataset.train_set
        Xs, Ys = dataset.full_batch(['Xs', 'Ys'])
        sample_X = Xs[:2]
        sample_Y = Ys[:2]

        model = class_(dataset.input_shapes)
        model.build()

        model.train(Xs, Ys, epoch=1)

        code = model.code(sample_X)
        print("code {code}".format(code=code))

        recon = model.recon(sample_X, sample_Y)
        print("recon {recon}".format(recon=recon))

        loss = model.metric(sample_X, sample_Y)
        print("loss {:}".format(loss))

        # generate(self, zs, Ys)

        proba = model.proba(sample_X)
        print("proba {}".format(proba))

        predict = model.predict(sample_X)
        print("predict {}".format(predict))

        score = model.score(sample_X, sample_Y)
        print("score {}".format(score))

        path = model.save()

        model = class_()
        model.load(path)
        print('model reloaded')

        code = model.code(sample_X)
        print("code {code}".format(code=code))

        recon = model.recon(sample_X, sample_Y)
        print("recon {recon}".format(recon=recon))

        loss = model.metric(sample_X, sample_Y)
        print("loss {:}".format(loss))

        # generate(self, zs, Ys)

        proba = model.proba(sample_X)
        print("proba {}".format(proba))

        predict = model.predict(sample_X)
        print("predict {}".format(predict))

        score = model.score(sample_X, sample_Y)
        print("score {}".format(score))
