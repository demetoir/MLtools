from script.data_handler.DatasetPackLoader import DatasetPackLoader
from script.model.sklearn_like_model.MLPClassifier import MLPClassifier


class Test_MLPClassifier:
    def test_CLf(self):
        class_ = MLPClassifier
        data_pack = DatasetPackLoader().load_dataset("titanic")
        dataset = data_pack['train']
        Xs, Ys = dataset.full_batch(['Xs', 'Ys'])
        sample_X = Xs[:2]
        sample_Y = Ys[:2]

        model = class_()
        model.train(Xs, Ys)

        predict = model.predict(sample_X)
        print(predict)

        score = model.score(Xs, Ys)
        print(score)

        proba = model.predict_proba(sample_X)
        print(proba)

        metric = model.metric(sample_X, sample_Y)
        print(metric)

        path = model.save()

        model = class_()
        model.load(path)
        model.train(Xs, Ys)

        predict = model.predict(sample_X)
        print(predict)

        score = model.score(Xs, Ys)
        print(score)

        proba = model.predict_proba(sample_X)
        print(proba)

        metric = model.metric(sample_X, sample_Y)
        print(metric)

        model.save()
