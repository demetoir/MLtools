from script.data_handler.DatasetPackLoader import DatasetPackLoader
from script.model.sklearn_like_model.MLPClassifier import MLPClassifier


class test_MLPClassifier:
    def __init__(self):
        self.test()

    def test(self):
        dataset = DatasetPackLoader().load_dataset("titanic")
        input_shapes = dataset.train_set.input_shapes

        Xs, Ys = dataset.train_set.full_batch(
            batch_keys=["Xs", "Ys"],
        )

        model = MLPClassifier(input_shapes)
        model.build()
        model.train(Xs, Ys, epoch=1)

        Xs, Ys = dataset.train_set.next_batch(
            5,
            batch_keys=["Xs", "Ys"],
        )

        predict = model.predict(Xs)
        print("predict {}".format(predict))

        loss = model.metric(Xs, Ys)
        print("loss {}".format(loss))

        proba = model.proba(Xs)
        print('prob {}'.format(proba))

        score = model.score(Xs, Ys)
        print('score {}'.format(score))

        path = model.save()
        model = MLPClassifier()
        model.load(path)

        predict = model.predict(Xs)
        print("predict {}".format(predict))

        loss = model.metric(Xs, Ys)
        print("loss {}".format(loss))

        proba = model.proba(Xs)
        print('prob {}'.format(proba))

        score = model.score(Xs, Ys)
        print('score {}'.format(score))
