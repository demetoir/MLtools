from script.data_handler.MNIST import MNIST
from script.model.sklearn_like_model.ImageClf import ImageClf


def test_ImageClf():
    # test on Mnist

    data_pack = MNIST()
    data_pack.load('./data/MNIST')
    train_set = data_pack['train']
    train_x, train_y = train_set.full_batch()
    test_x, test_y = data_pack['test'].full_batch()

    x, y = train_x[:100], train_y[:100]
    print(x.shape)
    print(y.shape)
    model = ImageClf()
    for i in range(10):
        model.train(x, y, epoch=1, batch_size=10)
        score = model.score(x, y)
        print(score)
        print(model.predict(x[:1]))
        print(model.predict_proba(x[:1]))
