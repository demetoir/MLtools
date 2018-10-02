from script.model.sklearn_like_model.BaseModel import BaseEpochCallback


class test_meta_epochCallback(BaseEpochCallback):
    def __call__(self, model, dataset, metric, epoch):
        print(metric)


def test_meta_callback():
    callback = test_meta_epochCallback()
    dc = {}
    key = 'key'
    dc[key] = 'meta_key'

    for i in range(10):
        callback(i, i, i, i)

    callback.trace_on(dc, key)
    for i in range(10):
        callback(i, i, i, i)

    callback.trace_off()
    for i in range(10):
        callback(i, i, i, i)

    callback.trace_on(dc, key)
    for i in range(10):
        callback(i, i, i, i)