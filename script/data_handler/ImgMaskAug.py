import multiprocessing as mp
import time
import imgaug as ia
from queue import Empty as QueueEmpty, Full as QueueFull
from script.data_handler.Base.BaseDataset import BaseDataset
from script.util.MixIn import LoggerMixIn


class ActivatorMask:
    def __init__(self, exception_list):
        self.exception_list = exception_list

    def __call__(self, images, augmenter, parents, default):
        if self.exception_list is not None and augmenter.name in self.exception_list:
            return False
        else:
            return default


class ImgMaskAug(LoggerMixIn):
    def __init__(self, images, masks, aug_seq, activator, n_batch, n_jobs=1, q_size=50, verbose=0):
        super().__init__(verbose)

        self.images = images
        self.masks = masks
        self.activator = activator
        self.hook_func = ia.HooksImages(activator=self.activator)
        self.n_batch = n_batch
        self.aug_seq = aug_seq
        self.n_jobs = n_jobs
        self.q_size = q_size

        self.manager = mp.Manager()
        self.q = self.manager.Queue(maxsize=self.q_size)
        self.join_signal = mp.Event()
        self.pool = None
        self.workers = []
        self.finished_signals = []

        for _ in range(n_jobs):
            finished_signal = mp.Event()
            self.finished_signals += [finished_signal]

            worker = mp.Process(
                target=ImgMaskAug._iter_augment,
                args=(
                    None,
                    self.images,
                    self.masks,
                    self.n_batch,
                    self.aug_seq,
                    self.hook_func,
                    self.q,
                    self.join_signal,
                    finished_signal
                ),
            )
            worker.daemon = True
            worker.start()
            self.log.info(f'start augmentation worker {_ + 1}/{n_jobs}')

            self.workers += [worker]

    def terminate(self):
        self.join_signal.set()
        time.sleep(0.01)

        while True:
            try:
                self.q.get(timeout=0.1)
            except QueueEmpty:
                break

        for worker in self.workers:
            worker.terminate()
            worker.join()

    def _iter_augment(self, images, masks, batch_size, aug_seq, hook_func, queue, join_signal, finish_signal):
        try:
            dataset = BaseDataset(x=images, y=masks)

            while True:
                image, mask = dataset.next_batch(batch_size, balanced_class=False)
                seq_det = aug_seq.to_deterministic()
                image_aug = seq_det.augment_images(image)
                mask_aug = seq_det.augment_images(mask, hooks=hook_func)

                while True:
                    try:
                        queue.put((image_aug, mask_aug))
                        break
                    except QueueFull:
                        pass

                if join_signal.is_set():
                    break
        except BaseException as e:
            self.log.error(e)
        finally:
            finish_signal.set()

    def get_batch(self):
        while True:
            try:
                image, mask = self.q.get(timeout=0.01)
                break
            except QueueEmpty:
                pass

        return image, mask

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.terminate()
        return True

    def __del__(self):
        self.terminate()
