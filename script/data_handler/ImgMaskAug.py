import multiprocessing as mp
import time
import imgaug as ia
from queue import Empty as QueueEmpty, Full as QueueFull
from script.data_handler.Base.BaseDataset import BaseDataset
from script.util.MixIn import LoggerMixIn
from script.util.misc_util import error_trace
from script.workbench.NpSharedObj import NpSharedObj


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

            shared_images = NpSharedObj.from_np(self.images)
            shared_masks = NpSharedObj.from_np(self.masks)
            worker = mp.Process(
                target=ImgMaskAug._iter_augment,
                args=(
                    None,
                    shared_images.encode(),
                    shared_masks.encode(),
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

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.terminate()
        return True

    def __del__(self):
        self.terminate()

    def _iter_augment(self, images, masks, batch_size, aug_seq, hook_func, queue, join_signal, finish_signal):
        try:
            images = NpSharedObj.decode(images).np
            masks = NpSharedObj.decode(masks).np
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
                        print(f'queue is full, need to reduce n_jobs')
                        pass

                if join_signal.is_set():
                    break
        except BaseException as e:
            print(error_trace(e))
        finally:
            finish_signal.set()

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

    def get_batch(self):
        while True:
            try:
                image, mask = self.q.get(timeout=0.01)
                break
            except QueueEmpty:
                print(f'queue is empt, bottle neck occur, need to raise n_jobs')
                pass

        return image, mask
