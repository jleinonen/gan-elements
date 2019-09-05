import numpy as np
from tensorflow.keras.datasets import mnist


class BatchGenerator(object):
    
    def __init__(self, data, batch_size=32, shuffle=True):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.N = self.data.shape[0]
        self.next_ind = np.array([], dtype=int)

    def __len__(self):
        return self.N

    def __iter__(self):
        return self

    def __next__(self):
        while len(self.next_ind) < self.batch_size:
            ind = np.arange(self.N, dtype=int)
            if self.shuffle:
                np.random.shuffle(ind)
            self.next_ind = np.concatenate([self.next_ind, ind])

        ind = self.next_ind[:self.batch_size]
        self.next_ind = self.next_ind[self.batch_size:]

        batch = self.data[ind,...]

        return batch


class MNISTBatchGenerator(BatchGenerator):
    def __init__(self, **kwargs):
        data = self.load_data()
        super(MNISTBatchGenerator, self).__init__(data, **kwargs)

    def load_data(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_all = np.concatenate((x_train, x_test))
        x = np.zeros((x_all.shape[0], 32, 32, 1), dtype=np.float32)
        x[:,2:-2,2:-2,0] = x_all
        x *= (2.0/255.0)
        x -= 1
        return x


class NoiseGenerator(object):
    def __init__(self, noise_shapes, batch_size=32, random_seed=None):
        self.noise_shapes = noise_shapes
        self.batch_size = batch_size
        self.prng = np.random.RandomState(seed=random_seed)

    def __iter__(self):
        return self

    def __next__(self, mean=0.0, std=1.0):

        def noise(shape):
            shape = (self.batch_size,) + shape

            n = self.prng.randn(*shape).astype(np.float32)
            if std != 1.0:
                n *= std
            if mean != 0.0:
                n += mean
            return n

        return [noise(s) for s in self.noise_shapes]
