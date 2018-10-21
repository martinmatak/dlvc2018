
from .dataset import Dataset
import numpy as np
import types

IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32
CHANNEL_NUM = 3


class Batch:
    '''
    A (mini)batch generated by the batch generator.
    '''

    def __init__(self):
        '''
        Ctor.
        '''

        self.data = None
        self.labels = None
        self.idx = None


class BatchGenerator:
    '''
    Batch generator.
    Returned batches have the following properties:
      data: numpy array holding batch data of shape (s, SHAPE_OF_DATASET_SAMPLES).
      labels: numpy array holding batch labels of shape (s, SHAPE_OF_DATASET_LABELS).
      idx: numpy array with shape (s,) encoding the indices of each sample in the original dataset.
    '''

    def __init__(self, dataset: Dataset, num: int, shuffle: bool, op: types.FunctionType = None):
        '''
        Ctor.
        Dataset is the dataset to iterate over.
        num is the number of samples per batch. the number in the last batch might be smaller than that.
        shuffle controls whether the sample order should be preserved or not.
        op is a function to apply to input samples.
        Raises TypeError on invalid argument types.
        Raises ValueError on invalid argument values, such as if num is > len(dataset).
        '''

        # dataset field
        if isinstance(dataset, Dataset):
            self.dataset = dataset
        else:
            raise TypeError("dataset has an inappropriate type.")

        # num field
        if isinstance(num, int):
            if num > len(dataset):
                raise ValueError("Number may not be grater than length of a dataset.")
            else:
                self.batch_size = num
        else:
            raise TypeError("num has an inappropriate type.")

        # shuffle field
        if isinstance(shuffle, bool):
            self.shuffle = shuffle
        else:
            raise TypeError("shuffle has an inappropriate type.")

        # op field
        if op is None:
            self.op = op
        elif isinstance(op, types.FunctionType):
            self.op = op
        else:
            raise TypeError("op has an inappropriate type.")

        # initialize indices
        if shuffle:
            self.indices = np.random.permutation(len(dataset))
        else:
            self.indices = np.arange(len(dataset))

    def __len__(self) -> int:
        '''
        Returns the number of batches generated per iteration.
        '''

        return int(np.floor(len(self.dataset) / self.batch_size))

    def __iter__(self) -> types.GeneratorType:
        '''
        Iterate over the wrapped dataset, returning the data as batches.
        '''

        data = np.zeros((self.batch_size, IMAGE_WIDTH, IMAGE_HEIGHT, CHANNEL_NUM), dtype=np.uint8)
        labels = np.zeros((self.batch_size, 1), dtype=np.int32)
        indices = np.zeros(self.batch_size, dtype=np.int32)
        for idx in  range(0, len(self)):
            sample_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
            for i, sample_id in enumerate(sample_indices):
                _, sample = self.dataset.__getitem__(sample_id)
                if self.op:
                    data[i] = self.op(sample['data'])
                else:
                    data[i] = sample['data']
                labels[i] = sample['label']
                indices[i] = sample['idx']
            batch = Batch()
            batch.data = data
            batch.labels = labels
            batch.idx = indices
            yield batch

