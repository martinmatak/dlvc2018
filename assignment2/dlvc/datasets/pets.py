from ..dataset import Sample, Subset, ClassificationDataset
import os, cv2
import numpy as np
import _pickle as cpickle

TRAINING_SIZE = 7959
VALIDATION_SIZE = 2041
TEST_SIZE = 2000


class PetsDataset(ClassificationDataset):
    '''
    Dataset of cat and dog images from CIFAR-10 (class 0: cat, class 1: dog).
    '''

    def num_classes(self) -> int:
        return 2

    def __init__(self, fdir: str, subset: Subset):
        '''
        Loads a subset of the dataset from a directory fdir that contains the Python version
        of the CIFAR-10, i.e. files "data_batch_1", "test_batch" and so on.
        Raises ValueError if fdir is not a directory or if a file inside it is missing.

        The subsets are defined as follows:
          - The training set contains all cat and dog images from "data_batch_1" to "data_batch_4", in this order.
          - The validation set contains all cat and dog images from "data_batch_5".
          - The test set contains all cat and dog images from "test_batch".

        Images are loaded in the order the appear in the data files
        and returned as uint8 numpy arrays with shape 32*32*3, in BGR channel order.
        '''

        if isinstance(subset, Subset):
            self.subset = subset
        else:
            raise ValueError("Subset has inappropriate type")

        if isinstance(fdir, str):
            self.fdir = fdir
        else:
            raise ValueError("Subset has inappropriate type")

        if not os.path.isdir(self.fdir):
            raise ValueError("Address is not a directory")

        if not os.path.isfile(self.fdir + "data_batch_1"):
            raise ValueError("File data_batch_1 is missing")
        if not os.path.isfile(self.fdir + "data_batch_2"):
            raise ValueError("File data_batch_2 is missing")
        if not os.path.isfile(self.fdir + "data_batch_3"):
            raise ValueError("File data_batch_3 is missing")
        if not os.path.isfile(self.fdir + "data_batch_4"):
            raise ValueError("File data_batch_4 is missing")
        if not os.path.isfile(self.fdir + "data_batch_5"):
            raise ValueError("File data_batch_5 is missing")
        if not os.path.isfile(self.fdir + "test_batch"):
            raise ValueError("File test_batch is missing")

        if self.subset == Subset.TEST:
            self.test_images, self.test_labels = self.load_data_from_file(self.fdir + "test_batch")

        elif self.subset == Subset.VALIDATION:
            self.val_images, self.val_labels = self.load_data_from_file(self.fdir + "data_batch_5")

        elif self.subset == Subset.TRAINING:
            file_names = []
            for i in range(1, 5):
                file_names.append("data_batch_" + str(i))

            train_images = []
            train_labels = []

            for file in file_names:
                temp_images, temp_labels = self.load_data_from_file(self.fdir + str(file))
                train_images.extend(temp_images)
                train_labels.extend(temp_labels)

            self.train_images = np.array(train_images)
            self.train_labels = np.array(train_labels)

    def load_data_from_file(self, file_address):
        new_label = []
        new_image = []

        with open(file_address, 'rb') as fo:
            read_dict = cpickle.load(fo, encoding='bytes')
            for i, label in enumerate(read_dict[b'labels']):

                if label == 3:  # is a cat
                    img = read_dict[b'data'][i].reshape((3, 32, 32)).transpose((1, 2, 0))
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    new_image.append(img_bgr)
                    new_label.append(0)

                elif label == 5:  # is a dog
                    img = read_dict[b'data'][i].reshape((3, 32, 32)).transpose((1, 2, 0))
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    new_image.append(img_bgr)
                    new_label.append(1)

        return np.array(new_image), np.array(new_label)

    def __len__(self) -> int:
        '''
        Returns the number of samples in the dataset.
        '''
        if self.subset == Subset.TRAINING:
            return len(self.train_labels)

        elif self.subset == Subset.VALIDATION:
            return len(self.val_labels)

        elif self.subset == Subset.TEST:
            return len(self.test_labels)

    def __getitem__(self, idx: int) -> Sample:
        '''
        Returns the idx-th sample in the dataset.
        Raises IndexError if the index is out of bounds.
        '''
        if idx >= 0:
            if idx < TRAINING_SIZE and self.subset == Subset.TRAINING:
                s = Sample(idx, self.train_images[idx], self.train_labels[idx])

            elif idx < VALIDATION_SIZE and self.subset == Subset.VALIDATION:
                s = Sample(idx, self.val_images[idx], self.val_labels[idx])

            elif idx < TEST_SIZE and self.subset == Subset.TEST:
                s = Sample(idx, self.test_images[idx], self.test_labels[idx])
            else:
                raise IndexError("Index is out of bound")
        else:
            raise IndexError("Index cannot be negative")

        return s
