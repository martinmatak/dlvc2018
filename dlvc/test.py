from dlvc.datasets.pets import PetsDataset
from dlvc.datasets.pets import TEST_SIZE, VALIDATION_SIZE, TRAINING_SIZE
from dlvc.dataset import Subset
import numpy as np

dir = '/Users/mmatak/dev/college/DLVC/cifar-10/cifar-10-batches-py/'

##########################################
#                 PART 1                 #
##########################################


# Number of samples in the individual datasets: 7959 (training), 2041 (validation), 2000 (test)

dataset_test = PetsDataset(dir, Subset.TEST)
assert(len(dataset_test) == TEST_SIZE), "Number of elements in test_dataset is different than " % TEST_SIZE

dataset_training = PetsDataset(dir, Subset.TRAINING)
assert(len(dataset_training) == TRAINING_SIZE),\
    "Number of elements in training_dataset is different than " % TRAINING_SIZE

dataset_validation = PetsDataset(dir, Subset.VALIDATION)
assert(len(dataset_validation) == VALIDATION_SIZE), \
    "Number of elements in validation_dataset is different than " % VALIDATION_SIZE


# Total number of cat and dog samples: 6000 per class


def count_labels(dataset):
    cats = 0
    dogs = 0
    for index in range(0, len(dataset)):
        sample = dataset[index]
        if sample.label == 0:
            cats += 1
        elif sample.label == 1:
            dogs += 1
        else:
            ValueError("Invalid label")
    return cats, dogs


cats_test, dogs_test = count_labels(dataset_test)
cats_training, dogs_training = count_labels(dataset_training)
cats_validation, dogs_validation = count_labels(dataset_validation)
cats = cats_test + cats_training + cats_validation
dogs = dogs_test + dogs_training + dogs_validation
assert(cats == dogs), "Number of cats is not equal to number of dogs"
assert(cats == 6000), "Number of cats is different than 6000 (it's " + str(cats) + ")."
assert(dogs == 6000), "Number of dogs is different than 6000 (it's "+ str(dogs) + ")."


# Image shape: always (32, 32, 3)
def assert_shape(dataset, shape):
    for idx in range(0, len(dataset)):
        sample = dataset[idx]
        assert (sample.data.shape == shape), "Sample with index " + \
            str(idx) + " in dataset " + str(dataset) + " has inappropriate shape: " \
            + str(sample.data.shape) + ", expected: " + str(shape) + "."


target_shape = (32, 32, 3)
assert_shape(dataset_validation, target_shape)
assert_shape(dataset_training, target_shape)
assert_shape(dataset_test, target_shape)


# image type: always np.uint8
def assert_type(dataset, type):
    for index in range(0, len(dataset)):
        sample = dataset[index]
        assert (sample.data.dtype == type), "Sample with index " + \
                                           str(index) + " in dataset " + str(dataset) + " has inappropriate type: " \
                                           + str(sample.data.dtype) + ", expected: " + str(type) + "."


target_type = np.uint8
assert_type(dataset_validation, target_type)
assert_type(dataset_training, target_type)
assert_type(dataset_test, target_type)

# Labels of first 10 training samples: 0 0 0 0 1 0 0 0 0 1
labels = "0 0 0 0 1 0 0 0 0 1".split(" ")
for index, label in enumerate(labels):
    assert str(dataset_training[index].label) == label, "Label of index " + str(index) + " is not correct: it is " + \
        str(dataset_training[index].label) + ", expected: " + str(label) + "."

# TODO: Make sure that the color channels are in BGR order (not RGB) by displaying the images and verifying the colors are correct (cv2.imshow, cv2.imwrite)
# How to check this?

