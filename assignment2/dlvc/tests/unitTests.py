from dlvc.datasets.pets import PetsDataset
from dlvc.datasets.pets import TEST_SIZE, VALIDATION_SIZE, TRAINING_SIZE
from dlvc.dataset import Subset
import numpy as np
from dlvc.batches import BatchGenerator
from dlvc.ops import vectorize, chain, type_cast
import cv2

'''
All small and quick tests go here.
'''

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

# Make sure that the color channels are in BGR order (not RGB) by displaying the images and verifying the colors are correct (cv2.imshow, cv2.imwrite)
# -> you should see a dog on a blue blanket here
sample = dataset_training[1337]
cv2.imwrite('dog_on_a_blue_blanket.jpg', sample.data)
#cv2.imshow("image", sample.data)
#cv2.waitKey()

##########################################
#                 PART 2                 #
##########################################

# The number of training batches is 1 if the batch size is set to the number of samples in the dataset
batch_generator = BatchGenerator(dataset_training, len(dataset_training), False)
num_of_batches = len(batch_generator)
expected = 1
assert num_of_batches == expected, "Number of batches is " + str(num_of_batches) + ", expected: " + str(expected)

# The number of training batches is 16 if the batch size is set to 500
batch_generator = BatchGenerator(dataset_training, 500, False)
num_of_batches = len(batch_generator)
expected = 16
assert num_of_batches == expected, "Number of batches is " + str(num_of_batches) + ", expected: " + str(expected)
# and the last batch has size 459
batch_idx = 0
for batch in batch_generator:
    batch_idx += 1
    if batch_idx == 16:
        assert len(batch.label) == 459, "Num of samples in the last batch is: " + str(len(batch.label)) + ", expected: 459"

# The data and label shapes are (500, 3072) and (500,), respectively, unless for the last batch
batch_generator = BatchGenerator(dataset_training, 500, shuffle=False, op=vectorize())
last_batch_idx = len(batch_generator) - 1
batch_idx = 0
for batch in batch_generator:
    # skip last batch
    if batch_idx == last_batch_idx:
        continue
    assert batch.data.shape == (500, 3072), "Batch data shape: " + str(batch.data.shape) + ", expected: (500, 3072)."
    assert batch.label.shape == (500,), "Batch labels shape: " + str(batch.label.shape) + ", expected: (500,)."
    batch_idx += 1

# The data type is always np.float32 and the label type is integral (one of the np.int and np.uint variants)
# Implemented: for label type np.uint8 since there is less than 256 labels
batch_generator = BatchGenerator(dataset_training, 500, False, op=chain([vectorize(), type_cast(dtype=np.float32)]))
for batch in batch_generator:
    assert batch.data.dtype == np.float32, "Batch data type: " + str(batch.data.dtype) + ", expected: np.float32."
    assert batch.label.dtype == np.uint8, "Batch labels type: " + str(batch.label.dtype) + ", expected: np.uint8."

# The first sample of the first training batch returned without shuffling
# has label 0 ...
batch_generator = BatchGenerator(dataset_training, 500, False, op=chain([type_cast(dtype=np.float32), vectorize()]))
first_sample_label_unshuffled = None
first_sample_data_unshuffled = None
expected_label = 0
for batch in batch_generator:
    for label in batch.label:
        first_sample_label_unshuffled = label
        break
    for data in batch.data:
        first_sample_data_unshuffled = data
        break
    break
assert first_sample_label_unshuffled == expected_label, "First sample label: " + str(first_sample_label_unshuffled)\
                                             + ", expected: " + str(expected_label)
# ... and data [116. 125. 125. 91. 101. ...]
treshold = 10e-6
assert abs(first_sample_data_unshuffled[0] - 116.) < treshold, "Pixel value not correct"
assert abs(first_sample_data_unshuffled[1] - 125.) < treshold, "Pixel value not correct"
assert abs(first_sample_data_unshuffled[2] - 125.) < treshold, "Pixel value not correct"
assert abs(first_sample_data_unshuffled[3] - 91.) < treshold, "Pixel value not correct"
assert abs(first_sample_data_unshuffled[4] - 101.) < treshold, "Pixel value not correct"

# The first sample of the first training batch returned with shuffling must be different than first in unshuffled batch
batch_generator = BatchGenerator(dataset_training, 500, True)
first_sample_data_shuffled = None
expected_label = 0
for batch in batch_generator:
    for data in batch.data:
        first_sample_data_shuffled = data
        break
    break
assert np.array_equal(first_sample_data_unshuffled, first_sample_data_shuffled) is False,\
    "Data is not shuffled - unexpected behaviour"

print("All unit tests are passing! Congratulations! :)")