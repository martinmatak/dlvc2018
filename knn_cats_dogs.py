from dlvc.datasets.pets import PetsDataset
from dlvc.dataset import Subset
from dlvc.models.knn import KnnClassifier
from dlvc.batches import BatchGenerator
from dlvc.ops import vectorize, chain, type_cast
from dlvc.test import Accuracy
import numpy as np

dir = '/Users/mmatak/dev/college/DLVC/cifar-10/cifar-10-batches-py/'

IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32
NUM_CHANNELS = 3

NUM_CLASSES = 2

pets_training = PetsDataset(dir, Subset.TRAINING)
pets_validation = PetsDataset(dir, Subset.VALIDATION)
pets_test = PetsDataset(dir, Subset.TEST)

batchGenerator_training = BatchGenerator(pets_training, len(pets_training), False,
                                         op=chain([type_cast(dtype=np.float32), vectorize()]))
batchGenerator_validation = BatchGenerator(pets_validation, len(pets_validation), False,
                                         op=chain([type_cast(dtype=np.float32), vectorize()]))
batchGenerator_test = BatchGenerator(pets_test, len(pets_test), False,
                                         op=chain([type_cast(dtype=np.float32), vectorize()]))

best_accuracy = Accuracy()
best_k = -1
results = {}
knn = None

for k in range(1, 100, 40):  # grid search example
    knn = KnnClassifier(k, IMAGE_HEIGHT*IMAGE_WIDTH*NUM_CHANNELS, NUM_CLASSES)
    accuracy = Accuracy()

    # train and compute validation accuracy ...
    for batch in batchGenerator_training:
        knn.train(batch.data, batch.labels)

    predictions = None
    validation_batch = None
    for batch in batchGenerator_validation:
        predictions = knn.predict(batch.data)
        validation_batch = batch

    accuracy.update(predictions, validation_batch.labels)
    results[k] = accuracy.accuracy()
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_k = k

for key, value in results.items():
    print("k=" + str(key), ", accuracy=", str(value))

# compute test accuracy
test_batch = None
test_batch_predictions = None
accuracy = Accuracy()
for batch in batchGenerator_test:
    test_batch = batch
    test_batch_predictions = knn.predict(batch.data)
accuracy.update(test_batch_predictions, test_batch.labels)
print(str(accuracy))
