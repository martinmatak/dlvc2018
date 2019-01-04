from dlvc.datasets.pets import PetsDataset
from dlvc.dataset import Subset
from dlvc.models.knn import KnnClassifier
import numpy as np
from dlvc.batches import BatchGenerator
from dlvc.ops import vectorize, chain, type_cast
import time

'''
Tests in this file should test the whole pipeline. They take more time than unit tests.
'''

# make sure the whole pipeline works:
#  when k=1 and
#  training and predict subset are equal and
#  batch size is equal to whole dataset then
#  kNN must have accuracy 100%

start = time.time()

pets = PetsDataset('/Users/mmatak/dev/college/DLVC/cifar-10/cifar-10-batches-py/', Subset.TEST)
num_classes = 2
k = 1
knn = KnnClassifier(k, 32*32*3, num_classes)
batchGenerator = BatchGenerator(pets, len(pets), False, op=chain([type_cast(dtype=np.float32), vectorize()]))

groundTruthLabels = None
for batch in batchGenerator:
    knn.train(batch.data, batch.label)
    groundTruthLabels = batch.label

predictedLabels = None

def measure_accuracy(predictedLabels: np.ndarray, groundTruthLabels: np.ndarray):
    correct = 0
    for index, trueLabel in enumerate(groundTruthLabels):
        predictedLabel = np.argmax(predictedLabels[index])
        if predictedLabel == trueLabel:
            correct += 1
    return correct * 1.0 / len(groundTruthLabels)

accuracy = 0
treshold = 10e-6
for batch in batchGenerator:
    predictedLabels = knn.predict(batch.data)
    accuracy = measure_accuracy(predictedLabels, groundTruthLabels)
assert abs(accuracy - 1.0) < treshold, "Accuracy is " + str(accuracy) + " expected: 1.0"

end = time.time()
####################################################
print("If this line gets executed, all end-to-end tests passed successfully :)")
print("total time needed: " + str(end-start) + " seconds")
#####################################################

