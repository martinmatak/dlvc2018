from dlvc.datasets.pets import PetsDataset
from dlvc.dataset import Subset
from dlvc.batches import BatchGenerator
from dlvc.ops import chain, type_cast, hwc2chw, mul, add
from dlvc.models.pytorch import CnnClassifier
from dlvc.test import Accuracy
import torch.nn as nn
import numpy as np
import time

dir = '/home/e1227507/datasets/cifar-10-batches-py/'
# dir = '/home/khaftool/PycharmProjects/Thesis/data/cifar-10-batches-py/'
# dir = '/Users/mmatak/dev/college/DLVC/cifar-10/cifar-10-batches-py/'

IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32
NUM_CHANNELS = 3

BATCH_SIZE = 128
NUM_CLASSES = 2
EPOCHS = 500
lr = 0.001
# weight decay 0 in this configuration, in part 3 this is changed
wd = 0.0

pets_training = PetsDataset(dir, Subset.TRAINING)
pets_validation = PetsDataset(dir, Subset.VALIDATION)
pets_test = PetsDataset(dir, Subset.TEST)


batchGenerator_training = BatchGenerator(pets_training, BATCH_SIZE, shuffle=True,
                                         op=chain([type_cast(dtype=np.float32),
                                                   add(-127.5),
                                                   mul(1 / 127.5),
                                                   hwc2chw()]))
batchGenerator_validation = BatchGenerator(pets_validation, BATCH_SIZE, shuffle=False,
                                         op=chain([type_cast(dtype=np.float32),
                                                   add(-127.5),
                                                   mul(1 / 127.5),
                                                   hwc2chw()]))
batchGenerator_test = BatchGenerator(pets_test, BATCH_SIZE, shuffle=False,
                                         op=chain([type_cast(dtype=np.float32),
                                                   add(-127.5),
                                                   mul(1 / 127.5),
                                                   hwc2chw()]))


class CatDogNet(nn.Module):
    def __init__(self):
        super(CatDogNet, self).__init__()
        self.linear_layer1 = nn.Linear(in_features=3072, out_features=1536)
        self.batch_norm_layer1 = nn.BatchNorm1d(num_features=1536)
        self.linear_layer3 = nn.Linear(in_features=1536, out_features=NUM_CLASSES)

    # override
    def forward(self, input):
        input = input.view(-1, 32 * 32 * 3)
        output = self.linear_layer1(input)
        output = self.batch_norm_layer1(output)
        output = self.linear_layer3(output)
        return output


net = CatDogNet()
clf = CnnClassifier(net, (BATCH_SIZE, NUM_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH), NUM_CLASSES, lr, wd)
loss_list = []
accuracy = Accuracy()
since = time.time()
for epoch in range(0, EPOCHS):
    print("Epoche: ", epoch + 1)

    for batch in batchGenerator_training:
        loss = clf.train(batch.data, batch.label)
        loss_list.append(loss)

    loss = np.array(loss_list)
    loss_mean = np.mean(loss)
    loss_deviation = np.std(loss)
    print("Train loss: ", loss_mean, "-+", loss_deviation)

    accuracy.reset()
    for batch in batchGenerator_validation:
        predictions = clf.predict(batch.data)
        accuracy.update(predictions.cpu().detach().numpy(), batch.label)

    print("Val " + str(accuracy))
time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))