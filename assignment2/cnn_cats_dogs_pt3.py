from dlvc.datasets.pets import PetsDataset
from dlvc.dataset import Subset
from dlvc.batches import BatchGenerator
from dlvc.ops import chain, type_cast, hwc2chw, mul, add, rcrop, hflip, vflip, rotate90
from dlvc.models.pytorch import CnnClassifier
from dlvc.test import Accuracy

import torch
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
EPOCHS = 1000
lr = 0.001
wd = 0.00000001

EARLY_STOPPING = True
EARLY_STOPPING_NUM_OF_EPOCHS = 100
USE_DROPOUT = True

pets_training = PetsDataset(dir, Subset.TRAINING)
pets_validation = PetsDataset(dir, Subset.VALIDATION)

batchGenerator_training = BatchGenerator(pets_training, BATCH_SIZE, shuffle=True,
                                         op=chain([type_cast(dtype=np.float32),
                                                   add(-127.5),
                                                   mul(1 / 127.5),
                                                   rcrop(25, 2, 'median'),
                                                   hwc2chw()]))
batchGenerator_validation = BatchGenerator(pets_validation, BATCH_SIZE, shuffle=False,
                                         op=chain([type_cast(dtype=np.float32),
                                                   add(-127.5),
                                                   mul(1 / 127.5),
                                                   hwc2chw()]))

class CatDogNet(nn.Module):
    def __init__(self):
        super(CatDogNet, self).__init__()
        # First Layer 2xConv and Max pool out_Shape = (16x16x32)
        self.conv1_layer1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.batch_norm1_layer1 = nn.BatchNorm2d(num_features=32)
        self.relu1_layer1 = nn.ReLU()

        self.conv2_layer1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.batch_norm2_layer1 = nn.BatchNorm2d(num_features=32)
        self.relu2_layer1 = nn.ReLU()

        self.max_pool_layer1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # second Layer 2xConv and Max pool out_shape = (8x8x64)
        self.conv1_layer2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.batch_norm1_layer2 = nn.BatchNorm2d(num_features=64)
        self.relu1_layer2 = nn.ReLU()

        self.conv2_layer2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.batch_norm2_layer2 = nn.BatchNorm2d(num_features=64)
        self.relu2_layer2 = nn.ReLU()

        self.max_pool_layer2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Third Layer 2xConv and average pool out_shape = (4x4x128)
        self.conv1_layer3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.batch_norm1_layer3 = nn.BatchNorm2d(num_features=128)
        self.relu1_layer3 = nn.ReLU()

        self.conv2_layer3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.batch_norm2_layer3 = nn.BatchNorm2d(num_features=128)
        self.relu2_layer3 = nn.ReLU()

        self.avg_pool_layer3 = nn.AvgPool2d(kernel_size=2, stride=2)

        # Add all the units into the Sequential layer in exact order
        self.cnn_net = nn.Sequential(self.conv1_layer1,
                                     self.batch_norm1_layer1,
                                     self.relu1_layer1,
                                     self.conv2_layer1,
                                     self.batch_norm2_layer1,
                                     self.relu2_layer1,
                                     self.max_pool_layer1,

                                     self.conv1_layer2,
                                     self.batch_norm1_layer2,
                                     self.relu1_layer2,
                                     self.conv2_layer2,
                                     self.batch_norm2_layer2,
                                     self.relu2_layer2,
                                     self.max_pool_layer2,

                                     self.conv1_layer3,
                                     self.batch_norm1_layer3,
                                     self.relu1_layer3,
                                     self.conv2_layer3,
                                     self.batch_norm2_layer3,
                                     self.relu2_layer3,
                                     self.avg_pool_layer3)

        self.dropout_layer = nn.Dropout(0.5)

        self.fc = nn.Linear(in_features=2048, out_features=NUM_CLASSES)

    # override
    def forward(self, input):
        output = self.cnn_net(input)
        if USE_DROPOUT:
            output = self.dropout_layer(output)
        output = output.view(-1, 4 * 4 * 128)
        output = self.fc(output)
        return output


net = CatDogNet()
clf = CnnClassifier(net, (BATCH_SIZE, NUM_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH), NUM_CLASSES, lr, wd)
loss_list = []
best_accuracy = 0.0
accuracy = Accuracy()
epochs_since_best_accuracy = 0
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
    print("best val accuracy: " + str(best_accuracy))

    if EARLY_STOPPING:
        epochs_since_best_accuracy += 1
        if best_accuracy < accuracy.accuracy():
            best_accuracy = accuracy.accuracy()
            torch.save(net.state_dict(), "best_model.pth")
            epochs_since_best_accuracy = 0
        if epochs_since_best_accuracy > EARLY_STOPPING_NUM_OF_EPOCHS:
            print(str(EARLY_STOPPING_NUM_OF_EPOCHS) +
                  " epochs passed without improvement in validation accuracy. Stopping the training now." +
		  "best validation accuracy: " + str(best_accuracy))
            break
time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
