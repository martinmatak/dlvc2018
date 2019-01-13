from dlvc.datasets.pets import PetsDataset
from dlvc.dataset import Subset
from dlvc.batches import BatchGenerator
from dlvc.ops import chain, type_cast, hwc2chw, mul, add, rcrop, hflip, vflip, rotate90, resize
from dlvc.models.pytorch import CnnClassifier
from dlvc.test import Accuracy
from torchvision import models
import torch
import torch.nn as nn
import numpy as np
import cv2

dir = '/home/e1227507/datasets/cifar-10-batches-py/'
# dir = '/Users/mmatak/dev/college/DLVC/cifar-10/cifar-10-batches-py/'
# dir = '/home/khaftool/PycharmProjects/Thesis/data/cifar-10-batches-py/'

IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32
NUM_CHANNELS = 3

BATCH_SIZE = 128
NUM_CLASSES = 2
EPOCHS = 1000
lr = 0.001
wd = 0.0000001

EARLY_STOPPING = True
EARLY_STOPPING_NUM_OF_EPOCHS = 100
USE_DROPOUT = True

USE_TRANSFER_LEARNING = True
FREEZE_CNN_PARAMETERS = True

pets_training = PetsDataset(dir, Subset.TRAINING)
pets_validation = PetsDataset(dir, Subset.VALIDATION)


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


def initialize_transfer_learning_model(model_name, num_classes, freeze_cnn_parameters):
    model_ft = None
    input_size = 0

    if model_name == "resnet":  # Resnet18
        model_ft = models.resnet18(pretrained=True)
        set_parameter(model_ft, freeze_cnn_parameters)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        # Alexnet
        model_ft = models.vgg11_bn(pretrained=True)
        set_parameter(model_ft, freeze_cnn_parameters)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    else:
        raise ValueError("Invalid model name")

    return model_ft, input_size


def set_parameter(model, freeze_parameters):
    if freeze_parameters:
        for param in model.parameters():
            param.requires_grad = False


if USE_TRANSFER_LEARNING:
    # there are two networks to use in transfer learning "resnet" and "alexnet"
    net = initialize_transfer_learning_model("resnet", NUM_CLASSES, FREEZE_CNN_PARAMETERS)
    net, input_size = net
    pad_mode_for_resizing = 'constant'
    op_chain = chain([type_cast(dtype=np.float32),
                      add(-127.5),
                      mul(1 / 127.5),
                      rcrop(25, 2, 'median'),
                      resize(input_size, pad_mode_for_resizing),
                      hwc2chw()])
else:
    net = CatDogNet()
    op_chain = chain([type_cast(dtype=np.float32), add(-127.5), mul(1 / 127.5), rcrop(25, 2, 'median'), hwc2chw()])

batchGenerator_training = BatchGenerator(pets_training, BATCH_SIZE, shuffle=True, op=op_chain)
batchGenerator_validation = BatchGenerator(pets_validation, BATCH_SIZE, shuffle=False, op=op_chain)

clf = CnnClassifier(net, (BATCH_SIZE, NUM_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH), NUM_CLASSES, lr, wd)
loss_list = []
best_accuracy = 0.0
accuracy = Accuracy()
epochs_since_best_accuracy = 0
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
        print("best val accuracy: " + str(best_accuracy))
