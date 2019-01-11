from ..model import Model

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim


class CnnClassifier(Model):
    '''
    Wrapper around a PyTorch CNN for classification.
    The network must expect inputs of shape NCHW with N being a variable batch size,
    C being the number of (image) channels, H being the (image) height, and W being the (image) width.
    The network must end with a linear layer with num_classes units (no softmax).
    The cross-entropy loss and SGD are used for training.
    '''

    def __init__(self, net: nn.Module, input_shape: tuple, num_classes: int, lr: float, wd: float):
        '''
        Ctor.
        net is the cnn to wrap. see above comments for requirements.
        input_shape is the expected input shape, i.e. (0,C,H,W).
        num_classes is the number of classes (> 0).
        lr: learning rate to use for training (sgd with Nesterov momentum of 0.9).
        wd: weight decay to use for training.
        '''
        if not isinstance(net, nn.Module):
            raise TypeError("Input shape has an inappropriate type.")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = net.to(self.device)

        if not (isinstance(input_shape, tuple) and len(input_shape) == 4):
            raise TypeError("Input shape must be a tuple of length 4.")
        else:
            self.in_shape = input_shape

        if isinstance(num_classes, int):
            if num_classes <= 0:
                raise ValueError("Number of classes must not be negative")
            else:
                self.num_classes = num_classes
        else:
            raise TypeError("Number of classes has an inappropriate type")

        if not isinstance(lr, float):
            raise TypeError("Learning rate has an inappropriate type.")
        else:
            self.lr = lr

        if not isinstance(wd, float):
            raise TypeError("Weight decay has an inappropriate type.")
        else:
            self.wd = wd
        self.warning_is_showed_flag_training = False
        self.warning_is_showed_flag_predict = False
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.wd)

        self.loss_fn = nn.CrossEntropyLoss()
        # inside the train() and predict() functions you will need to know whether the network itself
        # runs on the cpu or on a gpu, and in the latter case transfer input/output tensors via cuda()
        # and cpu(). do determine this, check the type of (one of the) parameters, which can be obtained
        # via parameters()(there is an is_cuda flag). you will want to initialize the optimizer and
        # loss function here. note that pytorch's cross-entropy loss includes normalization so no
        # softmax is required

        pass

    def input_shape(self) -> tuple:
        '''
        Returns the expected input shape as a tuple.
        '''
        return self.in_shape

    def output_shape(self) -> tuple:
        '''
        Returns the shape of predictions for a single sample as a tuple, which is (num_classes,).
        '''
        return self.num_classes,

    def train(self, data: np.ndarray, labels: np.ndarray) -> float:
        '''
        Train the model on batch of data.
        Data has shape (m,C,H,W) and type np.float32 (m is arbitrary).
        Labels has shape (m,) and integral values between 0 and num_classes - 1.
        Returns the training loss.
        Raises TypeError on invalid argument types.
        Raises ValueError on invalid argument values.
        Raises RuntimeError on other errors.
        '''
        # Check data
        if not isinstance(data, np.ndarray):
            raise TypeError("Data has an inappropriate type")
        elif data.dtype != np.float32:
            raise TypeError("Data must have np.float32 type")
        elif len(data.shape) != 4:
            raise ValueError("Prediction must have shape (m,C,H,W)")
        else:
            data_tensor = torch.Tensor(data).to(self.device)

        # Check Label
        if not isinstance(labels, np.ndarray):
            raise TypeError("Label has an inappropriate type")
        elif not ((0 <= m < self.num_classes) for m in labels):
            raise ValueError("Labels has a inappropriate value")
        elif len(labels.shape) != 1:
            raise ValueError("Prediction must have shape (m,)")
        else:
            label_tensor = torch.Tensor(labels).long().to(self.device)

        # Check if network is running on GPU or CPU
        t = next(iter(self.model.parameters()))
        if not t.is_cuda and self.warning_is_showed_flag_training is False:
            print("WARNING: Program should use GPU for training. Currently running on CPU")
            self.warning_is_showed_flag_training = True

        self.model.train()
        # Clear all accumulated gradients
        self.optimizer.zero_grad()
        # Predict
        outputs = self.model(data_tensor)
        # Calculate the loss
        loss = self.loss_fn(outputs, label_tensor)
        # Back propagate the loss
        loss.backward()
        # Adjust parameters according to the computed gradients
        self.optimizer.step()

        return loss.item()

    def predict(self, data: np.ndarray) -> np.ndarray:
        '''
        Predict softmax class scores from input data.
        Data has shape (m,C,H,W) and type np.float32 (m is arbitrary).
        The scores are an array with shape (n, output_shape()).
        Raises TypeError on invalid argument types.
        Raises ValueError on invalid argument values.
        Raises RuntimeError on other errors.
        '''
        if not isinstance(data, np.ndarray):
            raise TypeError("Data has an inappropriate type")
        elif data.dtype != np.float32:
            raise TypeError("Data must have np.float32 type")
        elif len(data.shape) != 4:
            raise ValueError("Prediction must have shape (m,C,H,W)")
        else:
            data_tensor = torch.Tensor(data).to(self.device)

        # Check if network is running on GPU or CPU
        t = next(iter(self.model.parameters()))
        if not t.is_cuda and self.warning_is_showed_flag_predict is False:
            print("WARNING: Program should use GPU for predicting. Currently running on CPU")
            self.warning_is_showed_flag_predict = True

        self.model.eval()

        outputs = self.model(data_tensor)

        softmax = nn.Softmax(dim=1)
        predictions = softmax(outputs)

        return predictions