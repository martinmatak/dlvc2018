
from ..model import Model

import numpy as np

class KnnClassifier(Model):
    '''
    k nearest neighbors classifier.
    Returns softmax class scores (see lecture slides).
    '''

    def __init__(self, k: int, input_dim: int, num_classes: int):
        '''
        Ctor.
        k is the number of nearest neighbors to consult (>= 1).
        input_dim is the length of input vectors (> 0).
        num_classes is the number of classes (> 1).
        '''

        # check k
        if isinstance(k, int):
            if k >= 1:
                self.k = k
            else:
                raise ValueError("K must be greater or equal to 1.")
        else:
            raise TypeError("K must be of type int.")

        # check input_dim
        if isinstance(input_dim, int):
            if input_dim > 0:
                self.input_dim = input_dim
            else:
                raise ValueError("Input dimension must be greater than 0.")
        else:
            raise TypeError("Input dimension must be of type int.")

        # check num_classes
        if isinstance(num_classes, int):
            if num_classes > 1:
                self.num_classes = num_classes
            else:
                raise ValueError("Number of classes must be greater than 1.")
        else:
            raise TypeError("Number of classes must be of type int.")

    def input_shape(self) -> tuple:
        '''
        Returns the expected input shape as a tuple, which is (0, input_dim).
        '''

        return 0, self.input_dim

    def output_shape(self) -> tuple:
        '''
        Returns the shape of predictions for a single sample as a tuple, which is (num_classes,).
        '''

        return self.num_classes,

    def train(self, data: np.ndarray, labels: np.ndarray) -> float:
        '''
        Train the model on batch of data.
        As training simply entails storing the data, the model is reset each time this method is called.
        Data are the input data, with shape (m, input_dim) and type np.float32 (m is arbitrary).
        Labels has shape (m,) and integral values between 0 and num_classes - 1.
        Returns 0 as there is no training loss to compute.
        Raises TypeError on invalid argument types.
        Raises ValueError on invalid argument values.
        Raises RuntimeError on other errors.
        '''

        # TODO check for errors
        self.data = data
        self.labels = labels
        return 0

    def predict(self, data: np.ndarray) -> np.ndarray:
        '''
        Predict softmax class scores from input data.
        Data are the input data, with a shape compatible with input_shape().
        The label array has shape (n, output_shape()) with n being the number of input samples.
        Raises TypeError on invalid argument types.
        Raises ValueError on invalid argument values.
        Raises RuntimeError on other errors.
        '''

        # TODO implement error checks
        num_of_samples = data.shape[0]
        labels = np.ndarray((num_of_samples, self.output_shape()))
        for idx in range(0, num_of_samples):
            sample = data[idx]
            neighbours = self.getNeighbors(sample)
            responses = self.getResponses(neighbours)
            labels[idx] = self.softmax(responses)
        return labels

    def getResponses(self, neighbors):
        '''
        :param neighbors: Instances to be counted
        :return: Dictionary with votes for each class
        '''

        classVotes = {}
        for x in range(len(neighbors)):
            response = neighbors[x].label
            if response in classVotes:
                classVotes[response] += 1
            else:
                classVotes[response] = 1
        return classVotes

    def euclideanDistance(self, instance1, instance2):
        # https://stackoverflow.com/questions/1401712/how-can-the-euclidean-distance-be-calculated-with-numpy
        dist = np.linalg.norm(instance1 - instance2)
        return dist

    def getNeighbors(self, instance):
        '''
        Return k nearest neighbors.
        '''
        distances = []
        for x in range(len(self.data)):
            dist = self.euclideanDistance(instance, self.data[x])
            distances.append((self.data[x], dist))
        distances.sort(key=np.operator.itemgetter(1))
        neighbors = []
        for x in range(self.k):
            neighbors.append(distances[x][0])
        return neighbors

    def softmax(self, votes):
        denominator = 0
        for value in votes.values():
            denominator += np.exp(value)
        result = {}
        for key, value in votes:
            result[key] = value * 1.0 / denominator
        return result