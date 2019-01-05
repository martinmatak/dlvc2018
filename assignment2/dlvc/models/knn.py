from ..model import Model
import numpy as np
import operator


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

        self.data = None
        self.labels = None

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

        if type(data) != np.ndarray:
            raise TypeError("Data type must be np.ndarray")
        if data.dtype != np.float32:
            raise TypeError("Data dtype must be np.float32")
        if len(data.shape) != 2:
            raise ValueError("Data shape tuple must have length 2")
        if data.shape[1] != self.input_dim:
            raise ValueError("Data shape tuple must have shape (m, input_dim) where m is arbitrary")

        if type(labels) != np.ndarray:
            raise TypeError("Labels type must be np.ndarray")
        if len(labels.shape) != 1:
            raise ValueError("Labels shape tuple must have length 1")
        for label in labels:
            if label < 0 or label > self.num_classes - 1:
                raise ValueError("Label out of scope!")
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

        if type(data) != np.ndarray:
            raise TypeError("Data must be of type np.ndarray")
        if len(data.shape) != len(self.input_shape()) or data.shape[1:] != self.input_shape()[1:]:
            raise ValueError("Data must be compatible with shape shape: " + str(self.input_shape()))

        num_of_samples = data.shape[0]
        labels = np.ndarray((num_of_samples, ) + self.output_shape())
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

        for i in range(0, self.num_classes):
            classVotes[i] = 0

        for x in range(len(neighbors)):
            response = neighbors[x]["label"]
            classVotes[response] += 1

        return classVotes

    # use if small differences in between pixel values are more important than bigger differences (outliers)
    def L1distance(self, vectorA: np.ndarray, vectorB: np.ndarray):
        if len(vectorA) != len(vectorB):
            raise ValueError("Vectors have different dimensions.")
        return np.abs(vectorA - vectorB).sum()

    # use if outliers are more important than small differences in between pixel values
    def L2distance(self, vectorA: np.ndarray, vectorB: np.ndarray):
        if len(vectorA) != len(vectorB):
            raise ValueError("Vectors have different dimensions.")
        return np.sqrt(np.square(vectorA - vectorB).sum())

    def getNeighbors(self, instance):
        '''
        Return k nearest neighbors.
        '''
        distances = []

        for x in range(len(self.data)):
            dist = self.L1distance(instance, self.data[x])
            distances.append((self.labels[x], self.data[x], dist))

        distances.sort(key=operator.itemgetter(2))
        neighbors = []
        for x in range(self.k):
            neighbors.append({})
            neighbors[x]["label"] = distances[x][0]
            neighbors[x]["data"] = distances[x][1]
        return neighbors

    def softmax(self, votes):
        denominator = 0.0
        for value in votes.values():
            denominator += np.exp(value)
        result = []

        sum_probabilities = 0.0

        for classIndex in range(0, self.num_classes):
            probability = np.exp(votes.get(classIndex, 0) * 1.0) / denominator
            sum_probabilities += probability
            result.append(probability)

        assert sum_probabilities == 1.0, sum_probabilities
        return result
