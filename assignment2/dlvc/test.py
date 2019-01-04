import numpy as np

from abc import ABCMeta, abstractmethod

class PerformanceMeasure(metaclass=ABCMeta):
    '''
    A performance measure.
    '''

    @abstractmethod
    def reset(self):
        '''
        Resets the internal state.
        '''

        pass

    @abstractmethod
    def update(self, prediction: np.ndarray, target: np.ndarray):
        '''
        Update the measure by comparing predicted data with ground-truth target data.
        Raises ValueError if the data shape or values are unsupported.
        '''

        pass

    @abstractmethod
    def __str__(self) -> str:
        '''
        Return a string representation of the performance.
        '''

        pass

    @abstractmethod
    def __lt__(self, other) -> bool:
        '''
        Return true if this performance measure is worse than another performance measure of the same type.
        Raises TypeError if the types of both measures differ.
        '''

        pass

    @abstractmethod
    def __gt__(self, other) -> bool:
        '''
        Return true if this performance measure is better than another performance measure of the same type.
        Raises TypeError if the types of both measures differ.
        '''

        pass


class Accuracy(PerformanceMeasure):
    '''
    Average classification accuracy.
    '''

    def __init__(self):
        '''
        Ctor.
        '''
        self.value = 0.0
        self.prediction = None
        self.target = None

        # recalculate the whole accuracy only when requested and a new batch is inserted since the last request
        self.state_updated = True

    def reset(self):
        '''
        Resets internal state.
        '''

        self.value = 0.0
        self.prediction = None
        self.target = None
        self.state_updated = True

    def update(self, prediction: np.ndarray, target: np.ndarray):
        '''
        Update the measure by comparing predicted data with ground-truth target data.
        prediction must have shape (s,c) with each row being a class-score vector.
        target must have shape (s,) and values between 0 and c-1 (true class labels).
        Raises ValueError if the data shape or values are unsupported.
        '''

        if type(prediction) != np.ndarray:
            raise TypeError("Prediction type must be np.ndarray")
        if len(prediction.shape) != 2:
            raise ValueError("Prediction must have shape (s,c)")

        if type(target) != np.ndarray:
            raise TypeError("Target type must be np.ndarray")
        if len(target.shape) != 1:
            raise ValueError("Target must have shape (s,)")

        if len(target) != len(prediction):
            raise ValueError("Target and prediction arrays don't have the same length.")

        if self.prediction is None:
            self.prediction = prediction.copy()
        else:
            np.append(self.prediction, prediction.copy())

        if self.target is None:
            self.target = target.copy()
        else:
            np.append(self.target, target.copy())

        self.state_updated = True

    def __str__(self):
        '''
        Return a string representation of the performance.
        '''
        return "accuracy: " + str(self.accuracy())

    def __lt__(self, other) -> bool:
        '''
        Return true if this accuracy is worse than another one.
        Raises TypeError if the types of both measures differ.
        '''

        if type(other) is None or type(other) != Accuracy:
            raise TypeError("types of both measures differ")

        return self.accuracy() < other.accuracy()

    def __gt__(self, other) -> bool:
        '''
        Return true if this accuracy is worse than another one.
        Raises TypeError if the types of both measures differ.
        '''

        if type(other) is None or type(other) != Accuracy:
            raise TypeError("types of both measures differ")

        return self.accuracy() > other.accuracy()

    def accuracy(self) -> float:
        '''
        Compute and return the accuracy as a float between 0 and 1.
        Returns 0 if no data is available (after resets).
        '''

        if self.state_updated:
            correct = 0
            for index, trueLabel in enumerate(self.target):
                predicted_label = np.argmax(self.prediction[index])
                if predicted_label == trueLabel:
                    correct += 1
            self.value = correct * 1.0 / len(self.target)
            self.state_updated = False

        return float(self.value)
