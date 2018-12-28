import numpy as np
import random

from typing import List, Callable
# All operations are functions that take and return numpy arrays
# See https://docs.python.org/3/library/typing.html#typing.Callable for what this line means
Op = Callable[[np.ndarray], np.ndarray]


def chain(ops: List[Op]) -> Op:
    '''
    Chain a list of operations together.
    '''

    def op(sample: np.ndarray) -> np.ndarray:
        for op_ in ops:
            sample = op_(sample)
        return sample

    return op


def type_cast(dtype: np.dtype) -> Op:
    '''
    Cast numpy arrays to the given type.
    '''

    def op(sample: np.ndarray) -> np.ndarray:
        sample = sample.astype(dtype)
        return sample

    return op


def vectorize() -> Op:
    '''
    Vectorize numpy arrays via "numpy.ravel()".
    '''

    def op(sample: np.ndarray) -> np.ndarray:
        sample = np.ravel(sample)
        return sample

    return op


def hwc2chw() -> Op:
    '''
    Flip a 3D array with shape HWC(0,1,2) to shape CHW(2,0,1).
    '''

    def op(sample: np.ndarray) -> np.ndarray:
        sample = sample.transpose(2, 0, 1)
        return sample

    return op


def chw2hwc() -> Op:
    '''
    Flip a 3D array with shape CHW(0,1,2) to HWC(1,2,0).
    '''

    def op(sample: np.ndarray) -> np.ndarray:
        sample = sample.transpose(1, 2, 0)
        return sample

    return op


def add(val: float) -> Op:
    '''
    Add a scalar value to all array elements.
    '''

    def op(sample: np.ndarray) -> np.ndarray:
        sample += val
        return sample

    return op


def mul(val: float) -> Op:
    '''
    Multiply all array elements by the given scalar.
    '''

    def op(sample: np.ndarray) -> np.ndarray:
        sample = np.multiply(sample, val)
        return sample

    return op

def norm() -> Op:
    '''
    Per-channel normalization based on statistics of training set
    '''

    #TODO implement

    pass


def hflip() -> Op:
    '''
    Flip arrays with shape HWC horizontally with a probability of 0.5.
    '''

    def op(sample: np.ndarray) -> np.ndarray:
        if random.random() < 0.5:
            sample = np.fliplr(sample)
        return sample

    return op


def rcrop(sz: int, pad: int, pad_mode: str) -> Op:
    '''
    Extract a square random crop of size sz from arrays with shape HWC.
    If pad is > 0, the array is first padded by pad pixels along the top, left, bottom, and right.
    How padding is done is governed by pad_mode, which should work exactly as the 'mode' argument of numpy.pad.
    Raises ValueError if sz exceeds the array width/height after padding.
    '''

    # TODO implement
    # https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.pad.html will be helpful

    pass