import numpy as np
import random
import cv2
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

    # TODO implement

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


def vflip() -> Op:
    '''
    Flip arrays with shape HWC vertically with a probability of 0.5.
    '''

    def op(sample: np.ndarray) -> np.ndarray:
        # rotate for 180 + horizontal flip
        if random.random() < 0.5:
            sample = np.rot90(sample, 2)
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

    def op(sample: np.ndarray) -> np.ndarray:
        height = sample.shape[0]
        width = sample.shape[1]
        if pad > 0:
            # npad is a tuple of (n_before, n_after) for each dimension
            npad = ((pad, pad), (pad, pad), (0, 0))
            sample = np.pad(sample, npad, pad_mode)

        if sz > sample.shape[0] or sz > sample.shape[1]:
            raise ValueError("Square to crop exceeds the array width/height (after padding).")

        y = random.randint(0, sample.shape[0] - sz)
        x = random.randint(0, sample.shape[1] - sz)

        cropped = sample[y:y + sz, x:x + sz, :]
        new_sample = cv2.resize(cropped, dsize=(height, width))
        return new_sample

    return op


def rotate90(k: int) -> Op:
    '''
    Rotates image from the first towards the second axis for k * 90 degrees with a probability of 0.5.
    If k is odd number, height and width of an image will be changed (swapped).
    :param k: how many times image will be rotated for 90 degrees.
    '''
    def op(sample: np.ndarray) -> np.ndarray:
        if random.random() < 0.5:
            sample = np.rot90(sample, k)
        return sample
    return op

