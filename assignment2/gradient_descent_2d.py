
import cv2
import numpy as np

import os
from collections import namedtuple

Vec2 = namedtuple('Vec2', ['x1', 'x2'])


class Fn:
    '''
    A 2D function evaluated on a grid.
    '''

    def __init__(self, fpath: str):
        '''
        Ctor that loads the function from a PNG file.
        Raises FileNotFoundError if the file does not exist.
        '''

        if not os.path.isfile(fpath):
            raise FileNotFoundError()

        self._fn = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
        self._fn = self._fn.astype(np.float32)
        self._fn /= (2**16-1)

    def visualize(self) -> np.ndarray:
        '''
        Return a visualization as a color image.
        Use the result to visualize the progress of gradient descent.
        '''

        vis = self._fn - self._fn.min()
        vis /= self._fn.max()
        vis *= 255
        vis = vis.astype(np.uint8)
        vis = cv2.applyColorMap(vis, cv2.COLORMAP_HOT)

        return vis

    def __call__(self, loc: Vec2) -> float:
        '''
        Evaluate the function at location loc.
        Raises ValueError if loc is out of bounds.
        '''
        if loc[0] < 0 or loc[0] >= self._fn.shape[0] \
                or loc[1] < 0 or loc[1] >= self._fn.shape[1]:
            raise ValueError("Index " + str(loc) + " is out of bounds")

        # you can simply round and map to integers. if so, make sure not to set eps and step_size too low
        # for bonus points you can implement some form of interpolation (linear should be sufficient)

        # return self._fn[int(loc[0]), int(loc[1])]
        return self.interpolated_value(loc)

    def interpolated_value(self, loc: Vec2) -> float:
        # https://stackoverflow.com/questions/32124170/bilinear-image-interpolation-scaling-a-calculation-example
        x = loc[0]
        y = loc[1]
        x1 = int(x)
        x2 = int(min(x + 1, self._fn.shape[0] - 1))

        y1 = int(y)
        y2 = int(min(y + 1, self._fn.shape[1] - 1))

        R1 = self._fn[x1, y1] + (x - x1) / (x2 - x1) * (self._fn[x2, y1] - self._fn[x1, y1])
        R2 = self._fn[x1, y2] + (x - x1) / (x2 - x1) * (self._fn[x2, y2] - self._fn[x1, y2])

        P = R2 + (y - y2) / (y2 - y1) * (R1 - R2)

        return P


def grad(fn: Fn, loc: Vec2, eps: float) -> Vec2:
    '''
    Compute the numerical gradient of a 2D function fn at location loc,
    using the given epsilon. See lecture 5 slides.
    Raises ValueError if loc is out of bounds of fn or if eps <= 0.
    '''

    if eps <= 0:
        raise ValueError("Eps must be > 0.")
    first_param = (fn(Vec2(loc[0] + eps, loc[1])) - fn(Vec2(loc[0] - eps, loc[1]))) / (2 * eps)
    second_param = (fn(Vec2(loc[0], loc[1] + eps)) - fn(Vec2(loc[0], loc[1] - eps))) / (2 * eps)

    return Vec2(first_param, second_param)


def gradient_too_small(gradient, treshold):
    return abs(gradient[0]) < treshold or abs(gradient[1]) < treshold


if __name__ == '__main__':
    # parse args

    import argparse

    parser = argparse.ArgumentParser(description='Perform gradient descent on a 2D function.')
    parser.add_argument('fpath', help='Path to a PNG file encoding the function')
    parser.add_argument('sx1', type=float, help='Initial value of the first argument')
    parser.add_argument('sx2', type=float, help='Initial value of the second argument')
    parser.add_argument('--eps', type=float, default=1.0, help='Epsilon for computing numeric gradients')
    parser.add_argument('--step_size', type=float, default=10.0, help='Step size')
    parser.add_argument('--beta', type=float, default=0, help='Beta parameter of momentum (0 = no momentum)')
    parser.add_argument('--nesterov', action='store_true', help='Use Nesterov momentum')
    args = parser.parse_args()

    # init

    fn = Fn(args.fpath)
    vis = fn.visualize()
    loc = Vec2(args.sx1, args.sx2)

    step_size = args.step_size
    beta = args.beta
    eps = args.eps
    nesterov = args.nesterov
    velocityX = 0
    velocityY = 0
    treshold = 1e-12
    # perform gradient descent

    firstIteration = True
    BLUE = [255, 0, 0]
    while True:
        # TODO implement normal gradient descent with nesterov momentum depending on the arguments (see lecture 4 slides)
        gradient = grad(fn=fn, loc=loc, eps=eps)
        velocityX = beta * velocityX - step_size * gradient[0]
        velocityY = beta * velocityY - step_size * gradient[1]
        new_loc = Vec2(loc[0] + velocityX, loc[1] + velocityY)

        if gradient_too_small(gradient, treshold):
            print("Gradient: " + str(gradient))
            print("Minimum found: " + str(new_loc) + ", value: " + str(fn(new_loc)))
            break

        if not firstIteration:
            cv2.line(vis, (int(loc[0]), int(loc[1])), (int(new_loc[0]), int(new_loc[1])), BLUE)
            cv2.imshow('Progress', vis)
            cv2.waitKey(20)
        loc = new_loc
        firstIteration = False

