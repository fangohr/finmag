import numpy as np
import math

from sandbox.normal_modes.normal_modes import differentiate_fd


if __name__=="__main__":
    def f(x):
        return np.array([math.exp(x[0])+math.sin(x[1]), 0.])

    x = np.array([1.2, 2.4])
    dx = np.array([3.5, 4.5])
    value = dx[0]*math.exp(x[0]) + dx[1]*math.cos(x[1])
    print differentiate_fd(f, x, dx), differentiate_fd(f, x, dx)[0] - value

