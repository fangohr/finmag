import numpy as np

x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)

X, Y = np.meshgrid(x, y)
Z = 1 + X * Y

assert Z[0].min() >= 1
assert Z[-1].max() <= 2
Z -= Z[0]
assert Z[-1].max() <= 1
