import numpy as np
from ai import Variable


def sphere(x, y):
    z = x**2 + y**2
    return z


def matyas(x, y):
    return 0.26 * (x**2 + y**2) - 0.48 * x * y


x = Variable(np.array(1.0))
y = Variable(np.array(1.0))
z = sphere(x, y)
z.backward()
print(x.grad, y.grad)

x.cleargrad()
y.cleargrad()
z = matyas(x, y)
z.backward()
print(x.grad, y.grad)
