import numpy as np
from ai import Variable

x0 = Variable(np.array([1, 2, 3]))
x1 = Variable(np.array([10]))

y = x0 + x1
print(y)
y.backward()

print(x1.grad)
print(x0.grad)
