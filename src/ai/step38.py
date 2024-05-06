import numpy as np

n = np.random.rand(1, 2, 3)
print(n)
y = n.reshape((2, 3))
print(y)
y = n.reshape(2, 3)
print(y)

from ai import Variable
import ai.functions as F

x = Variable(np.array(([1, 2, 3], [4, 5, 6])))
y = F.reshape(x, (6,))
y.backward(retain_grad=True)
print(x.grad)

x = Variable(np.array(([1, 2, 3], [4, 5, 6])))
y = F.transpose(x)
y.backward(retain_grad=True)
print(x.grad)
