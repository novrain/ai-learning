import numpy as np
from ai import Variable
import ai.functions as F

x = Variable(np.random.randn(2, 3))
W = Variable(np.random.randn(3, 4))
print(x)
print(W)

y = F.matmul(x, W)
print(y)
y.backward()
print(x.grad)
print(W.grad)
