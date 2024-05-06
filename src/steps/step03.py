import numpy as np
from steps.step02 import *

class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        return np.exp(self.input.data) * gy
    
def exp(x):
    return Exp()(x)

if __name__ == '__main__':
    A = Square()
    B = Exp()
    C = Square()

    x = Variable(np.array(0.5))
    a = A(x)
    b = B(a)
    y = C(b)
    print(y.data)

    assert y.creator == C
    assert y.creator.input == b
    assert y.creator.output == y
    assert y.creator.input.creator == B 