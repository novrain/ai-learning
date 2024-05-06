from steps.step01 import *
from steps.step02 import *
from steps.step03 import *


def numerical_diff(f, x, esp=1e-4):
    x0 = Variable(as_array(x.data - esp))
    x1 = Variable(as_array(x.data + esp))
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * esp)


if __name__ == "__main__":
    f = Square()
    x = Variable(np.array(2))
    d = numerical_diff(f, x)
    print(d)

    A = Square()
    B = Exp()
    C = Square()

    def f(x):
        return C(B(A(x)))

    x = Variable(np.array(0.5))
    d = numerical_diff(f, x)
    a = A(x)
    b = B(a)
    y = C(b)
    print(d)

    y.grad = np.array(1.0)
    b.grad = C.backward(y.grad)
    a.grad = B.backward(b.grad)
    x.grad = A.backward(a.grad)
    print(x.grad)

    C = y.creator
    b = C.input
    b.grad = C.backward(y.grad)

    B = b.creator
    a = B.input
    a.grad = B.backward(b.grad)

    A = a.creator
    x = A.input
    x.grad = A.backward(a.grad)
    print(x.grad)

    x.grad = -1
    y.grad = np.array(1.0)
    y.backward()
    print(x.grad)

    y.grad = None
    x = Variable(np.array(0.5))
    y = square(exp(square(x)))
    y.backward()
    print(x.grad)
