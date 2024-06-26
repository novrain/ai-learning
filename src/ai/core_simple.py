import weakref
import numpy as np
import contextlib

import ai

try:
    import cupy

    array_types = (np.ndarray, cupy.ndarray)
except ImportError:
    array_types = np.ndarray


def as_array(x, array_module=np):
    if np.isscalar(x):
        return array_module.array(x)
    return x


class Config:
    enable_backprop = True


@contextlib.contextmanager
def using_config(name, value):
    old = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old)


def no_grad():
    return using_config("enable_backprop", False)


class Variable:
    __array_priority__ = 200

    def __init__(self, data, name=None) -> None:
        if data is not None:
            if not isinstance(data, array_types):
                raise TypeError("{} is not supported".format(type(data)))

        self.data = data
        self.name = name
        self.grad = None
        self.creator = None
        self.generation = 0

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.data is None:
            return "variable(None)"
        p = str(self.data).replace("\n", "\n" + " " * 9)
        return "variable(" + p + ")"

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    # def backward(self):
    #     f = self.creator
    #     if f is not None:
    #         x = f.input
    #         x.grad = f.backward(self.grad)
    #         x.backward()
    def backward(self, retain_grad=False, create_graph=False):
        if self.grad is None:
            xp = ai.cuda.get_array_module(self.data)
            self.grad = Variable(xp.ones_like(self.data))

        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        while funcs:
            f = funcs.pop()
            gys = [output().grad for output in f.outputs]

            with using_config("enable_backprop", create_graph):
                gxs = f.backward(*gys)
                if not isinstance(gxs, tuple):
                    gxs = (gxs,)

                for x, gx in zip(f.inputs, gxs):
                    if x.grad is None:
                        x.grad = gx
                    else:
                        x.grad = x.grad + gx

                    if x.creator is not None:
                        add_func(x.creator)

            if not retain_grad:
                for y in f.outputs:
                    y().grad = None

    def cleargrad(self):
        self.grad = None

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return ai.functions.reshape(self, shape)

    def transpose(self):
        return ai.functions.transpose(self)

    @property
    def T(self):
        return ai.functions.transpose(self)

    def sum(self, axis=None, keepdims=False):
        return ai.functions.sum(self, axis, keepdims)

    def to_cpu(self):
        if self.data is not None:
            self.data = ai.cuda.as_numpy(self.data)

    def to_gpu(self):
        if self.data is not None:
            self.data = ai.cuda.as_cupy(self.data)


class Parameter(Variable):
    pass


def as_variable(x):
    if isinstance(x, Variable):
        return x
    return Variable(x)


class Function:
    def __call__(self, *inputs):
        inputs = [as_variable(x) for x in inputs]
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()


class Add(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 + x1
        return y

    def backward(self, gy):
        gx0, gx1 = gy, gy
        if self.x0_shape != self.x1_shape:
            gx0 = ai.functions.sum_to(gx0, self.x0_shape)
            gx1 = ai.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1


def add(x0, x1):
    x1 = as_array(x1, ai.cuda.get_array_module(x1.data))
    return Add()(x0, x1)


class Square(Function):
    def forward(self, x):
        return x**2

    def backward(self, gy):
        x = self.inputs[0]
        gx = 2 * x * gy
        return gx


def square(x):
    return Square()(x)


class Mul(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        return x0 * x1

    def backward(self, gy):
        x0, x1 = self.inputs[0], self.inputs[1]
        return gy * x1, gy * x0


def mul(x0, x1):
    x1 = as_array(x1, ai.cuda.get_array_module(x0.data))
    return Mul()(x0, x1)


class Neg(Function):
    def forward(self, xs):
        return -xs

    def backward(self, gy):
        return -gy


def neg(x):
    return Neg()(x)


class Sub(Function):
    def forward(self, x0, x1):
        y = x0 - x1
        return y

    def backward(self, gy):
        return gy, -gy


def sub(x0, x1):
    x1 = as_array(x1, ai.cuda.get_array_module(x1.data))
    return Sub()(x0, x1)


def rsub(x0, x1):
    x1 = as_array(x1, ai.cuda.get_array_module(x1.data))
    return Sub()(x1, x0)


class Div(Function):
    def forward(self, x0, x1):
        return x0 / x1

    def backward(self, gy):
        x0, x1 = self.inputs[0], self.inputs[1]
        return gy / x1, -x0 * gy / (x1**2)


def div(x0, x1):
    x1 = as_array(x1, ai.cuda.get_array_module(x1.data))
    return Div()(x0, x1)


def rdiv(x0, x1):
    x1 = as_array(x1, ai.cuda.get_array_module(x1.data))
    return Div()(x1, x0)


class Pow(Function):
    def __init__(self, c) -> None:
        self.c = c  # 不作为参数来看待，避免backward

    def forward(self, x):
        return x**self.c

    def backward(self, gy):
        x = self.inputs[0]
        c = self.c
        gx = c * x ** (c - 1) * gy
        return gx


def pow(x, c):
    return Pow(c)(x)


def setup_variable():
    Variable.__mul__ = mul
    Variable.__rmul__ = mul
    Variable.__add__ = add
    Variable.__radd__ = add
    Variable.__neg__ = neg
    Variable.__sub__ = sub
    Variable.__rsub__ = rsub
    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv
    Variable.__pow__ = pow
    Variable.__getitem__ = ai.functions.get_item
