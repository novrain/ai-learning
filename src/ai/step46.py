import numpy as np
from ai import Variable, Model
from ai import optimizers
import ai.layers as L
import ai.functions as F


class TwoLayerNet(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.l1 = L.Linear(hidden_size)
        self.l2 = L.Linear(out_size)

    def forward(self, x):
        y = F.sigmoid(self.l1(x))
        y = self.l2(y)
        return y


# x = Variable(np.random.randn(5, 10), name="x")
# model = TwoLayerNet(100, 10)
# model.plot(x)

np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

lr = 0.2
max_iter = 10000
hidden_size = 10

model = TwoLayerNet(100, 10)
optimizer = optimizers.SGD(lr)
optimizer.setup(model)

for i in range(max_iter):
    y_pred = model(x)
    loss = F.mean_squared_error(y, y_pred)

    model.cleargrads()
    loss.backward()

    # for p in model.params():
    #     p.data -= lr * p.grad.data
    optimizer.update()

    if i % 1000 == 0:
        print(loss)
