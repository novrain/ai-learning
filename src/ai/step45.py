import numpy as np
from ai import Variable
import ai.functions as F
import ai.layers as L

model = L.Layer()
model.l1 = L.Linear(5)
model.l2 = L.Linear(3)


def predict(model, x):
    y = model.l1(x)
    y = F.sigmoid(y)
    y = model.l2(y)
    return y


for p in model.params():
    print(p)

model.cleargrads()
