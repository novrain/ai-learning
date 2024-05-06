import matplotlib.pyplot as plt
import numpy as np

import ai.functions as F
from ai import Variable
from ai.utils import plot_dot_graph

x = Variable(np.array(1.0))
y = F.tanh(x)

x.name = "x"
y.name = "y"

y.backward(create_graph=True)

iters = 5

for i in range(iters):
    gx = x.grad
    x.cleargrad()
    gx.backward(create_graph=True)

gx = x.grad
gx.name = "gx" + str(iters + 1)
plot_dot_graph(gx, verbose=False, to_file="tanh.png")
