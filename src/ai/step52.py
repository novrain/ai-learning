import ai
from ai import optimizers
import ai.cuda
from ai.dataloaders import DataLoader
import ai.datasets
import matplotlib.pyplot as plt
import ai.functions as F

from ai.models import MLP

max_epoch = 5
batch_size = 100
hidden_size = 1000

train_set = ai.datasets.MNIST(train=True)
test_set = ai.datasets.MNIST(train=False)
train_loader = DataLoader(train_set, batch_size)
test_loader = DataLoader(test_set, batch_size, shuffle=False)


# print(len(train_set))
# print(len(test_set))

# x, t = train_set[0]
# print(type(x), x.shape)
# print(t)

# plt.imshow(x.reshape(28, 28), cmap="gray")
# plt.axis("off")
# plt.show()
# print("label:", t)


model = MLP((hidden_size, hidden_size, 10), activation=F.relu)
optimizer = optimizers.Adam().setup(model)

if ai.cuda.gpu_enable:
    train_loader.to_gpu()
    model.to_gpu()

for epoch in range(max_epoch):
    sum_loss, sum_acc = 0, 0

    for x, t in train_loader:
        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
        acc = F.accuracy(y, t)
        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(t)
        sum_acc += float(acc.data) * len(t)

    print("epoch: {}".format(epoch + 1))
    print(
        "train loss: {:.4f}, accuracy: {:.4f}".format(
            sum_loss / len(train_set), sum_acc / len(train_set)
        )
    )

    sum_loss, sum_acc = 0, 0
    with ai.no_grad():
        for x, t in test_loader:
            y = model(x)
            loss = F.softmax_cross_entropy(y, t)
            acc = F.accuracy(y, t)
            sum_loss += float(loss.data) * len(t)
            sum_acc += float(acc.data) * len(t)

        print(
            "test loss: {:.4f}, accuracy: {:.4f}".format(
                sum_loss / len(test_set), sum_acc / len(test_set)
            )
        )
