import ai
import ai.datasets

train_set = ai.datasets.MNIST(train=True, transform=None)
test_set = ai.datasets.MNIST(train=False, transform=None)

print(len(train_set))
print(len(test_set))
