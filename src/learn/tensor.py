import torch

tensor = torch.ones(1, 4)
print(tensor)
print(tensor.T)

print(torch.ones(2,3))
z1 = tensor @ tensor.T
z2 = tensor.mul(torch.ones(2,3).T)
print(z1)
print(z2)