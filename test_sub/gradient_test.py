import torch
a=torch.tensor([1.0],requires_grad=True)
b=torch.tensor([2.0])
y=1-a*b
loss=y+1
loss.backward()
print(a.grad)
print(y.grad)
print(loss)
