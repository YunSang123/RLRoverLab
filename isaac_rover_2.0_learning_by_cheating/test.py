import torch

input = torch.load('input.pt')
predictions = torch.load('predictions.pt')
targets = torch.load('targets.pt')

print(input.shape)
print(input)

print(predictions.shape)
print(predictions)

print(targets.shape)
print(targets)