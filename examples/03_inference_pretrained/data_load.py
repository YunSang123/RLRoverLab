import torch

data = torch.load('540k_data1.pt')

print(data["data"].shape)