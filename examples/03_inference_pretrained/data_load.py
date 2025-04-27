import torch

data = torch.load('260k_data1.pt')

print(data['data'][6,0,:7])