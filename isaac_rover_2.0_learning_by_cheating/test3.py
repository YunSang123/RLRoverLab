import torch

student_policy = torch.load('load/1epoch_1e-4lr.pt', weights_only=True)

for k,v in student_policy['state_dict'].items():
    print(k)