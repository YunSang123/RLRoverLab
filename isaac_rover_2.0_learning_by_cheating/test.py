import torch

student = torch.load("runs/test7/best.pt")
# print(student["state_dict"])

for k,v in student["state_dict"].items():
    print(k)