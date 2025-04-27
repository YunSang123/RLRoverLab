import torch

student = torch.load("./student_load/best.pt", map_location=torch.device('cpu'))

for k,v in student["state_dict"].items():
    print(k)
    print(v[0])