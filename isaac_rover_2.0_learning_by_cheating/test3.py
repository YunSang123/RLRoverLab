import torch

student_policy = torch.load('runs/test2/best.pt', weights_only=True)

for k,v in student_policy['state_dict'].items():
    print(k)
    print(v)

# teacher_policy = torch.load('teacher_model/best_agent_685k.pt', weights_only=True)

# for k, v in teacher_policy['policy'].items():
#     print(k)
#     print(v)