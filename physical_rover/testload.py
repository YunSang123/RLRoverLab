import torch

model_path = "model/teacher/osr_t278800.pt"

policy_model = torch.load(model_path)["policy"]

for key, value in policy_model.items():
    print(key, value)

print("===========================================")
print(policy_model["dense_encoder.encoder_layers.4.weight"])