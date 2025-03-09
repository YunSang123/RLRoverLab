import torch
import numpy as np
from torch.distributions import Normal
import gymnasium as gym
import math
from teacher_model import get_model_gaussian_conv
import inspect

"""
states 구성
actions             2
distance            1
heading             1
angle_diff          1
dense_height_scan   3721 = 61^2
sparse_height_scan  961 = 31^2
"""

num_obs_sparse = 961
num_obs_dense = 3721
num_obs = num_obs_sparse + num_obs_dense
num_actions = 2
device = torch.device("cuda:0")
observation_space = gym.spaces.Box(low=-math.inf, high=math.inf, shape=(num_obs,))
action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(num_actions,))

model = get_model_gaussian_conv(observation_space, action_space, device, num_obs_dense, num_obs_sparse)
model = model.to(device)

load_path = "model/teacher/osr_t278800.pt"
checkpoint = torch.load(load_path)
model.load_state_dict(checkpoint["policy"])
print("==================================")
for name, param in model.state_dict().items():
    print(name)
    print(param)

key = "states"
value = torch.rand(1, 4687, device='cuda:0')
states = {key:value}
print(f"{type(states)}")
print(f"key = {key}")
print(f"value = {value}")
print(f"len = {value.shape}")
action = model.compute(states)
print(action)

mean_actions, log_std, outputs = model.compute(states)

log_std = torch.clamp(log_std, -20, 2)

# distribution
distribution = Normal(mean_actions, log_std.exp())

# sample using the reparameterization trick
actions = distribution.rsample()

# clip actions
low = np.array([-1., -1.])
high = np.array([1., 1.])
clip_actions_min = torch.tensor(low, device=device, dtype=torch.float32)
clip_actions_max = torch.tensor(high, device=device, dtype=torch.float32)
actions = torch.clamp(actions, min=clip_actions_min, max=clip_actions_max)

lin_vel = actions[0][0].item()
ang_vel = actions[0][1].item()

print(lin_vel)
print(ang_vel)

# lin_vel = actions[0][0].item()
# ang_vel = actions[1].item()
# print(lin_vel, ang_vel)