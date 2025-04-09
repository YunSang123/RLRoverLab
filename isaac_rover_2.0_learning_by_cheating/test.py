import torch
from model import Student
from dataset import TeacherDataset

DEVICE = 'cuda:0'

student_model = torch.load('load/best_6epoch_8batchsize_1e-4lr.pt')
dataset = torch.load('teacher_model/data_test.pt')['data']

def cfg_fn():
    cfg = {
        "info":{
            "reset":            0,
            "actions":          0,
            "proprioceptive":   0,
            "exteroceptive":    0,
        },
        "learning":{
            "learning_rate": 1e-5,
            "epochs": 1,        # tmp = 5
            "batch_size": 8,    # batch_size = 8
        },
        "conv_encoder":{
            "activation_function": "leakyrelu",
            "encoder_features": [8,16,32,64],
            "output_size": 60,
        },

        "belief_encoder": {
            "hidden_dim":       300,
            "n_layers":         2,
            "activation_function":  "leakyrelu",
            "gb_features": [128,128,120],
            "ga_features": [128,128,120]},

        "belief_decoder": {
            "activation_function": "leakyrelu",
            "gate_features":    [1000,1500],
            "decoder_features": [1000,1500]
        },
        "mlp":{"activation_function": "leakyrelu",
            "network_features": [256,160,128]},
            }

    return cfg

train_ds = TeacherDataset("teacher_model/")
model = Student(info=train_ds.get_info(), cfg=cfg_fn(), teacher="teacher_model/agent_610k.pt").to(DEVICE)

model.load_state_dict(student_model['state_dict'])

state = torch.zeros((1, 1, 1124), device=DEVICE)

state[0,0,3] = 2

print("dataset.shape")
print(dataset.shape)

dataset = dataset[1000,200,:]

print("dataset.shape")
print(dataset.shape)

reset = dataset[0]
actions = dataset[1:3]
proprioceptive = dataset[3:7]
print(f"reset = {reset}")
print(f"actions = {actions}")
print(f"proprioceptive = {proprioceptive}")

h = model.belief_encoder.init_hidden(1).to(DEVICE)

# for i in range(10000):
#     a,p,h = model(state, h)
#     print(f"{i}번째 actions = {a}")
