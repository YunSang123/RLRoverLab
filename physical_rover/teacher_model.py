import inspect
from models import GaussianNeuralNetworkConv
# GaussianNeuralNetworkConv는 같은 경로의 models.py로부터 실행됨.

def get_model_gaussian_conv(
    observation_space,
    action_space,
    device,
    dense_encoder_input_size : int,
    sparse_encoder_input_size : int):
    mlp_input_size = 5

    model = GaussianNeuralNetworkConv(
        observation_space=observation_space,
        action_space=action_space,
        device=device,
        mlp_input_size=mlp_input_size,
        mlp_layers=[256, 160, 128],
        mlp_activation="leaky_relu",
        dense_encoder_input_size=dense_encoder_input_size,
        sparse_encoder_input_size=sparse_encoder_input_size,
        encoder_layers=[8, 16, 32, 64],
        encoder_activation="leaky_relu",
    )
    return model