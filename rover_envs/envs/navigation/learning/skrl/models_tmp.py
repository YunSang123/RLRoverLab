import torch
import torch.nn as nn
from skrl.models.torch.base import Model as BaseModel
from skrl.models.torch.deterministic import DeterministicMixin
from skrl.models.torch.gaussian import GaussianMixin


def get_activation(activation_name):
    """Get the activation function by name."""
    activation_fns = {
        "leaky_relu": nn.LeakyReLU(inplace=True),
        "relu": nn.ReLU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "elu": nn.ELU(),
        "relu6": nn.ReLU6(),
        "selu": nn.SELU(),
    }
    if activation_name not in activation_fns:
        raise ValueError(f"Activation function {activation_name} not supported.")
    return activation_fns[activation_name]


class HeightmapEncoder(nn.Module):
    def __init__(self, in_channels, encoder_features=[80, 60], encoder_activation="leaky_relu"):
        super().__init__()
        self.encoder_layers = nn.ModuleList()
        for feature in encoder_features:
            self.encoder_layers.append(nn.Linear(in_channels, feature))
            self.encoder_layers.append(get_activation(encoder_activation))
            in_channels = feature

    def forward(self, x):
        for layer in self.encoder_layers:
            x = layer(x)
        return x


class ConvHeightmapEncoder(nn.Module):
    def __init__(self, in_channels, encoder_features=[16, 32], encoder_activation="leaky_relu"):
        print("in_channels = ", in_channels)                        # 10201
        print("encoder_features = ", encoder_features)              # 8, 16, 32, 64
        print("encoder_activation = ", encoder_activation)          # leaky_relu
        super().__init__()
        # self.heightmap_size는 rover_env_cfg.py에서 height_scanner의 size의 제곱근과 같음. Ex) resolution=0.05, size=[5.0, 5.0] 이라고 하면, 한 변이 101개이므로, self.heightmap_size는 101이 나옴.
        self.heightmap_size = torch.sqrt(torch.tensor(in_channels)).int()   # tensor(101, dtype=torch.int32)
        
        print("self.heightmap_size = ",self.heightmap_size)         # tensor(101, dtype=torch.int32)
        # kernel = 가중치 필터
        kernel_size = 3
        
        # kernel이 움직이는 칸 수
        # 무조건 웬만하면 1로 하자.
        stride = 1
        
        # padding : 배열의 둘레를 확장하고 0으로 채우는 연산. Ex) 3,3일 경우 5,5로 되며 둘레가 다 0으로 채워짐.
        padding = 1
        
        self.encoder_layers = nn.ModuleList()
        in_channels = 1  # 1 channel for heightmap
        
        """
        kernel_size : 합성곱 필터 크기
        stride : 필터가 움직이는 간격
        padding : 배열의 둘레를 확장하기 위한 값
        
        nn.Conv2d : 입력 채널(in_channels)에서 출력 채널(feature)로의 합성곱 연산을 수행하는 nn.Conv2d 레이어를 추가.
        nn.BatchNorm2d : 배치 정규화를 수행하여 학습 속도를 높이고 안정성을 향상
        get_activation : 활성화 함수 추가
        nn.MaxPool2d : 최대 풀링(Max Pooling)을 수행하여 입력 데이터를 압축하고 주요 정보를 강조
        
        """
        for feature in encoder_features:
            # print("\n\n12342352358932y589027589")
            # print("feature = ", feature)    # feature가 불러와질때마다 8, 16, 32, 64가 출력됨.
            self.encoder_layers.append(nn.Conv2d(in_channels, feature, kernel_size=kernel_size,
                                       stride=stride, padding=padding, bias=False))
            # print("encoder_layers 출력 : \n",self.encoder_layers)
            self.encoder_layers.append(nn.BatchNorm2d(feature))
            # self.encoder_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            # print("encoder_layers 출력 : \n",self.encoder_layers)
            self.encoder_layers.append(get_activation(encoder_activation))
            # print("encoder_layers 출력 : \n",self.encoder_layers)
            self.encoder_layers.append(nn.Conv2d(feature, feature, kernel_size=kernel_size,
                                       stride=stride, padding=padding, bias=False))
            # print("encoder_layers 출력 : \n",self.encoder_layers)
            self.encoder_layers.append(nn.BatchNorm2d(feature))
            # print("encoder_layers 출력 : \n",self.encoder_layers)
            self.encoder_layers.append(get_activation(encoder_activation))
            # print("encoder_layers 출력 : \n",self.encoder_layers)
            # Pooling = 행렬을 압축해, 특정 데이터를 강조하는 역할을 수행!
            self.encoder_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            # print("encoder_layers 출력 : \n",self.encoder_layers)
            in_channels = feature
            # print("in_channels = ", in_channels)    # feature가 불러와질때마다 8, 16, 32, 64가 출력됨.
            # print("\n\n")
        out_channels = in_channels
        # print("out_channels = ", out_channels)      # 마지막으로 in_cprint("flatten_size : ", flatten_size)
        
        
        
        """_summary_
        목적 : CNN 레이어를 통과한 후 데이터의 너비(w)와 높이(h)를 계산하기 위함
        방법 : Kernel, stride, padding을 전부 고려해서 계산
        """
        flatten_size = [self.heightmap_size, self.heightmap_size]
        for _ in encoder_features:
            # Conv2D 레이어를 거치면 아래와 같이 너비와 높이가 변함.
            w = (flatten_size[0] - kernel_size + 2 * padding) // stride + 1
            print("w = ", w)
            h = (flatten_size[1] - kernel_size + 2 * padding) // stride + 1
            print("h = ", h)
            
            # Conv2D 레이어를 거치면 아래와 같이 너비와 높이가 변함.
            w = (w - kernel_size + 2 * padding) // stride + 1
            print("w = ", w)
            h = (h - kernel_size + 2 * padding) // stride + 1
            print("h = ", h)
            
            # Max Pooling을 거치면 아래와 같이 너비와 높이가 변함!
            w = (w - 2) // 2 + 1
            h = (h - 2) // 2 + 1
            flatten_size = [w, h]   # flatten_size :  [tensor(6, dtype=torch.int32), tensor(6, dtype=torch.int32)]
            
        self.conv_out_features = out_channels * flatten_size[0] * flatten_size[1]   # 64*6*6=tensor(2304, dtype=torch.int32)

        features = [80, 60]

        self.mlps = nn.ModuleList()
        in_channels = self.conv_out_features    # in_channels =  tensor(2304, dtype=torch.int32)
        
        for feature in features:
            self.mlps.append(nn.Linear(in_channels, feature))
            self.mlps.append(get_activation(encoder_activation))
            in_channels = feature
        # Mlp : 2304 -> 80 -> 60

        self.out_features = features[-1]

    def forward(self, x):
        # x is a flattened heightmap, reshape it to 2D
        # view함수는 텐서의 shape을 변경하는 함수임.
        # 처음에 -1은 자동으로 차원을 지정하라는 의미. 즉, 뒤의 값인 1에 맞게 알아서 shape이 변경됨.
        x = x.view(-1, 1, self.heightmap_size, self.heightmap_size)
        # print("%^&*(^*%*&%*&%^*(%&*(%*&%&*(%*&(%&*(%&*(())))))))")
        # print("x 출력중")
        # print(x)
        # print("x.shape = ",x.shape)

        for layer in self.encoder_layers:
            x = layer(x)
            # print("x = layer(x) 결과 출력")
            # print(x)

        x = x.view(-1, self.conv_out_features)
        # print("%^&*(^*%*&%*&%^*(%&*(%*&%&*(%*&(%&*(%&*(())))))))")
        # print("x 출력중")
        # print(x)
        for layer in self.mlps:
            x = layer(x)
            # print("x = layer(x) 결과 출력")
            # print(x)
        return x


class GaussianNeuralNetwork(GaussianMixin, BaseModel):
    """Gaussian neural network model."""

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        mlp_input_size=5,
        mlp_layers=[256, 160, 128],
        mlp_activation="leaky_relu",
        encoder_input_size=None,
        encoder_layers=[80, 60],
        encoder_activation="leaky_relu",
        **kwargs,
    ):
        """Initialize the Gaussian neural network model.

        Args:
            observation_space (gym.spaces.Space): The observation space of the environment.
            action_space (gym.spaces.Space): The action space of the environment.
            device (torch.device): The device to use for computation.
            encoder_features (list): The number of features for each encoder layer.
            encoder_activation (str): The activation function to use for each encoder layer.
        """
        BaseModel.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(
            self, clip_actions=True, clip_log_std=True, min_log_std=-20.0, max_log_std=2.0, reduction="sum"
        )

        self.mlp_input_size = mlp_input_size
        self.encoder_input_size = encoder_input_size

        in_channels = self.mlp_input_size
        if self.encoder_input_size is not None:
            self.dense_encoder = HeightmapEncoder(self.encoder_input_size, encoder_layers, encoder_activation)
            in_channels += encoder_layers[-1]

        self.mlp = nn.ModuleList()

        for feature in mlp_layers:
            self.mlp.append(nn.Linear(in_channels, feature))
            self.mlp.append(get_activation(mlp_activation))
            in_channels = feature

        action_space = action_space.shape[0]
        self.mlp.append(nn.Linear(in_channels, action_space))
        self.mlp.append(nn.Tanh())
        self.log_std_parameter = nn.Parameter(torch.zeros(action_space))

    def compute(self, states, role="actor"):
        # Split the states into proprioception and heightmap if the heightmap is used.
        if self.encoder_input_size is None:
            x = states["states"]
        else:
            encoder_output = self.dense_encoder(states["states"][:, self.mlp_input_size - 1:-1])
            x = states["states"][:, 0:self.mlp_input_size]
            x = torch.cat([x, encoder_output], dim=1)

        # Compute the output of the MLP.
        for layer in self.mlp:
            x = layer(x)

        return x, self.log_std_parameter, {}


class DeterministicNeuralNetwork(DeterministicMixin, BaseModel):
    """Gaussian neural network model."""

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        mlp_input_size=4,
        mlp_layers=[256, 160, 128],
        mlp_activation="leaky_relu",
        encoder_input_size=None,
        encoder_layers=[80, 60],
        encoder_activation="leaky_relu",
        **kwargs,
    ):
        """Initialize the Gaussian neural network model.

        Args:
            observation_space (gym.spaces.Space): The observation space of the environment.
            action_space (gym.spaces.Space): The action space of the environment.
            device (torch.device): The device to use for computation.
            encoder_features (list): The number of features for each encoder layer.
            encoder_activation (str): The activation function to use for each encoder layer.
        """
        BaseModel.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=False)

        self.mlp_input_size = mlp_input_size
        self.encoder_input_size = encoder_input_size

        in_channels = self.mlp_input_size
        if self.encoder_input_size is not None:
            self.dense_encoder = HeightmapEncoder(self.encoder_input_size, encoder_layers, encoder_activation)
            in_channels += encoder_layers[-1]

        self.mlp = nn.ModuleList()

        action_space = action_space.shape[0]
        for feature in mlp_layers:
            self.mlp.append(nn.Linear(in_channels, feature))
            self.mlp.append(get_activation(mlp_activation))
            in_channels = feature

        self.mlp.append(nn.Linear(in_channels, 1))

    def compute(self, states, role="actor"):
        if self.encoder_input_size is None:
            x = states["states"]
        else:
            x = states["states"][:, :self.mlp_input_size]
            encoder_output = self.dense_encoder(states["states"][:, self.mlp_input_size - 1:-1])
            x = torch.cat([x, encoder_output], dim=1)

        for layer in self.mlp:
            x = layer(x)

        return x, {}


class DeterministicActor(DeterministicMixin, BaseModel):
    """Deterministic actor model."""

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        mlp_input_size=4,
        mlp_layers=[256, 160, 128],
        mlp_activation="leaky_relu",
        encoder_input_size=None,
        encoder_layers=[80, 60],
        encoder_activation="leaky_relu",
        **kwargs,
    ):
        """Initialize the deterministic actor model.

        Args:
            observation_space (gym.spaces.Space): The observation space of the environment.
            action_space (gym.spaces.Space): The action space of the environment.
            device (torch.device): The device to use for computation.
            encoder_features (list): The number of features for each encoder layer.
            encoder_activation (str): The activation function to use for each encoder layer.
        """
        BaseModel.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=False)

        self.mlp_input_size = mlp_input_size
        self.encoder_input_size = encoder_input_size

        in_channels = self.mlp_input_size
        if self.encoder_input_size is not None:
            self.dense_encoder = HeightmapEncoder(self.encoder_input_size, encoder_layers, encoder_activation)
            in_channels += encoder_layers[-1]

        self.mlp = nn.ModuleList()

        action_space = action_space.shape[0]
        for feature in mlp_layers:
            self.mlp.append(nn.Linear(in_channels, feature))
            self.mlp.append(get_activation(mlp_activation))
            in_channels = feature

        self.mlp.append(nn.Linear(in_channels, action_space))

    def compute(self, states, role="actor"):
        if self.encoder_input_size is None:
            x = states["states"]
        else:
            x = states["states"][:, :self.mlp_input_size]
            encoder_output = self.dense_encoder(states["states"][:, self.mlp_input_size - 1:-1])
            x = torch.cat([x, encoder_output], dim=1)

        for layer in self.mlp:
            x = layer(x)

        return x, {}


class Critic(DeterministicMixin, BaseModel):
    """Critic model."""

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        mlp_input_size=4,
        mlp_layers=[256, 160, 128],
        mlp_activation="leaky_relu",
        encoder_input_size=None,
        encoder_layers=[80, 60],
        encoder_activation="leaky_relu",
        **kwargs,
    ):
        """Initialize the critic model.

        Args:
            observation_space (gym.spaces.Space): The observation space of the environment.
            action_space (gym.spaces.Space): The action space of the environment.
            device (torch.device): The device to use for computation.
            encoder_features (list): The number of features for each encoder layer.
            encoder_activation (str): The activation function to use for each encoder layer.
        """
        BaseModel.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=False)

        self.mlp_input_size = mlp_input_size
        self.encoder_input_size = encoder_input_size

        in_channels = self.mlp_input_size
        if self.encoder_input_size is not None:
            self.dense_encoder = HeightmapEncoder(self.encoder_input_size, encoder_layers, encoder_activation)
            in_channels += encoder_layers[-1]

        self.mlp = nn.ModuleList()

        for feature in mlp_layers:
            self.mlp.append(nn.Linear(in_channels, feature))
            self.mlp.append(get_activation(mlp_activation))
            in_channels = feature

        self.mlp.append(nn.Linear(in_channels, 1))

    def compute(self, states, role="actor"):
        if self.encoder_input_size is None:
            x = torch.cat([states["states"], states["taken_actions"]], dim=1)
        else:
            x = states["states"][:, :self.mlp_input_size]
            encoder_output = self.dense_encoder(states["states"][:, self.mlp_input_size - 1:-1])
            x = torch.cat([x, encoder_output], dim=1)

        for layer in self.mlp:
            x = layer(x)

        return x, {}


class GaussianNeuralNetworkConv(GaussianMixin, BaseModel):
    """Gaussian neural network model."""

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        mlp_input_size=5,
        mlp_layers=[256, 160, 128],
        mlp_activation="leaky_relu",
        encoder_input_size=None,
        encoder_layers=[80, 60],
        encoder_activation="leaky_relu",
        **kwargs,
    ):
        """Initialize the Gaussian neural network model.

        Args:
            observation_space (gym.spaces.Space): The observation space of the environment.
            action_space (gym.spaces.Space): The action space of the environment.
            device (torch.device): The device to use for computation.
            encoder_features (list): The number of features for each encoder layer.
            encoder_activation (str): The activation function to use for each encoder layer.
        """
        BaseModel.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(
            self, clip_actions=True, clip_log_std=True, min_log_std=-20.0, max_log_std=2.0, reduction="sum"
        )

        self.mlp_input_size = mlp_input_size            # self.mlp_input_size = 5
        self.encoder_input_size = encoder_input_size    # self.encoder_input_size = 10201

        in_channels = self.mlp_input_size               # in_channels = 5
        if self.encoder_input_size is not None:
            # encoder_layers = [8, 16, 32, 64]로 나옴. parsing됨.
            # encoder_activation =  leaky_relu
            
            self.encoder = ConvHeightmapEncoder(self.encoder_input_size, encoder_layers, encoder_activation)

            in_channels += self.encoder.out_features    # in_channels = 65. 원래 5였는데, self.encoder의 out_features가 60이어서 65가 됨.

        self.mlp = nn.ModuleList()

        # mlp_layers = [256,160,128]
        # 실제 action을 출력하는 policy network를 설계하는 단계.
        # Exteroception(60) + Proprioception(5)를 input으로 받음.
        # 65->256->160->128->2. 마지막 2는 action임. lin_vel, ang_vel
        for feature in mlp_layers:
            print("feature : ",feature)
            self.mlp.append(nn.Linear(in_channels, feature))
            self.mlp.append(get_activation(mlp_activation))
            in_channels = feature
        action_space = action_space.shape[0]
        self.mlp.append(nn.Linear(in_channels, action_space))
        self.mlp.append(nn.Tanh())
        self.log_std_parameter = nn.Parameter(torch.zeros(action_space))

    def compute(self, states, role="actor"):
        # Split the states into proprioception and heightmap if the heightmap is used.
        if self.encoder_input_size is None:
            x = states["states"]
        
        # Exteroception이 쓰였기 때문에, 분리를 해야함.
        # states 딕셔너리의 "states"키의 크기는 proprioception(5) + exteroception(10201) = 10206임.
        # 이때, "states"키의 첫 다섯개 원소가 proprioception이므로, 나머지 10201를 따로 exteroception으로 빼겠다는 의미.
                
        else:
            encoder_output = self.encoder(states["states"][:, self.mlp_input_size - 1:-1])  # encoder_output = 60
            
            # x라는 변수에 proprioception(5) 정보를 따로 저장함.
            x = states["states"][:, 0:self.mlp_input_size]  # x =  torch.Size([1, 5])
            
            # torch.cat = 텐서를 지정된 차원으로 이어붙이는 함수
            # dim=0 : 첫 번째 축에서 이어붙임. (행 기준)
            # dim=1 : 두 번째 축에서 이어붙임 (열 기준)
            
            # 결국, x(proprioception, 5)과 encoder_output(exteroception, 60)을 torch.cat함수로 이어줌.
            x = torch.cat([x, encoder_output], dim=1)   # x =  torch.Size([1, 65])

        # Compute the output of the MLP.
        for layer in self.mlp:
            x = layer(x)

        return x, self.log_std_parameter, {}


class DeterministicNeuralNetworkConv(DeterministicMixin, BaseModel):
    """Gaussian neural network model."""

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        mlp_input_size=4,
        mlp_layers=[256, 160, 128],
        mlp_activation="leaky_relu",
        encoder_input_size=None,
        encoder_layers=[80, 60],
        encoder_activation="leaky_relu",
        **kwargs,
    ):
        """Initialize the Gaussian neural network model.

        Args:
            observation_space (gym.spaces.Space): The observation space of the environment.
            action_space (gym.spaces.Space): The action space of the environment.
            device (torch.device): The device to use for computation.
            encoder_features (list): The number of features for each encoder layer.
            encoder_activation (str): The activation function to use for each encoder layer.
        """
        BaseModel.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=False)

        self.mlp_input_size = mlp_input_size
        self.encoder_input_size = encoder_input_size

        in_channels = self.mlp_input_size
        if self.encoder_input_size is not None:
            self.encoder = ConvHeightmapEncoder(self.encoder_input_size, encoder_layers, encoder_activation)
            in_channels += self.encoder.out_features

        self.mlp = nn.ModuleList()

        action_space = action_space.shape[0]
        for feature in mlp_layers:
            self.mlp.append(nn.Linear(in_channels, feature))
            self.mlp.append(get_activation(mlp_activation))
            in_channels = feature

        self.mlp.append(nn.Linear(in_channels, 1))

    def compute(self, states, role="actor"):
        if self.encoder_input_size is None:
            x = states["states"]
        else:
            x = states["states"][:, :self.mlp_input_size]
            encoder_output = self.encoder(states["states"][:, self.mlp_input_size - 1:-1])
            x = torch.cat([x, encoder_output], dim=1)

        for layer in self.mlp:
            x = layer(x)

        return x, {}
