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

class Layer(nn.Module):
    def __init__(self,in_channels,out_channels, activation_function="elu"):
        super(Layer,self).__init__()
        self.activation_functions = {
            "elu" : nn.ELU(),
            "relu" : nn.ReLU(inplace=True),
            "leakyrelu" :nn.LeakyReLU(),
            "sigmoid" : nn.Sigmoid(),
            "tanh" : nn.Tanh(),
            "relu6" : nn.ReLU6()
           } 
        self.layer = nn.Sequential(
            nn.Linear(in_channels,out_channels),
            self.activation_functions[activation_function]
        )
    def forward(self,x):
        return self.layer(x)

class Encoder(nn.Module):
    def __init__(
            self, info, cfg):
        super(Encoder,self).__init__()
        encoder_features = cfg["encoder_features"]          # [1500,1000]
        print(f"encoder_features = {encoder_features}")
        activation_function = cfg["activation_function"]    # [leakyrelu]
        
        self.encoder_layers = nn.ModuleList() 
        in_channels = info                         # encoder가 sparse면 in_channels는 441, encoder가 dense면 in_channels는 676
        for feature in encoder_features:
            self.encoder_layers.append(nn.Linear(in_channels, feature))
            self.encoder_layers.append(nn.LeakyReLU(inplace=True))
            in_channels = feature

    def forward(self, x):
        for layer in self.encoder_layers:
            x = layer(x)
        return x
    
class Belief_Encoder(nn.Module):
    def __init__(
            self, info, cfg, input_dim=120):
        super(Belief_Encoder,self).__init__()
        self.hidden_dim = cfg["hidden_dim"]                     # 300
        self.n_layers = cfg["n_layers"]                         # 2
        activation_function = cfg["activation_function"]        # leakyrelu
        proprioceptive = info["proprioceptive"]                 # 4
        input_dim = proprioceptive+input_dim                    # 4+2000=2004
        
        self.gru = nn.GRU(input_dim, self.hidden_dim, self.n_layers, batch_first=True)
        self.gb = nn.ModuleList()
        self.ga = nn.ModuleList()
        gb_features = cfg["gb_features"]                        # [128,256,512,1024]
        ga_features = cfg["ga_features"]                        # [128,256,512,1024]

        in_channels = self.hidden_dim                           # 300
        for feature in gb_features:                             # [128,128,120]
            self.gb.append(Layer(in_channels, feature, activation_function))
            in_channels = feature
        
        in_channels = self.hidden_dim                           # 300
        for feature in ga_features:                             # [128,128,120]
            self.ga.append(Layer(in_channels, feature, activation_function))
            in_channels = feature

        self.ga.append(nn.Sigmoid())

    def forward(self, p, l_e, h):
        # p = proprioceptive
        # e = exteroceptive
        # h = hidden state
        # x = input data, h = hidden state
        
        p = p.unsqueeze(1)
        
        x = torch.cat((p,l_e),dim=2)
        out, h = self.gru(x, h)
        x_b = x_a = out
        
        for layer in self.gb:
            x_b = layer(x_b)
        for layer in self.ga:
            x_a = layer(x_a)
        x_a = l_e * x_a
        # TODO IMPLEMENT GATE
        belief = x_b + x_a

        return belief, h, out

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to('cpu')
        return hidden

class Belief_Decoder(nn.Module):
    def __init__(
            self, info, cfg, n_input=50, hidden_dim=50,n_layers=2,activation_function="leakyrelu"):
        super(Belief_Decoder,self).__init__()
        exteroceptive = info["sparse"] + info["dense"]
        gate_features = cfg["gate_features"] #[128,256,512, exteroceptive]
        decoder_features = cfg["decoder_features"]#[128,256,512, exteroceptive]
        #n_input = cfg[""]
        gate_features.append(exteroceptive)
        decoder_features.append(exteroceptive)
        self.n_input = n_input
        self.gate_encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
    

        in_channels = self.n_input
        for feature in gate_features:
            self.gate_encoder.append(Layer(in_channels, feature, activation_function))
            in_channels = feature
        self.gate_encoder.append(nn.Sigmoid())  

        in_channels = self.n_input
        for feature in decoder_features:
            self.decoder.append(Layer(in_channels, feature, activation_function))
            in_channels = feature
        

    def forward(self, e, h):
        gate = h[-1]
        decoded = h[-1]
       # gate = gate.repeat(e.shape[1], 1, 1).permute(1,0,2)
       # decoded = decoded.repeat(e.shape[1], 1, 1).permute(1,0,2)
        for layer in self.gate_encoder:
            gate = layer(gate)

        for layer in self.decoder:
            decoded = layer(decoded)
        x = e*gate
        x = x + decoded
        return x
    
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(1.0)
            
class MLP(nn.Module):
    def __init__(
            self, info, cfg, belief_dim):
        super(MLP,self).__init__()
        self.mlp = nn.ModuleList()  # MLP for student policy
        proprioceptive = info["proprioceptive"]             # 4
        action_space = info["actions"]                      # 2
        activation_function = cfg["activation_function"]    # leakyrelu
        network_features = cfg["network_features"]          # [256,160,128]

        in_channels = proprioceptive + belief_dim           # 124
        for feature in network_features:
            self.mlp.append(nn.Linear(in_channels, feature))
            self.mlp.append(nn.LeakyReLU(inplace=True))
            in_channels = feature

        self.mlp.append(nn.Linear(in_channels,action_space))
        self.mlp.append(nn.Tanh())
        self.log_std_parameter = nn.Parameter(torch.zeros(action_space))

    def forward(self, p, belief):
        p = p.unsqueeze(1)
        x = torch.cat((p,belief),dim=2)
        
        for layer in self.mlp:
            x = layer(x)
        return x, self.log_std_parameter

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
        # print("in_channels = ", in_channels)                        # 10201
        # print("encoder_features = ", encoder_features)              # 8, 16, 32, 64
        # print("encoder_activation = ", encoder_activation)          # leaky_relu
        super().__init__()
        # self.heightmap_size는 rover_env_cfg.py에서 height_scanner의 size의 제곱근과 같음. Ex) resolution=0.05, size=[5.0, 5.0] 이라고 하면, 한 변이 101개이므로, self.heightmap_size는 101이 나옴.
        self.heightmap_size = torch.sqrt(torch.tensor(in_channels)).int()   # tensor(101, dtype=torch.int32)
        
        # print("self.heightmap_size = ",self.heightmap_size)         # tensor(101, dtype=torch.int32)
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
            # print("w = ", w)
            h = (flatten_size[1] - kernel_size + 2 * padding) // stride + 1
            # print("h = ", h)
            
            # Conv2D 레이어를 거치면 아래와 같이 너비와 높이가 변함.
            w = (w - kernel_size + 2 * padding) // stride + 1
            # print("w = ", w)
            h = (h - kernel_size + 2 * padding) // stride + 1
            # print("h = ", h)
            
            # Max Pooling을 거치면 아래와 같이 너비와 높이가 변함!
            w = (w - 2) // 2 + 1
            h = (h - 2) // 2 + 1
            flatten_size = [w, h]   # flatten_size :  [tensor(6, dtype=torch.int32), tensor(6, dtype=torch.int32)]
            
        self.conv_out_features = out_channels * flatten_size[0] * flatten_size[1]   # 64*6*6=tensor(2304, dtype=torch.int32)

        features = [80, 60]

        self.mlps = nn.ModuleList()
        in_channels = self.conv_out_features    # in_channels =  tensor(2304, dtype=torch.int32)
        print("isaac_rover/rover_envs/envs/navigation/learning/skrl/models.py에서 실행중\n"*10)
        print(f"in_channels = {in_channels}")
        
        for feature in features:
            print(f"")
            self.mlps.append(nn.Linear(in_channels, feature))
            self.mlps.append(get_activation(encoder_activation))
            in_channels = feature
        # Mlp : 2304 -> 80 -> 60

        self.out_features = features[-1]

    def forward(self, x):
        # x is a flattened heightmap, reshape it to 2D
        # view함수는 텐서의 shape을 변경하는 함수임.
        # 처음에 -1은 자동으로 차원을 지정하라는 의미. 즉, 뒤의 값인 1에 맞게 알아서 shape이 변경됨.
        # print("isaac_rover/rover_envs/envs/navigation/learning/skrl/models.py\n" * 20)
        # print("self.heightmap_size : ", self.heightmap_size)
        # print("x.shape : ", x.shape)  # 현재 x의 크기 확인
        # print("변경 전!")
        # print("x.shape = ",x.shape)
        x = x.view(-1, 1, self.heightmap_size, self.heightmap_size)
        # print(f"heightmap_size = {self.heightmap_size}")
        # print("%^&*(^*%*&%*&%^*(%&*(%*&%&*(%*&(%&*(%&*(())))))))")
        # print("x 출력중")
        # print(x)
        # print("변경 후!")
        # print("x.shape = ",x.shape)

        for layer in self.encoder_layers:
            x = layer(x)
            # print("x = layer(x) 결과 출력")
            # print(x.shape)

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
        mlp_input_size=4,
        mlp_layers=[512, 256, 128],
        mlp_activation="leaky_relu",
        dense_encoder_input_size=None,
        sparse_encoder_input_size=None,
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
        self.dense_encoder_input_size = dense_encoder_input_size
        self.sparse_encoder_input_size = sparse_encoder_input_size

        in_channels = self.mlp_input_size
        if self.dense_encoder_input_size is not None:
            self.dense_encoder = HeightmapEncoder(self.dense_encoder_input_size, encoder_layers, encoder_activation)
            in_channels += encoder_layers[-1]
            
        if self.sparse_encoder_input_size is not None:
            self.sparse_encoder = HeightmapEncoder(self.sparse_encoder_input_size, encoder_layers, encoder_activation)
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
        if self.dense_encoder_input_size is None:
            x = states["states"]
        else:
            dense_encoder_output = self.dense_encoder(states["states"][:, self.mlp_input_size:self.mlp_input_size+self.dense_encoder_input_size])
            sparse_encoder_output = self.sparse_encoder(states["states"][:, self.mlp_input_size+self.dense_encoder_input_size:])
            x = states["states"][:,:self.mlp_input_size]
            print(f"x = {x}")
            x = torch.cat([x, dense_encoder_output, sparse_encoder_output], dim=1)

        # Compute the output of the MLP.
        for layer in self.mlp:
            x = layer(x)

        return x, self.log_std_parameter, {}
    

class GaussianNeuralNetwork_Student(GaussianMixin, BaseModel):
    """Gaussian neural network model."""

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        mlp_input_size=5,
        mlp_layers=[512, 256, 128],
        mlp_activation="leaky_relu",
        dense_encoder_input_size=None,
        sparse_encoder_input_size=None,
        encoder_layers=[60, 20],
        encoder_activation="leaky_relu",
        student="",
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

        self.n_re = 1
        self.n_pr = 4
        self.n_sp = 441
        self.n_de = 676
        self.n_ac = 2
        encoder_layers = {
          "activation_function": "leakyrelu",
          "encoder_features": [60, 20],
        }
        cfg_belief_encoder = {
            "hidden_dim":       300,
            "n_layers":         2,
            "activation_function":  "leakyrelu",
            "gb_features": [128,256,512,1024,40],
            "ga_features": [128,256,512,1024,40]
        }
        cfg_belief_decoder = {
            "activation_function": "leakyrelu",
            "gate_features":    [1000,1500],
            "decoder_features": [1000,1500]
        }
        cfg_mlp = {"activation_function": "leakyrelu",
            "network_features": [512,256,128]}
        info = {"reset" : 1, "actions" : 2, "proprioceptive" : 4,"sparse" : 441, "dense" : 676}
        
        self.sparse_encoder = Encoder(self.n_sp, encoder_layers)
        self.dense_encoder = Encoder(self.n_de, encoder_layers)
        encoder_dim = encoder_layers["encoder_features"][-1] * 2
        self.belief_encoder = Belief_Encoder(info, cfg_belief_encoder, input_dim=encoder_dim)
        self.belief_decoder = Belief_Decoder(info, cfg_belief_decoder, cfg_belief_encoder["hidden_dim"])
        
        self.MLP = MLP(info, cfg_mlp, belief_dim=encoder_dim)
        
        student_policy = torch.load(student, weights_only=True)["state_dict"]
        
        # Filter out encoder to only maintain network MLP
        mlp_params = {k[4:]: v for k,v in student_policy.items() if (k.startswith("MLP") or "log_std_parameter" in k)}
        sparse_encoder_params = {k[15:]: v for k,v in student_policy.items() if "sparse_encoder" in k}
        dense_encoder_params = {k[14:]: v for k,v in student_policy.items() if "dense_encoder" in k}
        belief_encoder_params = {k[15:]: v for k,v in student_policy.items() if "belief_encoder" in k}
        # belief_decoder_params = {k[15:]: v for k,v in student_policy.items() if "belief_decoder" in k}
        
        # Load state dict
        self.MLP.load_state_dict(mlp_params)
        self.sparse_encoder.load_state_dict(sparse_encoder_params)
        self.dense_encoder.load_state_dict(dense_encoder_params)
        self.belief_encoder.load_state_dict(belief_encoder_params)
        
        print("student policy 출력!!!!!\n"*5)
        print(mlp_params)
        print(sparse_encoder_params)
        print(dense_encoder_params)
        print(belief_encoder_params)
        
        self.MLP.to("cuda")
        self.sparse_encoder.to("cuda")
        self.dense_encoder.to("cuda")
        self.belief_encoder.to("cuda")
        
        ####################################
        # self.mlp_input_size = mlp_input_size
        # self.dense_encoder_input_size = dense_encoder_input_size
        # self.sparse_encoder_input_size = sparse_encoder_input_size

        # in_channels = self.mlp_input_size
        # if self.dense_encoder_input_size is not None:
        #     self.dense_encoder = HeightmapEncoder(self.dense_encoder_input_size, encoder_layers, encoder_activation)
        #     in_channels += encoder_layers[-1]
            
        # if self.sparse_encoder_input_size is not None:
        #     self.sparse_encoder = HeightmapEncoder(self.sparse_encoder_input_size, encoder_layers, encoder_activation)
        #     in_channels += encoder_layers[-1]

        # self.mlp = nn.ModuleList()

        # for feature in mlp_layers:
        #     self.mlp.append(nn.Linear(in_channels, feature))
        #     self.mlp.append(get_activation(mlp_activation))
        #     in_channels = feature

        # action_space = action_space.shape[0]
        # self.mlp.append(nn.Linear(in_channels, action_space))
        # self.mlp.append(nn.Tanh())
        # self.log_std_parameter = nn.Parameter(torch.zeros(action_space))

    def compute(self, x, h, role="actor"):
        n_re = self.n_re    # 1
        n_ac = self.n_ac    # 2
        n_pr = self.n_pr    # 4
        n_sp = self.n_sp    # 441
        n_de = self.n_de    # 676
        # print(f"input_h = {h}")
        
        x = x["states"]     # shape = [num_envs, num_states]
        # print(f"n_re = {n_re}")
        # print(f"n_ac = {n_ac}")
        # print(f"n_pr = {n_pr}")
        # print(f"n_sp = {n_sp}")
        # print(f"n_de = {n_de}")
        print(f"x.shape = {x.shape}")
        
        proprioceptive = x[:,:n_pr]
        print(f"proprioception = {proprioceptive}")
        # print(f"input_distance = {proprioceptive[:,2]}")
        # print(f"input_heading = {proprioceptive[:,3]}")
        dense = x[:,n_pr:n_pr+n_de]
        sparse = x[:,n_pr+n_de:]
        # exteroceptive = torch.cat((sparse,dense),dim=2)
        
        # Pass exteroceptive information through encoder
        e_l1 = self.sparse_encoder(sparse)
        e_l2 = self.dense_encoder(dense)
        
        e_l1 = e_l1.unsqueeze(1)
        e_l2 = e_l2.unsqueeze(1)
        
        # print("encoder output shape")
        # print(e_l1.shape)
        # print(e_l2.shape)
        
        e_l = torch.cat((e_l1,e_l2), dim=2)
        
        belief, h, out = self.belief_encoder(proprioceptive,e_l,h)
        
        # estimated = self.belief_decoder(exteroceptive,out)
        
        actions, log_std = self.MLP(proprioceptive,belief)
        
        #################################################################################################
        # Split the states into proprioception and heightmap if the heightmap is used.
        # if self.dense_encoder_input_size is None:
        #     x = states["states"]
        # else:
        #     dense_encoder_output = self.dense_encoder(states["states"][:, self.mlp_input_size - 1:-1-self.sparse_encoder_input_size])
        #     sparse_encoder_output = self.sparse_encoder(states["states"][:, self.mlp_input_size + self.dense_encoder_input_size - 1:-1])
        #     x = states["states"][:, 0:self.mlp_input_size]
        #     x = torch.cat([x, dense_encoder_output, sparse_encoder_output], dim=1)

        # # Compute the output of the MLP.
        # for layer in self.mlp:
        #     x = layer(x)
        
        # print(f"output_h = {h}")

        return actions, self.MLP.log_std_parameter, h, {}


class DeterministicNeuralNetwork(DeterministicMixin, BaseModel):
    """Gaussian neural network model."""

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        mlp_input_size=4,
        mlp_layers=[512, 256, 128],
        mlp_activation="leaky_relu",
        dense_encoder_input_size=None,
        sparse_encoder_input_size=None,
        encoder_layers=[60, 20],
        encoder_activation="leaky_relu",
        student="",
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
        self.dense_encoder_input_size = dense_encoder_input_size
        self.sparse_encoder_input_size = sparse_encoder_input_size

        in_channels = self.mlp_input_size
        if self.dense_encoder_input_size is not None:
            self.dense_encoder = HeightmapEncoder(self.dense_encoder_input_size, encoder_layers, encoder_activation)
            in_channels += encoder_layers[-1]
        if self.sparse_encoder_input_size is not None:
            self.sparse_encoder = HeightmapEncoder(self.sparse_encoder_input_size, encoder_layers, encoder_activation)
            in_channels += encoder_layers[-1]

        self.mlp = nn.ModuleList()

        action_space = action_space.shape[0]
        for feature in mlp_layers:
            self.mlp.append(nn.Linear(in_channels, feature))
            self.mlp.append(get_activation(mlp_activation))
            in_channels = feature

        self.mlp.append(nn.Linear(in_channels, 1))

    def compute(self, states, role="actor"):
        if self.dense_encoder_input_size is None:
            x = states["states"]
        else:
            x = states["states"][:, :self.mlp_input_size]
            dense_encoder_output = self.dense_encoder(states["states"][:, self.mlp_input_size - 1:-1-self.sparse_encoder_input_size])
            sparse_encoder_output = self.sparse_encoder(states["states"][:, self.mlp_input_size + self.dense_encoder_input_size - 1:-1])
            x = torch.cat([x, dense_encoder_output, sparse_encoder_output], dim=1)

        for layer in self.mlp:
            x = layer(x)

        return x, {}
    

class DeterministicNeuralNetwork_Student(DeterministicMixin, BaseModel):
    """Gaussian neural network model."""

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        mlp_input_size=4,
        mlp_layers=[512, 256, 128],
        mlp_activation="leaky_relu",
        dense_encoder_input_size=None,
        sparse_encoder_input_size=None,
        encoder_layers=[60, 20],
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
        self.dense_encoder_input_size = dense_encoder_input_size
        self.sparse_encoder_input_size = sparse_encoder_input_size

        in_channels = self.mlp_input_size
        if self.dense_encoder_input_size is not None:
            self.dense_encoder = HeightmapEncoder(self.dense_encoder_input_size, encoder_layers, encoder_activation)
            in_channels += encoder_layers[-1]
        if self.sparse_encoder_input_size is not None:
            self.sparse_encoder = HeightmapEncoder(self.sparse_encoder_input_size, encoder_layers, encoder_activation)
            in_channels += encoder_layers[-1]

        self.mlp = nn.ModuleList()

        action_space = action_space.shape[0]
        for feature in mlp_layers:
            self.mlp.append(nn.Linear(in_channels, feature))
            self.mlp.append(get_activation(mlp_activation))
            in_channels = feature

        self.mlp.append(nn.Linear(in_channels, 1))

    def compute(self, states, role="actor"):
        if self.dense_encoder_input_size is None:
            x = states["states"]
        else:
            x = states["states"][:, :self.mlp_input_size]
            dense_encoder_output = self.dense_encoder(states["states"][:, self.mlp_input_size - 1:-1-self.sparse_encoder_input_size])
            sparse_encoder_output = self.sparse_encoder(states["states"][:, self.mlp_input_size + self.dense_encoder_input_size - 1:-1])
            x = torch.cat([x, dense_encoder_output, sparse_encoder_output], dim=1)

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
        mlp_layers=[512, 256, 128],
        mlp_activation="leaky_relu",
        dense_encoder_input_size=None,
        sparse_encoder_input_size=None,
        encoder_layers=[80, 60],
        encoder_activation="leaky_relu",
        **kwargs,
    ):
        print(f"observation_space = {observation_space}")
        print(f"{type(observation_space)}")
        print(f"action_space = {action_space}")
        print(f"{type(action_space)}")
        print(f"device = {device}")
        print(f"{type(device)}")
        print(f"mlp_input_size = {mlp_input_size}")
        print(f"{type(mlp_input_size)}")
        print(f"mlp_layers = {mlp_layers}")
        print(f"{type(mlp_layers)}")
        print(f"mlp_activation = {mlp_activation}")
        print(f"{type(mlp_activation)}")
        print(f"dense_encoder_input_size = {dense_encoder_input_size}")
        print(f"{type(dense_encoder_input_size)}")
        print(f"sparse_encoder_input_size = {sparse_encoder_input_size}")
        print(f"{type(sparse_encoder_input_size)}")
        print(f"encoder_layers = {encoder_layers}")
        print(f"{type(encoder_layers)}")
        print(f"encoder_activation = {encoder_activation}")
        print(f"{type(encoder_activation)}")
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
        self.dense_encoder_input_size = dense_encoder_input_size    # self.encoder_input_size = 10201
        self.sparse_encoder_input_size = sparse_encoder_input_size

        in_channels = self.mlp_input_size               # in_channels = 5
        if self.dense_encoder_input_size is not None:
            # encoder_layers = [8, 16, 32, 64]로 나옴. parsing됨.
            # encoder_activation =  leaky_relu
            
            self.dense_encoder = ConvHeightmapEncoder(self.dense_encoder_input_size, encoder_layers, encoder_activation)

            in_channels += self.dense_encoder.out_features    # in_channels = 65. 원래 5였는데, self.encoder의 out_features가 60이어서 65가 됨.
        
        if self.sparse_encoder_input_size is not None:
            # encoder_layers = [8, 16, 32, 64]로 나옴. parsing됨.
            # encoder_activation =  leaky_relu
            
            self.sparse_encoder = ConvHeightmapEncoder(self.sparse_encoder_input_size, encoder_layers, encoder_activation)

            in_channels += self.sparse_encoder.out_features    # in_channels = 125. 원래 65였는데, self.encoder의 out_features가 60이어서 125가 됨.


        self.mlp = nn.ModuleList()

        # mlp_layers = [256,160,128]
        # 실제 action을 출력하는 policy network를 설계하는 단계.
        # Exteroception(60) + Proprioception(5)를 input으로 받음.
        # 125->256->160->128->2. 마지막 2는 action임. lin_vel, ang_vel
        for feature in mlp_layers:
            # print("feature : ",feature)
            self.mlp.append(nn.Linear(in_channels, feature))
            self.mlp.append(get_activation(mlp_activation))
            in_channels = feature
        action_space = action_space.shape[0]
        self.mlp.append(nn.Linear(in_channels, action_space))
        self.mlp.append(nn.Tanh())
        self.log_std_parameter = nn.Parameter(torch.zeros(action_space))

    def compute(self, states, role="actor"):
        # print("=========================================")
        # print(f"{type(states)}")
        # for key, values in states.items():
            # print(f"key = {key}")
            # print(f"value = {values}")
            # print(f"type = {values.shape}")
        # Split the states into proprioception and heightmap if the heightmap is used.
        if self.dense_encoder_input_size is None:
            x = states["states"]
        
        # Exteroception이 쓰였기 때문에, 분리를 해야함.
        # states 딕셔너리의 "states"키의 크기는 proprioception(5) + exteroception(10201) = 10206임.
        # 이때, "states"키의 첫 다섯개 원소가 proprioception이므로, 나머지 10201를 따로 exteroception으로 빼겠다는 의미.
        else:
            dense_encoder_output = self.dense_encoder(states["states"][:, self.mlp_input_size - 1:-1-self.sparse_encoder_input_size])  # encoder_output = 60
            sparse_encoder_output = self.sparse_encoder(states["states"][:, self.mlp_input_size + self.dense_encoder_input_size - 1:-1])  # encoder_output = 60
            
            # x라는 변수에 proprioception(5) 정보를 따로 저장함.
            x = states["states"][:, 0:self.mlp_input_size]  # x =  torch.Size([1, 5])
            
            # torch.cat = 텐서를 지정된 차원으로 이어붙이는 함수
            # dim=0 : 첫 번째 축에서 이어붙임. (행 기준)
            # dim=1 : 두 번째 축에서 이어붙임 (열 기준)
            
            # 결국, x(proprioception, 5)과 encoder_output(exteroception, 60)을 torch.cat함수로 이어줌.
            x = torch.cat([x, dense_encoder_output, sparse_encoder_output], dim=1)   # x =  torch.Size([1, 125])
        
        # Compute the output of the MLP.
        for layer in self.mlp:
            x = layer(x)
            
        # print(f"compute된 결과 = {x}")
        # print(f"type of x = {type(x)}")
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
        dense_encoder_input_size=None,
        sparse_encoder_input_size=None,
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
        self.dense_encoder_input_size = dense_encoder_input_size
        self.sparse_encoder_input_size = sparse_encoder_input_size

        in_channels = self.mlp_input_size
        if self.dense_encoder_input_size is not None:
            self.dense_encoder = ConvHeightmapEncoder(self.dense_encoder_input_size, encoder_layers, encoder_activation)
            in_channels += self.dense_encoder.out_features
        if self.sparse_encoder_input_size is not None:
            self.sparse_encoder = ConvHeightmapEncoder(self.sparse_encoder_input_size, encoder_layers, encoder_activation)
            in_channels += self.sparse_encoder.out_features

        self.mlp = nn.ModuleList()

        action_space = action_space.shape[0]
        for feature in mlp_layers:
            self.mlp.append(nn.Linear(in_channels, feature))
            self.mlp.append(get_activation(mlp_activation))
            in_channels = feature

        self.mlp.append(nn.Linear(in_channels, 1))

    def compute(self, states, role="actor"):
        if self.dense_encoder_input_size is None:
            x = states["states"]
        else:
            dense_encoder_output = self.dense_encoder(states["states"][:, self.mlp_input_size - 1:-1-self.sparse_encoder_input_size])  # encoder_output = 60
            sparse_encoder_output = self.sparse_encoder(states["states"][:, self.mlp_input_size + self.dense_encoder_input_size - 1:-1])  # encoder_output = 60
            x = states["states"][:, :self.mlp_input_size]
            
            x = torch.cat([x, dense_encoder_output, sparse_encoder_output], dim=1)

        for layer in self.mlp:
            x = layer(x)

        return x, {}