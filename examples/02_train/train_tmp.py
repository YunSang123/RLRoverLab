import argparse
import math
import os
import random
from datetime import datetime

import carb
import gymnasium as gym
from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser("Welcome to Orbit: Omniverse Robotics Environments!")
parser.add_argument("--headless", action="store_true", default=False, help="Force display off at all times.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument("--num_envs", type=int, default=128, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="AAURoverEnv-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--agent", type=str, default="PPO", help="Name of the agent.")
args_cli = parser.parse_args()

# launch the simulator
config = {"headless": args_cli.headless}
# load cheaper kit config in headless
if args_cli.headless:
    app_experience = f"{os.environ['EXP_PATH']}/omni.isaac.sim.python.gym.headless.kit"
else:
    app_experience = f"{os.environ['EXP_PATH']}/omni.isaac.sim.python.kit"

app_launcher = AppLauncher(launcher_args=args_cli, experience=app_experience)
from omni.isaac.lab_tasks.utils.wrappers.skrl import SkrlVecEnvWrapper, process_skrl_cfg
simulation_app = app_launcher.app

carb_settings = carb.settings.get_settings()
carb_settings.set_bool(
    "rtx/raytracing/cached/enabled",
    False,
)
carb_settings.set_int(
    "rtx/descriptorSets",
    8192,
)
from omni.isaac.lab.envs import ManagerBasedRLEnv  # noqa: E402
from omni.isaac.lab.utils.dict import print_dict  # noqa: E402
from omni.isaac.lab.utils.io import dump_pickle, dump_yaml  # noqa: E402


def log_setup(experiment_cfg, env_cfg, agent):
    """
    Setup the logging for the experiment.

    Note:
        Copied from the ORBIT framework.
    """
    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")

    # specify directory for logging runs
    log_dir = datetime.now().strftime("%b%d_%H-%M-%S")
    if experiment_cfg["agent"]["experiment"]["experiment_name"]:
        log_dir = f'_{experiment_cfg["agent"]["experiment"]["experiment_name"]}'

    log_dir += f"_{agent}"

    # set directory into agent config
    experiment_cfg["agent"]["experiment"]["directory"] = log_root_path
    experiment_cfg["agent"]["experiment"]["experiment_name"] = log_dir

    # update log_dir
    log_dir = os.path.join(log_root_path, log_dir)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), experiment_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), experiment_cfg)
    return log_dir


def video_record(env: ManagerBasedRLEnv, log_dir: str, video: bool, video_length: int, video_interval: int) -> ManagerBasedRLEnv:
    """
    Function to check and setup video recording.

    Note:
        Copied from the ORBIT framework.

    Args:
        env (ManagerBasedRLEnv): The environment.
        log_dir (str): The log directory.
        video (bool): Whether or not to record videos.
        video_length (int): The length of the video (in steps).
        video_interval (int): The interval between video recordings (in steps).

    Returns:
        ManagerBasedRLEnv: The environment.
    """

    if video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos"),
            "step_trigger": lambda step: step % video_interval == 0,
            "video_length": video_length,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        return gym.wrappers.RecordVideo(env, **video_kwargs)

    return env


from omni.isaac.lab_tasks.utils import parse_env_cfg  # noqa: E402
from skrl.utils import set_seed  # noqa: E402, F401

import rover_envs.envs.navigation.robots  # noqa: E402, F401
# Import agents
from rover_envs.learning.train import get_agent  # noqa: E402
from rover_envs.utils.config import parse_skrl_cfg  # noqa: E402
from rover_envs.utils.skrl_utils import SkrlOrbitVecWrapper  # noqa: E402
from rover_envs.utils.skrl_utils import SkrlSequentialLogTrainer  # noqa: E402


# def train():
#     args_cli_seed = args_cli.seed if args_cli.seed is not None else random.randint(0, 100000000)
#     env_cfg = parse_env_cfg(args_cli.task, device="cuda:0" if not args_cli.cpu else "cpu", num_envs=args_cli.num_envs)
#     experiment_cfg = parse_skrl_cfg(args_cli.task + f"_{args_cli.agent}")

#     log_dir = log_setup(experiment_cfg, env_cfg, args_cli.agent)

#     # Create the environment
#     render_mode = "rgb_array" if args_cli.video else None
#     env = gym.make(args_cli.task, cfg=env_cfg, headless=args_cli.headless,
#                    viewport=args_cli.video, render_mode=render_mode)
#     # Check if video recording is enabled
#     env = video_record(env, log_dir, args_cli.video, args_cli.video_length, args_cli.video_interval)
#     # Wrap the environment
#     env: ManagerBasedRLEnv = SkrlOrbitVecWrapper(env)
#     set_seed(args_cli_seed if args_cli_seed is not None else experiment_cfg["seed"])

#     # Get the observation and action spaces
#     num_obs = env.unwrapped.observation_manager.group_obs_dim["policy"][0]
#     num_actions = env.unwrapped.action_manager.action_term_dim[0]
#     observation_space = gym.spaces.Box(low=-math.inf, high=math.inf, shape=(num_obs,))
#     action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(num_actions,))

#     trainer_cfg = experiment_cfg["trainer"]

#     agent = get_agent(args_cli.agent, env, env.observation_space, env.action_space, experiment_cfg, conv=True)
#     trainer = SkrlSequentialLogTrainer(cfg=trainer_cfg, agents=agent, env=env)
#     trainer.train()

#     env.close()
#     simulation_app.close()
from skrl.trainers.torch import SequentialTrainer
def train():
    args_cli_seed = args_cli.seed if args_cli.seed is not None else random.randint(0, 100000000)
    env_cfg = parse_env_cfg(args_cli.task, device="cuda:0" if not args_cli.cpu else "cpu", num_envs=args_cli.num_envs)
    experiment_cfg = parse_skrl_cfg(args_cli.task + f"_{args_cli.agent}")

    log_dir = log_setup(experiment_cfg, env_cfg, args_cli.agent)

    # Create the environment
    render_mode = "rgb_array" if args_cli.video else None
    env = gym.make(args_cli.task, cfg=env_cfg, headless=args_cli.headless,
                   viewport=args_cli.video, render_mode=render_mode)
    # Check if video recording is enabled
    env = video_record(env, log_dir, args_cli.video, args_cli.video_length, args_cli.video_interval)
    # Wrap the environment
    #env: ManagerBasedRLEnv = SkrlOrbitVecWrapper(env)
    env = SkrlVecEnvWrapper(env, ml_framework="torch") 
    set_seed(args_cli_seed if args_cli_seed is not None else experiment_cfg["seed"])

    # Get the observation and action spaces
    num_obs = env.unwrapped.observation_manager.group_obs_dim["policy"][0]
    num_actions = env.unwrapped.action_manager.action_term_dim[0]
    observation_space = gym.spaces.Box(low=-math.inf, high=math.inf, shape=(num_obs,))
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(num_actions,))
    print(f"Observation space: {observation_space.shape}")
    print(f'Action space: {action_space.shape}')
    print(f'num envs: {env.num_envs}')
    print(f'env obs space: {env.observation_space}')
    print(f'env action space: {env.action_space}')
    #exit()
    trainer_cfg = experiment_cfg["trainer"]

    agent = get_agent(args_cli.agent, env, observation_space, action_space, experiment_cfg, conv=True)
    trainer = SequentialTrainer(cfg=trainer_cfg, agents=agent, env=env)
    trainer.train()

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    train()