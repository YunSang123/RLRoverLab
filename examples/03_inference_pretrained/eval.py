import argparse
import torch
import math
import os
import random
from datetime import datetime

import gymnasium as gym
from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser("Welcome to Orbit: Omniverse Robotics Environments!")
parser.add_argument("--headless", action="store_true", default=False, help="Force display off at all times.")
parser.add_argument("--video", action="store_true", default=True, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=2000, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument("--num_envs", type=int, default=10, help="Number of environments to simulate.")
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
simulation_app = app_launcher.app

from omni.isaac.lab.envs import ManagerBasedRLEnv  # noqa: E402
from omni.isaac.lab.utils.dict import print_dict  # noqa: E402
from omni.isaac.lab.utils.io import dump_pickle, dump_yaml  # noqa: E402

# import omni.isaac.contrib_envs  # noqa: F401
# import omni.isaac.orbit_envs  # noqa: F401


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


from omni.isaac.lab_tasks.utils import parse_env_cfg  # noqa: E402
from skrl.utils import set_seed  # noqa: E402

import rover_envs.envs.navigation.robots  # noqa: E402, F401
# Import agents
from rover_envs.learning.train import get_agent  # noqa: E402
from rover_envs.utils.config import parse_skrl_cfg  # noqa: E402
from rover_envs.utils.skrl_utils import SkrlOrbitVecWrapper  # noqa: E402
from rover_envs.utils.skrl_utils import SkrlSequentialLogTrainer  # noqa: E402

from omni.isaac.lab_tasks.utils.wrappers.skrl import SkrlVecEnvWrapper

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# from omni.isaac.lab_tasks.utils.wrappers.skrl import SkrlSequentialLogTrainer  # noqa: E402


def main():
    args_cli_seed = args_cli.seed if args_cli.seed is not None else random.randint(0, 100000000)
    env_cfg = parse_env_cfg(args_cli.task, device="cuda:0" if not args_cli.cpu else "cpu", num_envs=args_cli.num_envs)
    experiment_cfg = parse_skrl_cfg(args_cli.task + f"_{args_cli.agent}")

    log_dir = log_setup(experiment_cfg, env_cfg, args_cli.agent)

    # Create the environment
    render_mode = "rgb_array" if args_cli.video else None
    env = gym.make(args_cli.task, cfg=env_cfg, headless=args_cli.headless, viewport=args_cli.video, render_mode=render_mode)
    # Check if video recording is enabled
    env = video_record(env, log_dir, args_cli.video, args_cli.video_length, args_cli.video_interval)
    # Wrap the environment
    env: ManagerBasedRLEnv = SkrlOrbitVecWrapper(env)
    set_seed(args_cli_seed if args_cli_seed is not None else experiment_cfg["seed"])

    # Get the observation and action spaces
    num_obs = env.observation_manager.group_obs_dim["policy"][0]
    num_actions = env.action_manager.action_term_dim[0]
    observation_space = gym.spaces.Box(low=-math.inf, high=math.inf, shape=(num_obs,))
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(num_actions,))

    trainer_cfg = experiment_cfg["trainer"]
    trainer_cfg["timesteps"] = 100000

    agent = get_agent(args_cli.agent, env, observation_space, action_space, experiment_cfg, conv=True)
    # Get the checkpoint path from the experiment configuration
    print(f'args_cli.task: {args_cli.task}')
    agent_policy_path = gym.spec(args_cli.task).kwargs.pop("best_model_path")
    agent_policy_path = "./t2mil.pt"
    print("agent_policy_path : ", agent_policy_path)
    
    agent.load(agent_policy_path)
    
    
    ###########################################################
    # custom 성능 평가

    # Metrics evaluation
    # num_episodes = 10
    # total_rewards = []
    # success_count = 0
    # success_times = []

    # for episode in range(num_episodes):
    #     state = env.reset()
    #     states = state[0]
    #     done = False
    #     episode_reward = 0
    #     timestep = 0
    #     print("{}번째 episode 실행중".format(episode+1))

    #     while not done:
    #         action = agent.act(states, timestep, timesteps=1000)[0]
    #         next_state, reward, done, truncated, info = env.step(action)
    #         episode_reward += reward
    #         timestep += 1
    #         state = next_state
    #         print("{}번째 timestep 실행중".format(timestep+1))

    #         if done.any():
    #             total_rewards.append(episode_reward)
    #             if "is_success" in info and info["is_success"]:
    #                 success_count += 1
    #                 success_times.append(timestep)

    # # Metrics calculation
    # avg_reward = float(sum(total_rewards)) / len(total_rewards)
    # success_rate = success_count / num_episodes
    # avg_success_time = float(sum(success_times)) / len(success_times) if success_times else None

    # # Print results
    # print(f"Average Reward: {avg_reward:.2f}")
    # print(f"Success Rate: {success_rate * 100:.2f}%")
    # print(f"Average Success Time: {avg_success_time:.2f}" if avg_success_time else "No successful episodes")

    # # Before plotting, ensure total_rewards elements are on CPU and converted to NumPy
    # total_rewards = [reward.cpu().item() if isinstance(reward, torch.Tensor) else reward for reward in total_rewards]

    # # Visualization
    # plt.figure(figsize=(12, 4))

    # # 1. Rewards
    # plt.subplot(1, 3, 1)
    # plt.plot(total_rewards, label="Reward")
    # plt.title("Episode Rewards")
    # plt.xlabel("Episode")
    # plt.ylabel("Reward")
    # plt.legend()

    # # 2. Success Rate
    # plt.subplot(1, 3, 2)
    # plt.bar(["Success", "Failure"], [success_count, num_episodes - success_count], color=["green", "red"])
    # plt.title("Success Rate")
    # plt.ylabel("Count")

    # # 3. Success Times
    # if success_times:
    #     success_times = [time.cpu().item() if isinstance(time, torch.Tensor) else time for time in success_times]
    #     plt.subplot(1, 3, 3)
    #     plt.hist(success_times, bins=10, color="blue")
    #     plt.title("Success Times")
    #     plt.xlabel("Time")
    #     plt.ylabel("Count")

    # plt.tight_layout()
    
    # output_path = "./evaluation_metrics.png"
    # plt.savefig(output_path)
    # print(f"Plot saved at: {output_path}")
    # plt.show()
    
    ###########################################################


    ########################################
    # 원래 eval.py 내용
    trainer_cfg = experiment_cfg["trainer"]
    print(trainer_cfg)

    trainer = SkrlSequentialLogTrainer(cfg=trainer_cfg, agents=agent, env=env)
    trainer.eval()
    ########################################
    

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
