from gymnasium.spaces.box import Box
from isaaclab.envs import ManagerBasedRLEnv
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory

from rover_envs.learning.train.get_models import get_models
from rover_envs.utils.config import convert_skrl_cfg

import inspect

def Student_agent(experiment_cfg, observation_space: Box, action_space: Box, env: ManagerBasedRLEnv, conv, student_model_path, env_num):

    # Define memory size
    memory_size = experiment_cfg["agent"]["rollouts"]
    memory = RandomMemory(memory_size=memory_size, num_envs=env.num_envs, device=env.device)
    # Get the models
    models, h = get_models("Student", env, observation_space, action_space, conv, student_model_path, env_num)

    # Agent cfg
    agent_cfg = PPO_DEFAULT_CONFIG.copy()
    agent_cfg.update(convert_skrl_cfg(experiment_cfg["agent"]))

    # Create the agent
    # print("21412515151351351351515131353513513513513\n"*20)
    # print(f"PPO 경로 = {inspect.getfile(PPO)}")
    # PPO 경로 = /isaac-sim/kit/python/lib/python3.10/site-packages/skrl/agents/torch/ppo/ppo.py
    agent = PPO(
        models=models,
        memory=memory,
        cfg=agent_cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device,
    )

    return agent, h  # noqa R504