from __future__ import annotations

from typing import TYPE_CHECKING

import torch
# Importing necessary modules from the isaaclab package
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def is_success(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    """
    Determine whether the target has been reached.

    This function checks if the rover is within a certain threshold distance from the target.
    If the target is reached, a scaled reward is returned based on the remaining time steps.
    """

    # Accessing the target's position
    target = env.command_manager.get_command(command_name)
    target_position = target[:, :2]

    angle = env.command_manager.get_command(command_name)[:, 3]

    # Calculating the distance and determining if the target is reached
    distance = torch.norm(target_position, p=2, dim=-1)
    
    # 원본 코드 : 거리 + 각도가 threshold 이내로 들어와야함.
    # return torch.where((distance < threshold) & (torch.abs(angle) < 0.1), True, False)

    # 수정된 코드 : 거리만 threshold 이내로 들어오기!
    return torch.where(distance < threshold, True, False)


def far_from_target(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    """
    Determine whether the target has been reached.

    This function checks if the rover is within a certain threshold distance from the target.
    If the target is reached, a scaled reward is returned based on the remaining time steps.
    """

    # Accessing the target's position w.r.t. the robot frame
    target = env.command_manager.get_command(command_name)
    target_position = target[:, :2]

    # Calculating the distance and determining if the target is reached
    distance = torch.norm(target_position, p=2, dim=-1)

    return torch.where(distance > threshold, True, False)


def collision_with_obstacles(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    """
    Checks for collision with obstacles.
    """
    # Accessing the contact sensor and its data
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # print(f"contact_sensor : {contact_sensor}")

    # Reshape as follows (num_envs, num_bodies, 3)
    force_matrix = contact_sensor.data.force_matrix_w.view(env.num_envs, -1, 3)

    # Calculating the force and returning true if it is above the threshold
    normalized_forces = torch.norm(force_matrix, dim=1)
    forces_active = torch.sum(normalized_forces, dim=-1) > 1

    # print(f"collision return = {torch.where(forces_active, True, False)}")
    return torch.where(forces_active, True, False)


def turn_over(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    """
    Check whether rover is turned over.
    """
    height_sensor = env.scene.sensors[sensor_cfg.name]
    condition = (abs(height_sensor.data.quat_w[:,1]) > threshold) | (abs(height_sensor.data.quat_w[:,2]) > threshold)
    # print(f"condition = {condition}")
    # torch.where을 활용하여 True 또는 False 반환
    return torch.where(condition, True, False)