from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster

# from isaaclab.command_generators import UniformPoseCommandGenerator

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def angle_to_target_observation(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Calculate the angle to the target."""

    # Get vector(x,y) from rover to target, in base frame of the rover.
    target_vector_b = env.command_manager.get_command(command_name)[:, :2]

    # Calculate the angle between the rover's heading [1, 0] and the vector to the target.
    angle = torch.atan2(target_vector_b[:, 1], target_vector_b[:, 0])

    return angle.unsqueeze(-1)


def distance_to_target_euclidean(env: ManagerBasedRLEnv, command_name: str):
    """Calculate the euclidean distance to the target."""
    target = env.command_manager.get_command(command_name)
    target_position = target[:, :2]
    distance: torch.Tensor = torch.norm(target_position, p=2, dim=-1)
    return distance.unsqueeze(-1)


def height_scan_rover(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Calculate the height scan of the rover.

    This function uses a ray caster to generate a height scan of the rover's surroundings.
    The height scan is normalized by the maximum range of the ray caster.
    """
    # extract the used quantities (to enable type-hinting)
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    # height scan: height = sensor_height - hit_point_z - 0.26878
    # Note: 0.26878 is the distance between the sensor and the rover's base
    # 0.26878은 sensor와 rover의 발바닥면 사이의 거리
    
    # print("\nisaac_rover/rover_envs/evs/navigation/mdp/observations.py에서 실행")
    # print("sensor.data.pos_w = ", sensor.data.pos_w)
    # print("sensor.data.ray_hits_w", sensor.data.ray_hits_w)
    # print("Height : ", sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - 0.26878)
    
    ######################
    # Debugging 중
    # print("🚀 Debugging sensor.data...")
    # print("sensor.data:", sensor.data)
    # if sensor.data is not None:
    #     print("sensor.data.pos_w:", getattr(sensor.data, "pos_w", "❌ None"))
    #     print("sensor.data.ray_hits_w:", getattr(sensor.data, "ray_hits_w", "❌ None"))
    # else:
    #     print("❌ sensor.data is None!")
    ######################
    # print(f"sensor.data : {sensor.data}")
    
    sensor.data.ray_hits_w = torch.nan_to_num(sensor.data.ray_hits_w, posinf=5.0, neginf=-5.0)
    
    pos_has_nan = torch.isnan(sensor.data.pos_w).any().item()
    pos_has_inf = torch.isinf(sensor.data.pos_w).any().item()
    ray_has_nan = torch.isnan(sensor.data.ray_hits_w).any().item()
    ray_has_inf = torch.isinf(sensor.data.ray_hits_w).any().item()
    
    
    if pos_has_nan == True:
        print("observations.py에서 실행!")
        print(f"position has nan value")
    if pos_has_inf == True:
        print("observations.py에서 실행!")
        print(f"position has inf value")
    if ray_has_nan == True:
        print("observations.py에서 실행!")
        print(f"ray_hit has nan value")
    if ray_has_inf == True:
        print("observations.py에서 실행!")
        print(f"ray_hit has inf value")
    
    a = sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - 0.26878
    # print("========================================================================")
    # print(f"sensor_data = {a}")
    # print(f"shape = {a.shape}")
    # print(f"type = {type(a)}")
    
    return sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - 0.26878


def angle_diff(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Calculate the angle difference between the rover's heading and the target."""
    # Get the angle to the target
    heading_angle_diff = env.command_manager.get_command(command_name)[:, 3]

    return heading_angle_diff.unsqueeze(-1)