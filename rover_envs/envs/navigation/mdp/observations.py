from __future__ import annotations
import inspect
from typing import TYPE_CHECKING
import math
import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# from isaaclab.command_generators import UniformPoseCommandGenerator

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# rover의 heading과 rover로부터 target position까지의 방향 벡터의 각도 차
# target의 orientation은 전혀 신경쓸 필요 없다!
def angle_to_target_observation(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Calculate the angle to the target."""

    # Get vector(x,y) from rover to target, in base frame of the rover.
    target_vector_b = env.command_manager.get_command(command_name)[:, :2]
    # print("observations.py")
    # print(f"target_vector_b = {target_vector_b}")
    # print(f"env.command_manager.get_command(command_name) = {env.command_manager.get_command(command_name)}")
    # Calculate the angle between the rover's heading [1, 0] and the vector to the target.
    angle = torch.atan2(target_vector_b[:, 1], target_vector_b[:, 0])
    
    heading_angle_diff = env.command_manager.get_command(command_name)[:, 3]
    # print(f"heading_angle_diff = {heading_angle_diff*180/math.pi}")
    # print(f"state_heading = {angle.unsqueeze(-1)/math.pi}")
    return angle.unsqueeze(-1)

# rover와 목표지점까지의 거리
def distance_to_target_euclidean(env: ManagerBasedRLEnv, command_name: str):
    """Calculate the euclidean distance to the target."""
    target = env.command_manager.get_command(command_name)
    target_position = target[:, :2]
    distance: torch.Tensor = torch.norm(target_position, p=2, dim=-1)
    
    # print(f"state_distance = {distance.unsqueeze(-1)*0.11}")
    
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
    
    ############################################
    # state에 nan이나 inf가 있으면 오류 발생해서 nan이나 inf를 0.0으로 치환
    if torch.isnan(sensor.data.ray_hits_w).any().item() or torch.isinf(sensor.data.ray_hits_w).any().item():
        print("ray_hits has nan or inf value, so it changed to 0.0")
        sensor.data.ray_hits_w = torch.nan_to_num(sensor.data.ray_hits_w, nan=0.0, posinf=0.0, neginf=0.0)
    
    # pos_has_nan = torch.isnan(sensor.data.pos_w).any().item()
    # pos_has_inf = torch.isinf(sensor.data.pos_w).any().item()
    # ray_has_nan = torch.isnan(sensor.data.ray_hits_w).any().item()
    # ray_has_inf = torch.isinf(sensor.data.ray_hits_w).any().item()
    
    # if pos_has_nan == True:
    #     print("observations.py에서 실행!")
    #     print(f"position has nan value")
    # if pos_has_inf == True:
    #     print("observations.py에서 실행!")
    #     print(f"position has inf value")
    # if ray_has_nan == True:
    #     print("observations.py에서 실행!")
    #     print(f"ray_hit has nan value")
    # if ray_has_inf == True:
    #     print("observations.py에서 실행!")
    #     print(f"ray_hit has inf value")
    
    # x = sensor.data.ray_hits_w[..., 0]
    # y = sensor.data.ray_hits_w[..., 1]
    # z = -sensor.data.pos_w[:, 2].unsqueeze(1) + sensor.data.ray_hits_w[..., 2] - 0.26878
    
    # x = x.to('cpu')
    # y = y.to('cpu')
    # z = z.to('cpu')
    
    # sparse
    # if len(x[0]) == 441:
    #     print("sparse")
    #     print(f"1사분면 = {z[0,20]}")
    #     print(f"2사분면 = {z[0,0]}")
    #     print(f"3사분면 = {z[0,420]}")
    #     print(f"4사분면 = {z[0,440]}")
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')

    #     ax.scatter(x,y,z, c='green', marker='o')
    #     ax.scatter(x[0,0], y[0,0], z[0,0], c='blue', marker='x')
    #     ax.scatter(x[0,20], y[0,20], z[0,20], c='red', marker='x')
    #     ax.scatter(x[0,21], y[0,21], z[0,21], c='purple', marker='x')
    #     ax.scatter(x[0,420], y[0,420], z[0,420], c='yellow', marker='x')
    #     ax.scatter(x[0,440], y[0,440], z[0,440], c='black', marker='x')
        
    #     ax.set_xlabel('X')
    #     ax.set_ylabel('Y')
    #     ax.set_zlabel('Z')
    #     plt.title("sparse_map")
        
    #     plt.savefig("sparse_map.png")
    
    # dense
    # if len(x[0]) == 676:
    #     print("dense")
    #     print(f"1사분면 = {z[0,25]}")
    #     print(f"2사분면 = {z[0,0]}")
    #     print(f"3사분면 = {z[0,650]}")
    #     print(f"4사분면 = {z[0,675]}")
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')

    #     ax.scatter(x,y,z, c='green', marker='o')
        
    #     ax.set_xlabel('X')
    #     ax.set_ylabel('Y')
    #     ax.set_zlabel('Z')
    #     plt.title("dense_map")
        
    #     plt.savefig("dense_map.png")
    

    
    return sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - 0.26878

# rover의 heading과 target pose의 orientation 사이의 각도 차
def angle_diff(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Calculate the angle difference between the rover's heading and the target."""
    # Get the angle to the target
    heading_angle_diff = env.command_manager.get_command(command_name)[:, 3]
    # print("observations.py")
    # print(f"heading_angle_diff = {heading_angle_diff*180/3.146}")
    return heading_angle_diff.unsqueeze(-1)