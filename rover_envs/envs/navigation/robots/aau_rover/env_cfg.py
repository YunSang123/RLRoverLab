from __future__ import annotations

from omni.isaac.lab.utils import configclass

import rover_envs.mdp as mdp
from rover_envs.assets.robots.aau_rover_simple import AAU_ROVER_SIMPLE_CFG
from rover_envs.envs.navigation.rover_env_cfg import RoverEnvCfg


@configclass
class AAURoverEnvCfg(RoverEnvCfg):
    """Configuration for the AAU rover environment."""

    def __post_init__(self):
        super().__post_init__()

        # Define robot
        self.scene.robot = AAU_ROVER_SIMPLE_CFG.replace(prim_path="{ENV_REGEX_NS}/rover")

        # Define parameters for the Ackermann kinematics.
        # Baseline rover의 config
        # self.actions.actions = mdp.AckermannActionCfg(
        #     asset_name="robot",
        #     wheelbase_length=0.849,
        #     middle_wheel_distance=0.894,
        #     rear_and_front_wheel_distance=0.77,
        #     wheel_radius=0.075,
        #     min_steering_radius=0.45,
        #     steering_joint_names=[".*Steer_Revolute"],
        #     drive_joint_names=[".*Drive_Continuous"],
        #     offset=-0.0135
        # )
        
        # Define parameters for the Ackermann kinematics.
        # OSR의 config
        self.actions.actions = mdp.AckermannActionCfg(
            asset_name="robot",
            d1=0.177,                                   # FL와 FR 사이의 거리의 절반
            d2=0.310,                                   # MR와 RR의 수직 거리
            d3=0.274,                                   # MR와 FR의 수직 거리
            d4=0.253,                                   # ML와 MR 사이의 거리의 절반
            wheel_radius=0.075,
            min_steering_radius=0.45,
            max_steering_radius=6.4,
            steering_joint_names=[".*Steer_Revolute"],
            drive_joint_names=[".*Drive_Continuous"],
            offset=-0.0135
        )
