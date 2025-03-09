import os

from omni.isaac.lab import sim as sim_utils
from omni.isaac.lab.assets import AssetBaseCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils.configclass import configclass

from rover_envs.envs.navigation.utils.terrains.terrain_importer import RoverTerrainImporter

# base_path = os.path.dirname(os.path.abspath(__file__))
# ground_terrain_path = os.path.join(base_path, "terrain1", "terrain_only.usd")
# obstacles_path = os.path.join(base_path, "terrain1", "rocks_merged.usd")
# hidden_terrain_path = os.path.join(base_path, "terrain1", "terrain_merged.usd")


@configclass
class MarsTerrainSceneCfg(InteractiveSceneCfg):
    """
    Mars Terrain Scene Configuration
    """
    # Hidden Terrain (merged terrain of ground and obstacles) for raycaster.
    # This is done because the raycaster doesn't work with multiple meshes
    
    ###############################
    # 
    
    # hidden_terrain = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    
    hidden_terrain = AssetBaseCfg(
        prim_path="/World/terrain/hidden_terrain",
        spawn=sim_utils.UsdFileCfg(
            visible=False,
            usd_path=os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "debug1",               # terrain1이 원본!
                # "terrain1",
                # "taehu.usdc",
                "terrain_merged.usd"    # terrain_merged.usd가 원본
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    # Ground Terrain
    terrain = TerrainImporterCfg(
        class_type=RoverTerrainImporter,
        prim_path="/World/terrain",
        terrain_type="usd",
        collision_group=-1,
        usd_path=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "debug1",                 # terrain1이 원본!
            # "terrain1",
            # "flat_terrain_only.usd",
            "terrain_only.usd"    # terrain_only.usd가 원본!
        ),
    )

    # Obstacles
    obstacles = AssetBaseCfg(
        prim_path="/World/terrain/obstacles",
        spawn=sim_utils.UsdFileCfg(
            visible=True,
            usd_path=os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "debug1",             # terrain1이 원본!
                # "terrain1",
                # "no_obstacles.usd",
                "rocks_merged.usd",   # rocks_merged.usd가 원본!
                
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )
    
    # ground = AssetBaseCfg(prim_path="/World/terrain/hidden_terrain", spawn=sim_utils.GroundPlaneCfg())
