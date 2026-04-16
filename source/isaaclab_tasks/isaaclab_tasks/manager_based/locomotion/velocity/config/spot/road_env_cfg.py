# road_env_cfg.py

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.envs import ViewerCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_assets.robots.spot import SPOT_ARM_CFG

from .flat_env_cfg_arm import (
    SpotFlatEnvCfg,
    SpotActionsCfg,
    SpotCommandsCfg,
    SpotObservationsCfg,
    SpotRewardsCfg,
    SpotTerminationsCfg,
    SpotEventCfg,
)

from . import road_mdp
from .road_terrain import RoadMeshTerrainCfg

ROAD_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(192.0, 72.0),
    border_width=0.0,
    num_rows=1,
    num_cols=1,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 0.0),
    use_cache=False,
    sub_terrains={
        "road": RoadMeshTerrainCfg(
            proportion=1.0,
            obj_path="/home/partnersteam2/IsaacRobotics/assets/TUSC_REMESH.obj",
        ),
    },
)

@configclass
class SpotRoadEventCfg(SpotEventCfg):
    reset_base = EventTerm(
        func=road_mdp.reset_on_road,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "spawn_height_offset": 0.08,
            "yaw_jitter": 0.3,
            "forward_vel_range": (0.0, 0.4),
        },
    )

@configclass
class SpotRoadTerminationsCfg(SpotTerminationsCfg):
    terrain_out_of_bounds = DoneTerm(
        func=road_mdp.off_road_termination,
        params={"asset_cfg": SceneEntityCfg("robot")},
        time_out=True,
    )

    fell_below = DoneTerm(
        func=road_mdp.fell_below_road,
        params={"asset_cfg": SceneEntityCfg("robot"), "margin": 2.0},
    )

@configclass
class SpotRoadRewardsCfg(SpotRewardsCfg):
    stay_on_road = RewardTermCfg(
        func=road_mdp.stay_on_road_reward,
        weight=1.5,
        params={"asset_cfg": SceneEntityCfg("robot"), "std": 1.5},
    )

    follow_road = RewardTermCfg(
        func=road_mdp.follow_road_direction_reward,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

@configclass
class SpotRoadEnvCfg(SpotFlatEnvCfg):
    """Spot walking on a 3D-scanned road loaded via terrain generator."""

    events: SpotRoadEventCfg = SpotRoadEventCfg()
    terminations: SpotRoadTerminationsCfg = SpotRoadTerminationsCfg()
    rewards: SpotRoadRewardsCfg = SpotRoadRewardsCfg()

    viewer = ViewerCfg(
        eye=(35.0, 5.0, 25.0),
        lookat=(125.0, -5.0, 9.0),
        origin_type="world",
        env_index=0,
        asset_name="robot",
    )

    def __post_init__(self):
        super().__post_init__()

        # ── timing ────────────────────────────────────────────────
        self.decimation = 10
        self.episode_length_s = 20.0
        self.sim.dt = 0.002
        self.sim.render_interval = self.decimation

        # ── robot ─────────────────────────────────────────────────
        self.scene.robot = SPOT_ARM_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )

        # ── road via terrain generator
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=ROAD_TERRAIN_CFG,
            max_init_terrain_level=0,
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.4, 0.4, 0.4),
                roughness=0.8,
            ),
            debug_vis=False,
        )

        # ── all envs at world origin ──────────────────────────────
        self.scene.env_spacing = 0.0
        self.scene.num_envs = 64

        # ── no height scanner ─────────────────────────────────────
        self.scene.height_scanner = None

        # ── disable terrain curriculum ────────────────────────────
        self.curriculum = None

        # ── contact forces sensor ─────────────────────────────────
        self.scene.contact_forces.update_period = self.sim.dt

        # ── commands ──────────────────────────────────────────────
        self.commands.base_velocity.ranges = mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.5, 1.5),
            lin_vel_y=(-0.3, 0.3),
            ang_vel_z=(-0.5, 0.5),
        )

class SpotRoadEnvCfg_PLAY(SpotRoadEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False

        if hasattr(self.events, "physics_material") and self.events.physics_material is not None:
            self.events.physics_material.params["static_friction_range"] = (1.0, 1.0)
            self.events.physics_material.params["dynamic_friction_range"] = (1.0, 1.0)

        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.ranges = mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(1.1, 1.8),
            lin_vel_y=(-0.1, 0.1),
            ang_vel_z=(-0.2, 0.2),
        )