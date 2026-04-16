# worldlabs_env_cfg.py

import isaaclab.sim as sim_utils
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

from . import worldlabs_mdp

@configclass
class SpotWorldLabsEventCfg(SpotEventCfg):
    reset_base = EventTerm(
        func=worldlabs_mdp.reset_at_origin,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "spawn_height_offset": 0.7,
        },
    )

@configclass
class SpotWorldLabsTerminationsCfg(SpotTerminationsCfg):
    terrain_out_of_bounds = DoneTerm(
        func=worldlabs_mdp.fell_off,
        params={"asset_cfg": SceneEntityCfg("robot"), "min_height": -5.0},
        time_out=True,
    )

@configclass
class SpotWorldLabsEnvCfg(SpotFlatEnvCfg):
    """Spot walking in a World Labs generated environment."""

    events: SpotWorldLabsEventCfg = SpotWorldLabsEventCfg()
    terminations: SpotWorldLabsTerminationsCfg = SpotWorldLabsTerminationsCfg()

    viewer = ViewerCfg(
        eye=(5.0, 5.0, 3.0),
        lookat=(0.0, 0.0, 0.0),
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

        # ── World Labs scene as terrain ───────────────────────────
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="usd",
            usd_path="/home/partnersteam2/WorldLab/WorldLabs/stage/TEST_COL_BUILD.usd",
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
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

class SpotWorldLabsEnvCfg_PLAY(SpotWorldLabsEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 8
        self.observations.policy.enable_corruption = False

        self.events.physics_material.params["static_friction_range"] = (1.0, 1.0)
        self.events.physics_material.params["dynamic_friction_range"] = (1.0, 1.0)

        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.ranges = mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(1.0, 1.5),
            lin_vel_y=(-0.2, 0.2),
            ang_vel_z=(-0.3, 0.3),
        )