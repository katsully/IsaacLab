# worldlabs_env_cfg.py

import os
import isaaclab.sim as sim_utils
from isaaclab.envs import ViewerCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.config.spot.mdp as spot_mdp
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

from isaaclab_assets.robots.spot import SPOT_CFG
from . import worldlabs_mdp

# ── Path to the World Labs environment USD ────────────────────────────
# This is the output of worldlabs_to_isaac.py (test_env.usd)
# The run_worldlabs_spot.sh script copies it here automatically
WORLDLABS_USD_PATH = "/home/partnersteam2/isaac_lab_spot/IsaacLab/IsaacLab/WorldLabs/test_env.usd"

# ═════════════════════════════════════════════════════════════════════
#  ACTIONS — legs only, no arm
# ═════════════════════════════════════════════════════════════════════

@configclass
class SpotNoArmActionsCfg:
    joint_pos_hip = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*_h[xy]"],
        scale=0.25,
        use_default_offset=True,
    )
    joint_pos_knee = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*_kn"],
        scale=0.5,
        use_default_offset=True,
    )

# ═════════════════════════════════════════════════════════════════════
#  COMMANDS
# ═════════════════════════════════════════════════════════════════════

@configclass
class SpotNoArmCommandsCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.1,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.5, 1.5), lin_vel_y=(-0.1, 0.1), ang_vel_z=(-0.5, 0.5)
        ),
    )

# ═════════════════════════════════════════════════════════════════════
#  OBSERVATIONS
# ═════════════════════════════════════════════════════════════════════

@configclass
class SpotNoArmObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.1, n_max=0.1)
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.1, n_max=0.1)
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.5, n_max=0.5)
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

# ═════════════════════════════════════════════════════════════════════
#  EVENTS
# ═════════════════════════════════════════════════════════════════════

@configclass
class SpotWorldLabsEventCfg:
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.3, 0.8),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="body"),
            "mass_distribution_params": (-2.5, 2.5),
            "operation": "add",
        },
    )

    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="body"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=worldlabs_mdp.reset_at_origin,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "spawn_height_offset": 0.6,
            "spread": 2.0,
        },
    )

    reset_robot_joints = EventTerm(
        func=spot_mdp.reset_joints_around_default,
        mode="reset",
        params={
            "position_range": (-0.2, 0.2),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_h[xy]", ".*_kn"]
            ),
        },
    )

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
        },
    )

# ═════════════════════════════════════════════════════════════════════
#  REWARDS — legs only
# ═════════════════════════════════════════════════════════════════════

@configclass
class SpotNoArmRewardsCfg:
    air_time = RewardTermCfg(
        func=spot_mdp.air_time_reward,
        weight=2.0,
        params={
            "mode_time": 0.15,
            "velocity_threshold": 0.5,
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
        },
    )
    base_angular_velocity = RewardTermCfg(
        func=spot_mdp.base_angular_velocity_reward,
        weight=5.0,
        params={"std": 2.0, "asset_cfg": SceneEntityCfg("robot")},
    )
    base_linear_velocity = RewardTermCfg(
        func=spot_mdp.base_linear_velocity_reward,
        weight=5.0,
        params={"std": 1.0, "ramp_rate": 0.5, "ramp_at_vel": 1.0, "asset_cfg": SceneEntityCfg("robot")},
    )
    foot_clearance = RewardTermCfg(
        func=spot_mdp.foot_clearance_reward,
        weight=0.5,
        params={
            "std": 0.05,
            "tanh_mult": 2.0,
            "target_height": 0.1,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
        },
    )
    gait = RewardTermCfg(
        func=spot_mdp.GaitReward,
        weight=10.0,
        params={
            "std": 0.1,
            "max_err": 0.2,
            "velocity_threshold": 0.5,
            "synced_feet_pair_names": (("fl_foot", "hr_foot"), ("fr_foot", "hl_foot")),
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces"),
        },
    )
    action_smoothness = RewardTermCfg(func=spot_mdp.action_smoothness_penalty, weight=-1.5)
    air_time_variance = RewardTermCfg(
        func=spot_mdp.air_time_variance_penalty,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
    )
    base_motion = RewardTermCfg(
        func=spot_mdp.base_motion_penalty, weight=-4.0, params={"asset_cfg": SceneEntityCfg("robot")}
    )
    base_orientation = RewardTermCfg(
        func=spot_mdp.base_orientation_penalty, weight=-3.0, params={"asset_cfg": SceneEntityCfg("robot")}
    )
    foot_slip = RewardTermCfg(
        func=spot_mdp.foot_slip_penalty,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "threshold": 1.0,
        },
    )
    joint_acc = RewardTermCfg(
        func=spot_mdp.joint_acceleration_penalty,
        weight=-1.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_h[xy]", ".*_kn"])},
    )
    joint_pos = RewardTermCfg(
        func=spot_mdp.joint_position_penalty,
        weight=-1.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stand_still_scale": 5.0,
            "velocity_threshold": 0.5,
        },
    )
    joint_torques = RewardTermCfg(
        func=spot_mdp.joint_torques_penalty,
        weight=-1.0e-3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    joint_vel = RewardTermCfg(
        func=spot_mdp.joint_velocity_penalty,
        weight=-1.0e-2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_h[xy]", ".*_kn"])},
    )

# ═════════════════════════════════════════════════════════════════════
#  TERMINATIONS
# ═════════════════════════════════════════════════════════════════════

@configclass
class SpotWorldLabsTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    body_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["body", ".*leg"]), "threshold": 1.0},
    )
    fell_off = DoneTerm(
        func=worldlabs_mdp.fell_off,
        params={"asset_cfg": SceneEntityCfg("robot"), "min_height": -5.0},
        time_out=True,
    )

# ═════════════════════════════════════════════════════════════════════
#  MAIN CONFIG
# ═════════════════════════════════════════════════════════════════════

@configclass
class SpotWorldLabsEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Standard Spot (no arm) in a World Labs generated environment."""

    observations: SpotNoArmObservationsCfg = SpotNoArmObservationsCfg()
    actions: SpotNoArmActionsCfg = SpotNoArmActionsCfg()
    commands: SpotNoArmCommandsCfg = SpotNoArmCommandsCfg()
    rewards: SpotNoArmRewardsCfg = SpotNoArmRewardsCfg()
    terminations: SpotWorldLabsTerminationsCfg = SpotWorldLabsTerminationsCfg()
    events: SpotWorldLabsEventCfg = SpotWorldLabsEventCfg()

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
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.friction_combine_mode = "multiply"
        self.sim.physics_material.restitution_combine_mode = "multiply"

        # ── standard Spot (no arm) — 12 joints ───────────────────
        self.scene.robot = SPOT_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )

        # ── World Labs scene USD (output of worldlabs_to_isaac.py)
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="usd",
            usd_path=WORLDLABS_USD_PATH,
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