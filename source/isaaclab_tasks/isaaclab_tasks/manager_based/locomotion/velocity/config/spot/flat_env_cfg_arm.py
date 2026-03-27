# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.envs import ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.config.spot.mdp as spot_mdp
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

# from isaaclab.assets import RigidObjectCfg
# from .custom_terrains import RoughWithPlinthsTerrainCfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.spot import SPOT_CFG, SPOT_ARM_CFG  # isort: skip


# COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
#   size=(8.0, 8.0),
#   border_width=20.0,
#   num_rows=9,
#   num_cols=21,
#   horizontal_scale=0.1,
#   vertical_scale=0.005,
#   slope_threshold=0.75,
#   difficulty_range=(0.0, 1.0),
#   use_cache=False,
#   sub_terrains={
#       # Easy start - flat floor, no plinths
#       "flat": terrain_gen.MeshPlaneTerrainCfg(
#           proportion=0.1,
#       ),
#       # Medium - gentle rough terrain WITH plinths
#       "rough_plinths_easy": RoughWithPlinthsTerrainCfg(
#           proportion=0.45,
#           noise_range=(0.02, 0.05),   # gentle roughness
#           noise_step=0.02,
#           obstacle_width_range=(0.5, 0.7),
#           obstacle_height_range=(0.9, 1.2),
#           num_obstacles=2,
#           platform_width=2.0,
#       ),
#       # Hard - very rough terrain WITH plinths
#       "rough_plinths_hard": RoughWithPlinthsTerrainCfg(
#           proportion=0.45,
#           noise_range=(0.05, 0.15),   # much rougher
#           noise_step=0.02,
#           obstacle_width_range=(0.4, 0.8),
#           obstacle_height_range=(0.9, 1.5),
#           num_obstacles=3,            # more plinths
#           platform_width=1.5,         # smaller safe zone
#       ),
#   },
# )



COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(# ✅ NO flat platform - terrain is rough throughout
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=9,
    num_cols=21,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.2),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.6, noise_range=(0.02, 0.08), noise_step=0.02, border_width=0.25
        ),
         # Isolated pillars on flat floor - no holes
        "obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=0.9,
            obstacle_height_mode="fixed",
            obstacle_width_range=(0.5, 0.7),
            obstacle_height_range=(0.9, 1.2),
            num_obstacles=2,
            platform_width=3.0,
            
        ),
    },
)

# # We are building a custom generator using the exact 'boxes' from the source code
# COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
#     size=(8.0, 8.0),
#     border_width=20.0,
#     num_rows=10,
#     num_cols=10,
#     horizontal_scale=0.1,
#     vertical_scale=0.005,
#     slope_threshold=0.75,
#     use_cache=False,
#     sub_terrains={
#         # 10% Flat safe-zone to spawn
#         "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.1),
        
#         # 90% EXACT BOXES FROM DEMO SOURCE
#         "boxes": terrain_gen.MeshRandomGridTerrainCfg(
#             proportion=0.9, 
#             grid_width=0.45, 
#             grid_height_range=(0.05, 0.2), 
#             platform_width=2.0
#         ),
#     },
# )

@configclass
class SpotActionsCfg:
    """Action specifications for the MDP."""

    joint_pos_hip = mdp.JointPositionActionCfg(
      asset_name="robot",
      joint_names=[".*_h[xy]"],
      scale=0.2,
      use_default_offset=True
  )

    #knees sepereated to allow more deviation
    joint_pos_knee = mdp.JointPositionActionCfg(
      asset_name="robot",
      joint_names=[".*_kn"],
      scale=0.5,
      use_default_offset=True
  )


    joint_pos_arm = mdp.JointPositionActionCfg(
      asset_name="robot",
      joint_names=[
          "arm0_sh0", "arm0_sh1", "arm0_el0", "arm0_el1",
          "arm0_wr0", "arm0_wr1", "arm0_f1x"
      ],
      scale=0.0,              # ← zero scale = arm frozen at filming position
      use_default_offset=True # ← target = default_joint_pos + 0.0 * action
  )

@configclass
class SpotCommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.1,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.5, 1.0), lin_vel_y=(-0.1, 0.1), ang_vel_z=(-0.5, 0.5)
        ),
    )


@configclass
class SpotObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # `` observation terms (order preserved)
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

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class SpotEventCfg:
    """Configuration for randomization."""

    # startup
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

    # reset
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
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.5),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (-0.2, 0.2),
                "pitch": (-0.2, 0.2),
                "yaw": (-0.5, 0.5),
            },
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
    reset_arm_joints = EventTerm(
      func=spot_mdp.reset_joints_around_default,
      mode="reset",
      params={
          "position_range": (0.0, 0.0),   # ← zero jitter, exact default
          "velocity_range": (0.0, 0.0),   # ← zero velocity
          "asset_cfg": SceneEntityCfg(
              "robot",
              joint_names=[
                  "arm0_sh0",
                  "arm0_sh1",
                  "arm0_el0",
                  "arm0_el1",
                  "arm0_wr0",
                  "arm0_wr1",
                  "arm0_f1x",
              ]
          ),
      },
  )


    # # NEW: Randomize the UR10 Mount position on every reset
    # reset_obstacle_position = EventTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("obstacle"), 
    #         "pose_range": {"x": (1.0, 4.0), "y": (-2.0, 2.0), "yaw": (-3.14, 3.14)}, 
    #         "velocity_range": {},
    #     },
    # )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
        },
    )


@configclass
class SpotRewardsCfg:
    # -- task
    air_time = RewardTermCfg(
        func=spot_mdp.air_time_reward,
        weight=2.0,
        params={
            "mode_time": 0.08,
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
        weight=15.0,
        params={
            "std": 0.1,
            "max_err": 0.2,
            "velocity_threshold": 0.5,
            "synced_feet_pair_names": (("fl_foot", "hr_foot"), ("fr_foot", "hl_foot")),
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces"),
        },
    )

    # arm position in stow state
    joint_deviation_hip = RewardTermCfg(
        func=spot_mdp.joint_deviation_l1,
        weight=-5.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "arm0_sh1",
                    "arm0_el0",
                    "arm0_el1",
                    "arm0_sh0",
                    "arm0_wr0",
                    "arm0_wr1",
                    "arm0_f1x",
                ],
            )
        },
    )

    # -- penalties
    action_smoothness = RewardTermCfg(func=spot_mdp.action_smoothness_penalty, weight=-2.5)
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
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_h[xy]",
                    "arm0_sh1",
                    "arm0_el0",
                    "arm0_el1",
                    "arm0_sh0",
                    "arm0_wr0",
                    "arm0_wr1",
                    "arm0_f1x",
                ],
            )
        },
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
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_h[xy]",
                    "arm0_sh1",
                    "arm0_el0",
                    "arm0_el1",
                    "arm0_sh0",
                    "arm0_wr0",
                    "arm0_wr1",
                    "arm0_f1x",
                ],
            )
        },
    )


@configclass
class SpotTerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    body_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["body", ".*leg"]), "threshold": 1.0},
    )
    terrain_out_of_bounds = DoneTerm(
        func=mdp.terrain_out_of_bounds,
        params={"asset_cfg": SceneEntityCfg("robot"), "distance_buffer": 3.0},
        time_out=True,
    )


@configclass
class SpotFlatEnvCfg(LocomotionVelocityRoughEnvCfg):

    # Basic settings'
    observations: SpotObservationsCfg = SpotObservationsCfg()
    actions: SpotActionsCfg = SpotActionsCfg()
    commands: SpotCommandsCfg = SpotCommandsCfg()

    # MDP setting
    rewards: SpotRewardsCfg = SpotRewardsCfg()
    terminations: SpotTerminationsCfg = SpotTerminationsCfg()
    events: SpotEventCfg = SpotEventCfg()

    # Viewer
    viewer = ViewerCfg(eye=(10.5, 10.5, 0.3), origin_type="world", env_index=0, asset_name="robot")

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # general settings
        self.decimation = 10  # 50 Hz
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.002  # 500 Hz
        self.sim.render_interval = self.decimation
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.friction_combine_mode = "multiply"
        self.sim.physics_material.restitution_combine_mode = "multiply"
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        self.scene.contact_forces.update_period = self.sim.dt

        # switch robot to Spot-d
        self.scene.robot = SPOT_ARM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # terrain
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=COBBLESTONE_ROAD_CFG,
            max_init_terrain_level=COBBLESTONE_ROAD_CFG.num_rows - 1,
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
            visual_material=sim_utils.MdlFileCfg(
                mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
                project_uvw=True,
                texture_scale=(0.25, 0.25),
            ),
            debug_vis=False,
        )
        # # NEW: The UR10 Mount Object (Larger & Static)
        # self.scene.obstacle = RigidObjectCfg(
        #     prim_path="{ENV_REGEX_NS}/Obstacle",
        #     spawn=sim_utils.UsdFileCfg(
        #         usd_path="/home/partnersteam2/IsaacRobotics/assets/Collected_ur10_mount/ur10_mount.usd",
        #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
        #             kinematic_enabled=True  # Freezes the object in place (acts like a solid wall)
        #         ),
        #         scale=(1.8, 1.8, 1.8), # Change this to 3.0 or 4.0 if you need it even bigger!
        #     ),

        #     init_state=RigidObjectCfg.InitialStateCfg(pos=(1.0, 0.0, 1.0)), 
        # )

        # no height scan
        self.scene.height_scanner = None


class SpotFlatEnvCfg_PLAY(SpotFlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None

        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        #self.events.base_external_force_torque = None
        #self.events.push_robot = None
        self.commands.base_velocity.rel_standing_envs = 0.0  # no standing
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.ranges = mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.8, 1.0),   # consistent walking speed
            lin_vel_y=(-0.5, 0.5),   # no sideways
            ang_vel_z=(-0.5, 0.5),   # no turning
        )
