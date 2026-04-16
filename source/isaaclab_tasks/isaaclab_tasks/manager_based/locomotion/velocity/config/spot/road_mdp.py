# road_mdp.py

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation

from .road_waypoints import ROAD_WAYPOINTS, ROAD_HALF_WIDTH
from .road_utils import RoadCorridor

# ── lazy singleton ────────────────────────────────────────────────────
_corridor: RoadCorridor | None = None

def _get_corridor(device: str) -> RoadCorridor:
    global _corridor
    if _corridor is None:
        _corridor = RoadCorridor(ROAD_WAYPOINTS, ROAD_HALF_WIDTH, device=device)
    return _corridor

# =====================================================================
#  RESET — spawn robots on the road
# =====================================================================
def reset_on_road(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    spawn_height_offset: float = 0.50,
    yaw_jitter: float = 0.3,
    forward_vel_range: tuple[float, float] = (0.0, 0.4),
):
    """Place each reset robot at a random point on the road, facing along it."""
    robot: Articulation = env.scene[asset_cfg.name]
    road = _get_corridor(env.device)
    n = len(env_ids)

    pos, yaw = road.sample_positions(n)
    pos[:, 2] += spawn_height_offset

    # Randomly flip 50% of robots to face the opposite direction
    flip_mask = torch.rand(n, device=env.device) > 0.5
    yaw[flip_mask] += torch.pi

    # small random yaw jitter
    yaw += (torch.rand(n, device=env.device) - 0.5) * 2 * yaw_jitter

    # quaternion (w, x, y, z)
    half = yaw * 0.5
    quat = torch.zeros(n, 4, device=env.device)
    quat[:, 0] = torch.cos(half)
    quat[:, 3] = torch.sin(half)

    # forward velocity along road
    speed = torch.empty(n, device=env.device).uniform_(*forward_vel_range)
    vx = speed * torch.cos(yaw)
    vy = speed * torch.sin(yaw)

    # assemble state — add env_origins so write_root_pose_to_sim lands correctly
    root = robot.data.default_root_state[env_ids].clone()
    root[:, 0:3] = pos + env.scene.env_origins[env_ids]
    root[:, 3:7] = quat
    root[:, 7] = vx
    root[:, 8] = vy
    root[:, 9:13] = 0.0

    robot.write_root_pose_to_sim(root[:, :7], env_ids)
    robot.write_root_velocity_to_sim(root[:, 7:13], env_ids)

# =====================================================================
#  TERMINATION — off the road corridor
# =====================================================================
def off_road_termination(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """True if the robot has left the road corridor."""
    robot = env.scene[asset_cfg.name]
    road = _get_corridor(env.device)
    return ~road.is_on_road(robot.data.root_pos_w[:, :2])

def fell_below_road(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    margin: float = 2.0,
) -> torch.Tensor:
    """True if robot fell well below the road surface."""
    robot = env.scene[asset_cfg.name]
    road = _get_corridor(env.device)
    surface_z = road.height_at(robot.data.root_pos_w[:, :2])
    return robot.data.root_pos_w[:, 2] < (surface_z - margin)

# =====================================================================
#  REWARDS
# =====================================================================
def stay_on_road_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    std: float = 1.0,
) -> torch.Tensor:
    """Gaussian reward: 1.0 at centerline, decays toward edges."""
    robot = env.scene[asset_cfg.name]
    road = _get_corridor(env.device)
    dist = road.distance_to_centerline(robot.data.root_pos_w[:, :2])
    return torch.exp(-(dist / std) ** 2)

def follow_road_direction_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward for walking in the direction the road points."""
    robot = env.scene[asset_cfg.name]
    road = _get_corridor(env.device)

    # robot velocity direction
    vel_xy = robot.data.root_lin_vel_w[:, :2]
    speed = torch.norm(vel_xy, dim=1).clamp(min=1e-6)
    vel_unit = vel_xy / speed.unsqueeze(1)

    # road tangent at robot position
    road_yaw = road.heading_at(robot.data.root_pos_w[:, :2])
    road_dir = torch.stack([torch.cos(road_yaw), torch.sin(road_yaw)], dim=1)

    # dot product: 1 = same direction, -1 = opposite
    alignment = (vel_unit * road_dir).sum(dim=1)

    # only reward when actually moving
    moving = (speed > 0.3).float()
    return alignment.clamp(min=0.0) * moving