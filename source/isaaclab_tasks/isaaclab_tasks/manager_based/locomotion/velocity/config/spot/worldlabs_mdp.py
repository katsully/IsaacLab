# worldlabs_mdp.py

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation

def reset_at_origin(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    spawn_height_offset: float = 0.5,
):
    """Spawn robots at origin."""
    robot: Articulation = env.scene[asset_cfg.name]
    n = len(env_ids)

    x = torch.zeros(n, device=env.device)
    y = torch.zeros(n, device=env.device)
    z = torch.full((n,), spawn_height_offset, device=env.device)

    yaw = torch.empty(n, device=env.device).uniform_(-3.14, 3.14)
    half = yaw * 0.5
    quat = torch.zeros(n, 4, device=env.device)
    quat[:, 0] = torch.cos(half)
    quat[:, 3] = torch.sin(half)

    pos = torch.stack([x, y, z], dim=1)

    root = robot.data.default_root_state[env_ids].clone()
    root[:, 0:3] = pos + env.scene.env_origins[env_ids]
    root[:, 3:7] = quat
    root[:, 7:13] = 0.0

    robot.write_root_pose_to_sim(root[:, :7], env_ids)
    robot.write_root_velocity_to_sim(root[:, 7:13], env_ids)

def fell_off(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    min_height: float = -5.0,
) -> torch.Tensor:
    """Terminate if robot falls below ground."""
    robot = env.scene[asset_cfg.name]
    return robot.data.root_pos_w[:, 2] < min_height