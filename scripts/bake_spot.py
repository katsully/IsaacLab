# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Play + Bake to USD — based on the exact play.py flow."""

import argparse
import os
import sys
import torch

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play + Bake an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during playback.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric for USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="RL agent configuration entry point.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--use_pretrained_checkpoint", action="store_true", help="Use pre-trained checkpoint from Nucleus.")
parser.add_argument("--num_frames", type=int, default=1000, help="Number of frames to bake.")
parser.add_argument("--output", type=str, default="bakes/spot_promo.usda", help="Output USD file path.")

# append RSL-RL and AppLauncher cli args
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras if recording video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---- Imports after launch ----
import gymnasium as gym
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, ManagerBasedRLEnvCfg, DirectRLEnvCfg, DirectMARLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import omni.usd
from pxr import Usd, UsdGeom, Gf, UsdPhysics

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

def quat_to_matrix(pos, quat_wxyz):
    """Convert position and wxyz quaternion to Gf.Matrix4d."""
    w, x, y, z = float(quat_wxyz[0]), float(quat_wxyz[1]), float(quat_wxyz[2]), float(quat_wxyz[3])
    px, py, pz = float(pos[0]), float(pos[1]), float(pos[2])
    rot = Gf.Rotation(Gf.Quatd(w, x, y, z))
    mtx = Gf.Matrix4d()
    mtx.SetRotateOnly(rot)
    mtx.SetTranslateOnly(Gf.Vec3d(px, py, pz))
    return mtx

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    # Setup paths and configurations
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed

    # Logic for finding the checkpoint
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # Environment setup
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    
    # Video wrapper
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(os.path.dirname(resume_path), "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Runner setup
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    # ---- Bake Preparation ----
    NUM_FRAMES = args_cli.num_frames
    OUTPUT_PATH = os.path.abspath(args_cli.output)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    raw_env = env.unwrapped
    robot = raw_env.scene["robot"]
    body_names = robot.data.body_names
    num_envs = raw_env.scene.num_envs
    sim_fps = 1.0 / (raw_env.cfg.sim.dt * raw_env.cfg.decimation)

    # Match body prim paths
    stage = omni.usd.get_context().get_stage()
    env_body_paths = []
    for env_idx in range(num_envs):
        env_paths = []
        robot_prim_path = f"/World/envs/env_{env_idx}/Robot"
        robot_prim = stage.GetPrimAtPath(robot_prim_path)
        for bname in body_names:
            found_path = None
            for prim in Usd.PrimRange(robot_prim):
                if prim.GetName() == bname:
                    found_path = str(prim.GetPath())
                    break
            env_paths.append(found_path)
        env_body_paths.append(env_paths)

    # ---- Simulation Loop ----
    all_transforms = []
    obs, _ = env.get_observations()
    
    print(f"[BAKE] Simulating {NUM_FRAMES} frames...")
    for frame_count in range(NUM_FRAMES):
        if not simulation_app.is_running():
            break
        
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            policy_nn.reset(dones)

        # Record world-space transforms
        pos = robot.data.body_pos_w.cpu().clone()
        quat = robot.data.body_quat_w.cpu().clone()
        all_transforms.append(torch.cat([pos, quat], dim=-1))

        if frame_count % 100 == 0:
            print(f"  Frame {frame_count}/{NUM_FRAMES}")

    # ---- Save to USD ----
    print(f"[BAKE] Baking to {OUTPUT_PATH}...")
    all_transforms = torch.stack(all_transforms, dim=0)
    stage.GetRootLayer().Export(OUTPUT_PATH)
    
    bake_stage = Usd.Stage.Open(OUTPUT_PATH)
    bake_stage.SetStartTimeCode(0)
    bake_stage.SetEndTimeCode(NUM_FRAMES - 1)
    bake_stage.SetTimeCodesPerSecond(sim_fps)

    # Clean up physics and set keyframes
    for prim in Usd.PrimRange(bake_stage.GetPseudoRoot()):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Set(False)
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)

    for env_idx in range(num_envs):
        for body_idx, bpath in enumerate(env_body_paths[env_idx]):
            if bpath is None: continue
            prim = bake_stage.GetPrimAtPath(bpath)
            xf = UsdGeom.Xformable(prim)
            xf.ClearXformOpOrder()
            op = xf.AddTransformOp()
            for f in range(len(all_transforms)):
                data = all_transforms[f, env_idx, body_idx]
                op.Set(quat_to_matrix(data[:3], data[3:]), Usd.TimeCode(f))

    bake_stage.GetRootLayer().Save()
    print("✅ Bake Complete!")
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()