# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Play + Bake to USD — based on the exact play.py flow."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os
import time

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play + Bake an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# BAKE arguments
parser.add_argument("--num_frames", type=int, default=1000, help="Number of frames to bake.")
parser.add_argument("--output", type=str, default="bakes/spot_promo.usda", help="Output USD file path.")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

# USD imports for baking
import omni.usd
from pxr import Usd, UsdGeom, Gf, UsdPhysics

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

def quat_to_matrix(pos, quat_wxyz):
    """Convert position + quaternion (w,x,y,z) to a Gf.Matrix4d."""
    w, x, y, z = float(quat_wxyz[0]), float(quat_wxyz[1]), float(quat_wxyz[2]), float(quat_wxyz[3])
    px, py, pz = float(pos[0]), float(pos[1]), float(pos[2])
    rot = Gf.Rotation(Gf.Quatd(w, x, y, z))
    mtx = Gf.Matrix4d()
    mtx.SetRotateOnly(rot)
    mtx.SetTranslateOnly(Gf.Vec3d(px, py, pz))
    return mtx

def collect_parent_world_transforms(bake_stage, env_body_paths, num_envs):
    """Pre-compute each body prim's parent inverse world transform for world→local conversion."""
    parent_transforms = []
    for env_idx in range(num_envs):
        env_parent_tfs = []
        for body_idx, bpath in enumerate(env_body_paths[env_idx]):
            if bpath is None:
                env_parent_tfs.append(None)
                continue
            prim = bake_stage.GetPrimAtPath(bpath)
            if not prim or not prim.IsValid():
                env_parent_tfs.append(None)
                continue
            parent = prim.GetParent()
            if parent and parent.IsValid() and parent.IsA(UsdGeom.Xformable):
                parent_xf = UsdGeom.Xformable(parent)
                parent_world = parent_xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                env_parent_tfs.append(parent_world.GetInverse())
            else:
                env_parent_tfs.append(Gf.Matrix4d(1.0))  # identity
        parent_transforms.append(env_parent_tfs)
    return parent_transforms

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent and bake to USD."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    # ============================================================
    # BAKE SETUP
    # ============================================================
    NUM_FRAMES = args_cli.num_frames
    OUTPUT_PATH = os.path.abspath(args_cli.output)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # Get robot reference from the unwrapped env
    raw_env = env.unwrapped
    robot = raw_env.scene["robot"]
    body_names = robot.data.body_names
    num_envs = raw_env.scene.num_envs
    sim_fps = 1.0 / (raw_env.cfg.sim.dt * raw_env.cfg.decimation)

    print(f"[BAKE] Bodies: {len(body_names)}, Envs: {num_envs}, FPS: {sim_fps}")
    print(f"[BAKE] Recording {NUM_FRAMES} frames ({NUM_FRAMES / sim_fps:.1f}s)")
    print(f"[BAKE] Output: {OUTPUT_PATH}")

    # Collect prim paths for all robot bodies in all envs
    stage = omni.usd.get_context().get_stage()
    env_body_paths = []
    for env_idx in range(num_envs):
        env_paths = []
        robot_prim = stage.GetPrimAtPath(f"/World/envs/env_{env_idx}/Robot")
        if robot_prim:
            for bname in body_names:
                found = False
                for prim in Usd.PrimRange(robot_prim):
                    if prim.GetName() == bname:
                        env_paths.append(str(prim.GetPath()))
                        found = True
                        break
                if not found:
                    env_paths.append(None)
        else:
            env_paths = [None] * len(body_names)
        env_body_paths.append(env_paths)

    matched = sum(1 for p in env_body_paths[0] if p is not None)
    print(f"[BAKE] Matched {matched}/{len(body_names)} bodies per env")

    all_transforms = []

    # ============================================================
    # SIMULATION LOOP — identical to play.py
    # ============================================================
    obs = env.get_observations()
    frame_count = 0

    print("=" * 60)
    print(f"[BAKE] Simulating {NUM_FRAMES} frames...")
    print("=" * 60)

    while simulation_app.is_running() and frame_count < NUM_FRAMES:
        with torch.inference_mode():
            # agent stepping — EXACTLY like play.py
            actions = policy(obs)
            # env stepping — EXACTLY like play.py
            obs, _, dones, _ = env.step(actions)
            # reset recurrent states — EXACTLY like play.py
            policy_nn.reset(dones)

        # BAKE: capture body transforms from GPU (world space)
        pos = robot.data.body_pos_w.cpu().clone()
        quat = robot.data.body_quat_w.cpu().clone()
        all_transforms.append(torch.cat([pos, quat], dim=-1))

        frame_count += 1
        if frame_count % 100 == 0:
            print(f"  Frame {frame_count}/{NUM_FRAMES} ({100 * frame_count / NUM_FRAMES:.0f}%)")

    # ============================================================
    # BAKE TO USD
    # ============================================================
    print("=" * 60)
    print("[BAKE] Writing to USD...")
    print("=" * 60)

    all_transforms = torch.stack(all_transforms, dim=0)  # (F, E, B, 7)
    print(f"[BAKE] Transform data: {all_transforms.shape}")

    # Export current stage as base (includes terrain, meshes, materials)
    stage.GetRootLayer().Export(OUTPUT_PATH)
    print(f"[BAKE] Exported base stage")

    # Reopen for editing
    bake_stage = Usd.Stage.Open(OUTPUT_PATH)
    bake_stage.SetStartTimeCode(0)
    bake_stage.SetEndTimeCode(NUM_FRAMES - 1)
    bake_stage.SetTimeCodesPerSecond(sim_fps)

    # Disable physics so Composer doesn't re-simulate
    for prim in Usd.PrimRange(bake_stage.GetPseudoRoot()):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Set(False)
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)

    # Pre-compute parent inverse transforms for world→local conversion
    print("[BAKE] Computing parent transforms...")
    parent_inv_transforms = collect_parent_world_transforms(bake_stage, env_body_paths, num_envs)

    # Write time-sampled transforms in LOCAL space
    baked_count = 0
    for env_idx in range(num_envs):
        for body_idx, bpath in enumerate(env_body_paths[env_idx]):
            if bpath is None:
                continue

            parent_inv = parent_inv_transforms[env_idx][body_idx]
            if parent_inv is None:
                continue

            prim = bake_stage.GetPrimAtPath(bpath)
            if not prim or not prim.IsValid():
                continue

            xf = UsdGeom.Xformable(prim)
            xf.ClearXformOpOrder()
            op = xf.AddTransformOp()

            for f in range(NUM_FRAMES):
                p = all_transforms[f, env_idx, body_idx, :3]
                q = all_transforms[f, env_idx, body_idx, 3:]

                # World-space transform from simulation
                world_mtx = quat_to_matrix(p, q)

                # Convert world → local by removing parent's world transform
                local_mtx = world_mtx * parent_inv

                op.Set(local_mtx, Usd.TimeCode(f))

            baked_count += 1

        if env_idx % 8 == 0:
            print(f"  Env {env_idx}/{num_envs}")

    bake_stage.GetRootLayer().Save()

    print("=" * 60)
    print(f"✅ DONE!")
    print(f"   File:     {OUTPUT_PATH}")
    print(f"   Size:     {os.path.getsize(OUTPUT_PATH) / (1024 * 1024):.1f} MB")
    print(f"   Prims:    {baked_count}")
    print(f"   Frames:   {NUM_FRAMES}")
    print(f"   Duration: {NUM_FRAMES / sim_fps:.1f}s @ {sim_fps:.0f} fps")
    print(f"")
    print(f"   → Open in USD Composer and scrub the timeline!")
    print("=" * 60)

    # close the simulator
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()