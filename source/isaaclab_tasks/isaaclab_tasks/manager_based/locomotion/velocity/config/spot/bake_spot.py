"""
Bake Spot simulation to USD for Composer.
Works WITH Fabric enabled - reads transforms from GPU tensors.
"""

import argparse
import os
import torch
import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-Spot-Play-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--num_frames", type=int, default=500)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--output", type=str, default="spot_promo.usda")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ---- Imports after launch ----
import gymnasium as gym
import omni.usd
from pxr import Usd, UsdGeom, Gf, UsdPhysics

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg
import isaaclab_tasks  # registers gym tasks

from isaaclab_tasks.manager_based.locomotion.velocity.config.spot.flat_env_cfg import (
    SpotFlatEnvCfg_PLAY,
)

def quat_to_matrix(pos, quat_wxyz):
    """Convert position + quaternion (w,x,y,z) to a Gf.Matrix4d."""
    w, x, y, z = float(quat_wxyz[0]), float(quat_wxyz[1]), float(quat_wxyz[2]), float(quat_wxyz[3])
    px, py, pz = float(pos[0]), float(pos[1]), float(pos[2])

    rot = Gf.Rotation(Gf.Quatd(w, x, y, z))
    mtx = Gf.Matrix4d()
    mtx.SetRotateOnly(rot)
    mtx.SetTranslateOnly(Gf.Vec3d(px, py, pz))
    return mtx

def main():
    # ============================================================
    # PHASE 1: Run simulation, collect ALL body transforms
    # ============================================================
    print("=" * 60)
    print("PHASE 1: Running simulation and collecting transforms...")
    print("=" * 60)

    cfg = SpotFlatEnvCfg_PLAY()
    cfg.scene.num_envs = args.num_envs

    env = ManagerBasedRLEnv(cfg=cfg)
    robot = env.scene["robot"]

    # Get body info
    body_names = robot.data.body_names
    num_bodies = len(body_names)
    print(f"[INFO] Robot bodies ({num_bodies}): {body_names}")

    # Get env prim paths to reconstruct USD paths later
    num_envs = env.scene.num_envs
    fps = 1.0 / (cfg.sim.dt * cfg.decimation)
    print(f"[INFO] {num_envs} envs, {args.num_frames} frames @ {fps:.0f} fps")

    # Load trained policy
    policy_path = os.path.join(os.path.dirname(args.checkpoint), "exported_policy.pt")
    if os.path.exists(policy_path):
        print(f"[INFO] Loading exported policy: {policy_path}")
        policy = torch.jit.load(policy_path, map_location=env.device)
    else:
        # Try loading the raw checkpoint as a JIT model
        print(f"[INFO] Loading checkpoint: {args.checkpoint}")
        policy = torch.jit.load(args.checkpoint, map_location=env.device)

    # Storage for all frames: (num_frames, num_envs, num_bodies, 7) -> pos(3) + quat(4)
    all_transforms = []

    obs, _ = env.reset()

    for frame in range(args.num_frames):
        with torch.no_grad():
            actions = policy(obs["policy"])

        obs, _, _, _, _ = env.step(actions)

        # Read body transforms from GPU tensors
        # body_pos_w: (num_envs, num_bodies, 3)
        # body_quat_w: (num_envs, num_bodies, 4) in (w, x, y, z)
        pos = robot.data.body_pos_w.cpu().clone()    # (E, B, 3)
        quat = robot.data.body_quat_w.cpu().clone()  # (E, B, 4)

        # Concatenate into (E, B, 7)
        frame_data = torch.cat([pos, quat], dim=-1)
        all_transforms.append(frame_data)

        if frame % 50 == 0:
            print(f"  Simulating... frame {frame}/{args.num_frames}")

    # Stack: (num_frames, num_envs, num_bodies, 7)
    all_transforms = torch.stack(all_transforms, dim=0)
    print(f"[INFO] Collected transforms: {all_transforms.shape}")

    # Save the stage path info before closing
    stage = omni.usd.get_context().get_stage()

    # Collect the actual USD prim paths for each env's robot bodies
    # Pattern: /World/envs/env_{i}/Robot/{body_name}
    env_body_paths = []
    for env_idx in range(num_envs):
        env_paths = []
        env_prim_path = f"/World/envs/env_{env_idx}"
        robot_prim = stage.GetPrimAtPath(f"{env_prim_path}/Robot")

        if robot_prim:
            # Find actual body prims by name
            for body_name in body_names:
                found = False
                for prim in Usd.PrimRange(robot_prim):
                    if prim.GetName() == body_name:
                        env_paths.append(str(prim.GetPath()))
                        found = True
                        break
                if not found:
                    env_paths.append(None)
        else:
            env_paths = [None] * num_bodies

        env_body_paths.append(env_paths)

    # Also grab the full stage as a starting point for our output
    source_layer = stage.GetRootLayer().identifier
    print(f"[INFO] Source stage: {source_layer}")

    # ============================================================
    # PHASE 2: Write transforms into USD as time-samples
    # ============================================================
    print("=" * 60)
    print("PHASE 2: Baking transforms into USD...")
    print("=" * 60)

    # Flatten the live stage to a new file first
    output_path = os.path.abspath(args.output)
    stage.GetRootLayer().Export(output_path)
    print(f"[INFO] Exported base stage to {output_path}")

    # Re-open the exported stage for editing
    bake_stage = Usd.Stage.Open(output_path)
    bake_stage.SetStartTimeCode(0)
    bake_stage.SetEndTimeCode(args.num_frames - 1)
    bake_stage.SetTimeCodesPerSecond(fps)

    # Disable physics on the baked stage so Composer doesn't re-simulate
    for prim in Usd.PrimRange(bake_stage.GetPseudoRoot()):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rb = UsdPhysics.RigidBodyAPI(prim)
            rb.GetRigidBodyEnabledAttr().Set(False)

    # Write time-sampled transforms
    total_prims = 0
    for env_idx in range(num_envs):
        for body_idx, body_path in enumerate(env_body_paths[env_idx]):
            if body_path is None:
                continue

            prim = bake_stage.GetPrimAtPath(body_path)
            if not prim or not prim.IsValid():
                continue

            xformable = UsdGeom.Xformable(prim)

            # Clear existing xform ops and add a single transform op
            xformable.ClearXformOpOrder()
            xform_op = xformable.AddTransformOp()

            # Write every frame as a time-sample
            for frame in range(args.num_frames):
                pos = all_transforms[frame, env_idx, body_idx, :3]
                quat = all_transforms[frame, env_idx, body_idx, 3:]  # w,x,y,z
                mtx = quat_to_matrix(pos, quat)
                xform_op.Set(mtx, Usd.TimeCode(frame))

            total_prims += 1

        if env_idx % 8 == 0:
            print(f"  Baking env {env_idx}/{num_envs}...")

    # Save
    bake_stage.GetRootLayer().Save()

    print("=" * 60)
    print(f"✅ DONE!")
    print(f"   Output:     {output_path}")
    print(f"   Envs:       {num_envs}")
    print(f"   Bodies:     {total_prims} prims baked")
    print(f"   Frames:     {args.num_frames}")
    print(f"   Duration:   {args.num_frames / fps:.1f}s @ {fps:.0f} fps")
    print(f"")
    print(f"   Open in USD Composer → scrub timeline → render!")
    print("=" * 60)

    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()