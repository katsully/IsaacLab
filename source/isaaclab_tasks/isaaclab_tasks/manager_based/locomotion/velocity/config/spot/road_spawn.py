# road_spawn.py

import torch
import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

def add_road_to_stage(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    usd_path: str,
    prim_path: str = "/World/RoadMesh",
):
    """Add road USD as a direct reference into the stage. Runs once at startup."""
    stage = omni.usd.get_context().get_stage()

    # Only add once
    if stage.GetPrimAtPath(prim_path).IsValid():
        return

    # Create prim and add USD reference — same as dragging into Composer
    prim = stage.DefinePrim(prim_path, "Xform")
    prim.GetReferences().AddReference(usd_path)

    # Add collision to ALL meshes inside (handles nested hierarchy)
    count = 0
    for descendant in Usd.PrimRange(prim):
        if descendant.IsA(UsdGeom.Mesh):
            if not descendant.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI.Apply(descendant)
            if not descendant.HasAPI(UsdPhysics.MeshCollisionAPI):
                mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(descendant)
                # Ensure we use high-fidelity collision for scanned roads
                mesh_collision.GetApproximationAttr().Set("triangleMesh")
            count += 1

    print(f"[RoadSpawn] Added {count} collision mesh(es) from {usd_path}")