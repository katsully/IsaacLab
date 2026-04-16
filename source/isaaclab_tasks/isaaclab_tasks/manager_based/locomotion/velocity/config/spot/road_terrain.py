# road_terrain.py

from __future__ import annotations

import numpy as np
import trimesh
from isaaclab.terrains import SubTerrainBaseCfg
from isaaclab.utils import configclass

def road_mesh_terrain(difficulty: float, cfg) -> tuple:
    """Load road mesh, reposition to start at (0,0,0) — same convention as custom_terrains.py."""

    # Load and clean the mesh
    mesh = trimesh.load_mesh(cfg.obj_path, process=True)
    mesh.fix_normals()
    mesh.visual = trimesh.visual.ColorVisuals()

    # Move mesh so min corner is at (0, 0, 0)
    bounds_min = mesh.bounds[0]
    mesh.apply_translation(-bounds_min)

    # Now mesh goes from (0,0,0) to (width, length, height)
    bounds_max = mesh.bounds[1]
    width = bounds_max[0]
    length = bounds_max[1]

    print(f"[RoadTerrain] Original min was: {bounds_min}")
    print(f"[RoadTerrain] Mesh now spans: (0,0,0) to ({width:.1f}, {length:.1f}, {bounds_max[2]:.1f})")
    print(f"[RoadTerrain] Origin: ({width/2:.1f}, {length/2:.1f}, 0.0)")

    # Origin at center — exactly like custom_terrains.py
    origin = np.array([width / 2.0, length / 2.0, 0.0])

    return [mesh], origin

@configclass
class RoadMeshTerrainCfg(SubTerrainBaseCfg):
    """Terrain config that loads a road mesh from an OBJ file."""
    function: object = road_mesh_terrain
    obj_path: str = "/home/partnersteam2/IsaacRobotics/assets/TUSC_REMESH.obj"