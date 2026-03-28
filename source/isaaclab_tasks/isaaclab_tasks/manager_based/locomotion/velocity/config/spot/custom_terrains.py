# custom_terrains.py
from __future__ import annotations

import numpy as np
import trimesh
from isaaclab.terrains import SubTerrainBaseCfg
from isaaclab.utils import configclass

def rough_terrain_with_plinths(difficulty: float, cfg) -> tuple:
    """Returns ([mesh, plinth1, plinth2...], origin) for Isaac Lab."""
    width, length = cfg.size
    cx = width / 2.0
    cy = length / 2.0
    all_meshes = []

    # Rough base
    noise_amplitude = (
        cfg.noise_range[0] + difficulty * (cfg.noise_range[1] - cfg.noise_range[0])
    )
    step = 0.2
    rows = int(width / step) + 1
    cols = int(length / step) + 1

    xs = np.linspace(0.0, width, rows)
    ys = np.linspace(0.0, length, cols)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    zz = np.random.uniform(-noise_amplitude, noise_amplitude, (rows, cols))

    # Smooth border - fade to 0 at tile edges
    border_width = 1.0
    border_px = max(1, int(border_width / step))
    for i in range(border_px):
        fade = i / border_px
        zz[i, :] *= fade
        zz[-(i + 1), :] *= fade
        zz[:, i] *= fade
        zz[:, -(i + 1)] *= fade

    # Build floor mesh
    verts = np.column_stack([xx.flatten(), yy.flatten(), zz.flatten()])
    i_idx, j_idx = np.meshgrid(
        np.arange(rows - 1), np.arange(cols - 1), indexing="ij"
    )
    i_idx = i_idx.flatten()
    j_idx = j_idx.flatten()
    v00 = i_idx * cols + j_idx
    v10 = (i_idx + 1) * cols + j_idx
    v01 = i_idx * cols + j_idx + 1
    v11 = (i_idx + 1) * cols + j_idx + 1
    faces = np.vstack([
        np.column_stack([v00, v10, v01]),
        np.column_stack([v10, v11, v01]),
    ])
    floor_mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    floor_mesh.fix_normals()
    all_meshes.append(floor_mesh)

    # Plinths - avoid centre and edges
    half_p = cfg.platform_width / 2.0
    for _ in range(cfg.num_obstacles):
        obs_w = float(np.random.uniform(*cfg.obstacle_width_range))
        obs_h = float(np.random.uniform(*cfg.obstacle_height_range))
        half_w = obs_w / 2.0

        for _ in range(200):
            ox = float(np.random.uniform(
                border_width + half_w + 0.3,
                width - border_width - half_w - 0.3
            ))
            oy = float(np.random.uniform(
                border_width + half_w + 0.3,
                length - border_width - half_w - 0.3
            ))
            too_close = (
                abs(ox - cx) < half_p + half_w + 0.3
                and abs(oy - cy) < half_p + half_w + 0.3
            )
            if not too_close:
                break

        # Get MINIMUM terrain height under entire plinth footprint
        r_start = max(0, int((ox - half_w) / step))
        r_end = min(rows - 1, int((ox + half_w) / step)) + 1
        c_start = max(0, int((oy - half_w) / step))
        c_end = min(cols - 1, int((oy + half_w) / step)) + 1
        base_z = float(np.min(zz[r_start:r_end, c_start:c_end]))

        plinth = trimesh.creation.box(extents=[obs_w, obs_w, obs_h])
        plinth.apply_translation([ox, oy, base_z + obs_h / 2.0])
        all_meshes.append(plinth)

    origin = np.array([cx, cy, 0.0])
    return all_meshes, origin

@configclass
class RoughWithPlinthsTerrainCfg(SubTerrainBaseCfg):
    """Rough terrain with plinth obstacles - proper collision."""

    function: object = rough_terrain_with_plinths

    noise_range: tuple[float, float] = (0.02, 0.08)
    noise_step: float = 0.02
    obstacle_width_range: tuple[float, float] = (0.3, 0.4)
    obstacle_height_range: tuple[float, float] = (0.5, 0.8)
    num_obstacles: int = 1
    platform_width: float = 2.0