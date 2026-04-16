#!/usr/bin/env python3
"""
run_worldlabs.py

Full pipeline:
1. Builds World Labs Marble folder → Isaac Sim USD
2. Launches Spot (no arm) playback in the environment

Usage:
python run_worldlabs.py
"""

import subprocess
import sys
import shutil
import os
from pathlib import Path

# ═══════════════════════════════════════════════════════════════
# EDIT THESE
# ═══════════════════════════════════════════════════════════════

# Folder containing .usdz and .glb from World Labs Marble
MARBLE_INPUT = "/home/partnersteam2/isaac_lab_spot/IsaacLab/IsaacLab/WorldLabs/TEST_V1"

# IsaacLab root
ISAACLAB_ROOT = "/home/partnersteam2/isaac_lab_spot/IsaacLab/IsaacLab"

# WorldLabs dir (where builder script and output USD live)
WORLDLABS_DIR = "/home/partnersteam2/isaac_lab_spot/IsaacLab/IsaacLab/WorldLabs"

# Builder script
BUILDER_SCRIPT = "/home/partnersteam2/isaac_lab_spot/IsaacLab/IsaacLab/WorldLabs/worldlabs_to_isaac.py"

# Spot checkpoint (no arm, trained on flat terrain)
CHECKPOINT = "logs/rsl_rl/spot_flat/2026-04-16_15-20-20/model_1400.pt"

# Number of Spots to spawn
NUM_ENVS = 5

# ═══════════════════════════════════════════════════════════════

def main():
    print("")
    print("═" * 60)
    print("  World Labs → Spot Pipeline")
    print("═" * 60)
    print(f"  Marble input : {MARBLE_INPUT}")
    print(f"  Builder      : {BUILDER_SCRIPT}")
    print(f"  Output       : {WORLDLABS_DIR}/test_env.usd")
    print(f"  Checkpoint   : {CHECKPOINT}")
    print(f"  Num envs     : {NUM_ENVS}")
    print("")

    # ══════════════════════════════════════════════════════════
    # STEP 1: Build the USD environment
    # ══════════════════════════════════════════════════════════
    print("═" * 60)
    print("  Step 1: Building World Labs USD environment")
    print("═" * 60)
    print("")

    result = subprocess.run([
        sys.executable, BUILDER_SCRIPT,
        "--input", MARBLE_INPUT,
        "--output", "test_env.usd",
    ], check=True)

    # Copy test_env.usd to WorldLabs dir (where worldlabs_env_cfg.py reads it)
    src = Path(MARBLE_INPUT) / "test_env.usd"
    dst = Path(WORLDLABS_DIR) / "test_env.usd"

    if src != dst:
        shutil.copy2(src, dst)
        print(f"\n  Copied: {src}")
        print(f"      → : {dst}")

    # Copy extracted_usdz folder (referenced by the USD)
    src_usdz = Path(MARBLE_INPUT) / "extracted_usdz"
    dst_usdz = Path(WORLDLABS_DIR) / "extracted_usdz"
    if src_usdz.exists() and src_usdz != dst_usdz:
        if dst_usdz.exists():
            shutil.rmtree(dst_usdz)
        shutil.copytree(src_usdz, dst_usdz)
        print(f"  Copied: extracted_usdz/")

    # Copy GLB files (collider mesh)
    for glb in Path(MARBLE_INPUT).glob("*.glb"):
        dst_glb = Path(WORLDLABS_DIR) / glb.name
        if glb != dst_glb:
            shutil.copy2(glb, dst_glb)
            print(f"  Copied: {glb.name}")

    print("")
    print("  ✅ Environment built successfully")
    print("")

    # ══════════════════════════════════════════════════════════
    # STEP 2: Launch Spot playback
    # ══════════════════════════════════════════════════════════
    print("═" * 60)
    print("  Step 2: Launching Spot (no arm) in the environment")
    print("═" * 60)
    print("")

    os.chdir(ISAACLAB_ROOT)

    subprocess.run([
        sys.executable,
        "scripts/reinforcement_learning/rsl_rl/play.py",
        "--task", "Isaac-Velocity-WorldLabs-Spot-Play-v0",
        "--num_envs", str(NUM_ENVS),
        "--checkpoint", CHECKPOINT,
    ], check=True)

if __name__ == "__main__":
    main()