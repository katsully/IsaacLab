#!/usr/bin/env python3
"""
worldlabs_to_isaac.py

Converts World Labs Marble output into an Isaac Sim-ready USD environment.
Automates: USDZ extraction, Gauss transform, physics collider, dome light.

pip install usd-core is needed

python worldlabs_to_isaac.py --input ./TEST_V1 --output test_env.usd to run


Usage:
Standalone:  python worldlabs_to_isaac.py
With args:   python worldlabs_to_isaac.py --input /path/to/dir --output my_env.usd
Isaac Sim:   Paste into Script Editor, call main() at bottom
"""

import os
import sys
import zipfile
import shutil
import argparse
from pathlib import Path

from pxr import (
    Usd, UsdGeom, UsdPhysics, UsdLux, UsdShade,
    Gf, Sdf, Vt, Kind, UsdUtils
)

# ─── Defaults ────────────────────────────────────────────────────────────────
DEFAULT_BASE_DIR  = "/home/partnersteam2/WorldLab/WorldLabs/TEST_V1"
DEFAULT_OUTPUT    = "test_env.usd"

GAUSS_SCALE       = Gf.Vec3f(2.0, 2.0, 2.0)
GAUSS_ROTATE_X    = 90.0        # Set to 90.0 if splat needs rotating — off by default
GAUSS_Y_OFFSET    = 0.0        # Tweak to raise/lower splat for floor alignment
DOME_INTENSITY    = 1000.0

class WorldLabsEnvBuilder:
    """
    End-to-end builder that turns a World Labs Marble folder into a
    physics-enabled USD scene for Isaac Sim.
    """

    def __init__(self, base_dir: str, output_name: str = DEFAULT_OUTPUT):
        self.base_dir    = Path(base_dir).resolve()
        self.output_name = output_name
        self.extract_dir = self.base_dir / "extracted_usdz"
        self.stage       = None
        self.usdz_path   = None
        self.usda_path   = None
        self.glb_path    = None

    # ── File Discovery ────────────────────────────────────────────────────

    def _find_single(self, pattern: str, required: bool = True):
        hits = sorted(self.base_dir.glob(pattern))
        if not hits:
            if required:
                raise FileNotFoundError(
                    f"No {pattern} found in {self.base_dir}")
            return None
        if len(hits) > 1:
            print(f"  ⚠  Multiple {pattern} found — using first: {hits[0].name}")
        return hits[0]

    def discover_files(self):
        print("\n🔍  Discovering files …")
        self.usdz_path = self._find_single("*.usdz")
        self.glb_path  = self._find_single("*.glb", required=False)
        print(f"   USDZ     : {self.usdz_path.name}")
        print(f"   Collider : {self.glb_path.name if self.glb_path else '—  (none found)'}")

    # ── USDZ Extraction ──────────────────────────────────────────────────

    def extract_usdz(self):
        print("\n📦  Extracting USDZ …")
        if self.extract_dir.exists():
            shutil.rmtree(self.extract_dir)
        self.extract_dir.mkdir(parents=True)

        with zipfile.ZipFile(self.usdz_path, "r") as zf:
            zf.extractall(self.extract_dir)

        candidates = list(self.extract_dir.rglob("default.usda"))
        if not candidates:
            candidates = list(self.extract_dir.rglob("*.usda"))
        if not candidates:
            raise FileNotFoundError(
                "No .usda found inside extracted USDZ — is this a valid Marble export?")

        self.usda_path = candidates[0]
        print(f"   USDA located: {self.usda_path.relative_to(self.base_dir)}")

    # ── Stage Bootstrap ───────────────────────────────────────────────────

    def create_stage(self):
        out = str(self.base_dir / self.output_name)
        print(f"\n🏗️   Creating new stage → {out}")

        if os.path.exists(out):
            os.remove(out)

        self.stage = Usd.Stage.CreateNew(out)
        UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.y)
        UsdGeom.SetStageMetersPerUnit(self.stage, 1.0)

        world = UsdGeom.Xform.Define(self.stage, "/World")
        self.stage.SetDefaultPrim(world.GetPrim())
        Usd.ModelAPI(world.GetPrim()).SetKind(Kind.Tokens.assembly)

    # ── Gauss Splat ───────────────────────────────────────────────────────

    def add_gauss(self):
        """
        Reference the extracted USDA under /World/Gauss.

        Applied ops (local opinions override anything from the referenced layer):
          • Scale    2×               (always)
          • Rotate X GAUSS_ROTATE_X° (0 by default — set to 90 to re-enable)
          • Translate Y GAUSS_Y_OFFSET (floor alignment tweak)

        xformOpOrder is stamped explicitly so the referenced layer's own
        op order cannot bleed through.
        """
        print("\n🌌  Adding Gaussian-splat scene …")

        gauss_xform = UsdGeom.Xform.Define(self.stage, "/World/Gauss")
        prim = gauss_xform.GetPrim()

        prim.GetReferences().AddReference(
            str(self.usda_path),
            Sdf.Path.emptyPath
        )

        # ── Build ops ────────────────────────────────────────────────────
        active_ops = []

        # Scale
        s = gauss_xform.AddScaleOp(opSuffix="sizeMatch")
        s.Set(GAUSS_SCALE)
        active_ops.append("xformOp:scale:sizeMatch")

        # Rotate X — only wired in when non-zero
        if GAUSS_ROTATE_X != 0.0:
            r = gauss_xform.AddRotateXOp(opSuffix="orientFix")
            r.Set(GAUSS_ROTATE_X)
            active_ops.append("xformOp:rotateX:orientFix")

        # Translate
        t = gauss_xform.AddTranslateOp(opSuffix="floorAlign")
        t.Set(Gf.Vec3d(0.0, GAUSS_Y_OFFSET, 0.0))
        active_ops.append("xformOp:translate:floorAlign")

        # Stamp xformOpOrder so referenced layer cannot override
        prim.GetAttribute("xformOpOrder").Set(Vt.TokenArray(active_ops))

        print(f"   Scale       = {GAUSS_SCALE}")
        print(f"   Rotate    X = {GAUSS_ROTATE_X}°  "
              f"{'(active)' if GAUSS_ROTATE_X != 0.0 else '(disabled — set GAUSS_ROTATE_X=90 to enable)'}")
        print(f"   Translate Y = {GAUSS_Y_OFFSET}")

        return prim

    # ── Physics Scene ─────────────────────────────────────────────────────

    def add_physics_scene(self):
        print("\n⚛️   Adding PhysicsScene …")
        ps = UsdPhysics.Scene.Define(self.stage, "/World/PhysicsScene")
        ps.CreateGravityDirectionAttr(Gf.Vec3f(0.0, -1.0, 0.0))
        ps.CreateGravityMagnitudeAttr(9.81)

    # ── Dome Light ────────────────────────────────────────────────────────

    def add_dome_light(self):
        print("\n💡  Adding dome light …")
        dl = UsdLux.DomeLight.Define(self.stage, "/World/DomeLight")
        dl.CreateIntensityAttr(DOME_INTENSITY)
        dl.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
        dl.CreateTextureFormatAttr("latlong")
        print(f"   Intensity = {DOME_INTENSITY}")

    # ── Collider GLB ──────────────────────────────────────────────────────

    def add_collider(self, gauss_prim):
        """
        Reference the collider GLB as a child of /World/Gauss/Collider.
        • CollisionAPI applied to the Xform root and recursively to child Meshes.
        • Mesh visibility is hidden via two methods:
            1. PrimRange traversal  — works when GLB has been pre-converted to USD.
            2. OverridePrim at the known path geometry_0/Mesh0  — works even when
               USD can't parse the raw GLB at script time; the opinion is already
               in the layer when Isaac Sim composes and converts the GLB.
        """
        if self.glb_path is None:
            print("\n⚠   Skipping collider — no GLB found.")
            return None

        print("\n🧱  Adding collider mesh …")
        collider_sdf = gauss_prim.GetPath().AppendChild("Collider")
        cxf   = UsdGeom.Xform.Define(self.stage, collider_sdf)
        cprim = cxf.GetPrim()

        usd_version = self._try_convert_glb()
        ref_path    = str(usd_version) if usd_version else str(self.glb_path)

        cprim.GetReferences().AddReference(ref_path)
        print(f"   Referenced: {Path(ref_path).name}")

        # Physics on the Xform root
        UsdPhysics.CollisionAPI.Apply(cprim)
        mc = UsdPhysics.MeshCollisionAPI.Apply(cprim)
        mc.CreateApproximationAttr("none")

        # Also tag every nested Mesh found via traversal
        self._apply_collision_recursive(cprim)

        # Hide mesh prims
        self._hide_meshes_only(cprim, collider_sdf)

        print("   Collision applied  |  Mesh hidden  |  Xform kept intact")
        return cxf

    def _apply_collision_recursive(self, prim):
        """Walk descendants and apply CollisionAPI to every Mesh prim."""
        for child in Usd.PrimRange(prim):
            if child.IsA(UsdGeom.Mesh):
                if not child.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI.Apply(child)
                    UsdPhysics.MeshCollisionAPI.Apply(child).CreateApproximationAttr("none")

    def _hide_meshes_only(self, root_prim, root_sdf_path: Sdf.Path):
        """
        Two-pass mesh hiding:

        Pass 1 — PrimRange traversal
          Works when the referenced asset is a proper USD file that the runtime
          can compose (e.g. a pre-converted .usd/.usdc).  Any Mesh prim found
          gets MakeInvisible() called directly.

        Pass 2 — OverridePrim at known paths
          When the reference target is a raw .glb the standalone USD runtime
          cannot parse it, so PrimRange finds no children.  We instead stamp
          a  visibility = invisible  opinion via OverridePrim at the paths we
          know the GLB will produce once Isaac Sim converts it:

              <Collider>/geometry_0/Mesh0      ← primary target
              <Collider>/Mesh0                 ← fallback (flat GLBs)
              <Collider>/world/geometry_0/Mesh0 ← fallback (nested GLBs)

          USD opinion strength guarantees our layer wins over the referenced
          default, so the mesh is invisible the moment Isaac Sim loads the file.
        """
        hidden_traversal = 0

        # ── Pass 1: traversal ─────────────────────────────────────────────
        for prim in Usd.PrimRange(root_prim):
            if prim.IsA(UsdGeom.Mesh):
                UsdGeom.Imageable(prim).MakeInvisible()
                hidden_traversal += 1
                print(f"   [traversal] hidden: {prim.GetPath()}")

        if hidden_traversal:
            print(f"   Pass 1 — {hidden_traversal} mesh(es) hidden via traversal")
        else:
            print("   Pass 1 — no mesh prims found via traversal "
                  "(GLB not yet converted — falling back to override pass)")

        # ── Pass 2: explicit override at known GLB path patterns ──────────
        #
        # These paths match the hierarchy World Labs / standard GLB exporters
        # produce.  OverridePrim creates a spec in OUR layer even if the prim
        # doesn't exist yet; when Isaac Sim composes + converts the GLB the
        # invisible opinion is already authored and wins.
        known_relative_paths = [
            "geometry_0/Mesh0",               # standard Marble collider GLB
            "Mesh0",                          # flat single-mesh GLB
            "world/geometry_0/Mesh0",         # nested-root GLB variant
            "world/Mesh0",                    # nested flat variant
        ]

        for rel in known_relative_paths:
            target = root_sdf_path.AppendPath(rel)
            try:
                over = self.stage.OverridePrim(target)
                over.CreateAttribute(
                    "visibility",
                    Sdf.ValueTypeNames.Token,
                    custom=False
                ).Set(UsdGeom.Tokens.invisible)
                print(f"   [override]  visibility=invisible → {target}")
            except Exception as e:
                print(f"   [override]  could not write {target}: {e}")

    def _try_convert_glb(self):
        """Best-effort GLB → USD conversion for standalone use."""
        usd_out = self.glb_path.with_suffix(".usd")
        if usd_out.exists():
            print(f"   Pre-converted USD exists: {usd_out.name}")
            return usd_out

        # Attempt 1: Omniverse asset converter (inside Isaac Sim Kit)
        try:
            import omni.kit.asset_converter  # noqa
            print("   Omniverse asset converter detected — runtime will convert GLB.")
            return None
        except ImportError:
            pass

        # Attempt 2: trimesh
        try:
            import trimesh
            print("   Converting GLB via trimesh …")
            scene = trimesh.load(str(self.glb_path), force="scene")
            try:
                scene.export(str(usd_out), file_type="usdc")
                print(f"   ✓ Converted to {usd_out.name}")
                return usd_out
            except Exception:
                pass
            obj_path = self.glb_path.with_suffix(".obj")
            scene.export(str(obj_path))
            print(f"   Exported intermediate OBJ: {obj_path.name}")
            print("   ⚠  Full GLB→USD needs Isaac Sim runtime or `usdcat`.")
            return None
        except ImportError:
            pass

        print("   ⚠  No converter available — GLB referenced directly (works in Isaac Sim).")
        return None

    # ── Save ──────────────────────────────────────────────────────────────

    def save(self):
        out = str(self.base_dir / self.output_name)
        self.stage.GetRootLayer().Save()
        print(f"\n💾  Stage saved → {out}")
        print(f"   Layer preview:\n{self.stage.GetRootLayer().ExportToString()[:600]}…\n")

    # ── Main Pipeline ─────────────────────────────────────────────────────

    def build(self):
        divider = "═" * 64
        print(f"\n{divider}")
        print("  World Labs Marble  →  Isaac Sim USD  Environment Builder")
        print(f"{divider}")
        print(f"  Input dir : {self.base_dir}")
        print(f"  Output    : {self.output_name}")

        self.discover_files()
        self.extract_usdz()
        self.create_stage()
        self.add_physics_scene()

        gauss_prim = self.add_gauss()
        self.add_dome_light()
        self.add_collider(gauss_prim)

        self.save()

        print(f"{divider}")
        print("  ✅  BUILD COMPLETE")
        print(f"{divider}")
        print(f"""
Next steps
──────────
1.  Open  {self.output_name}  in Isaac Sim
2.  If the splat needs rotating, set GAUSS_ROTATE_X = 90.0 and re-run.
3.  Tweak GAUSS_Y_OFFSET (currently {GAUSS_Y_OFFSET}) to align splat floor
  with your collision mesh.
4.  Drop a test cube with RigidBody to verify physics/collision.
5.  Swap in your robot URDF/USD and start training 🚀
""")

        return str(self.base_dir / self.output_name)

# ─── CLI Entry Point ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert World Labs Marble output → Isaac Sim USD env")
    parser.add_argument("-i", "--input",  default=DEFAULT_BASE_DIR,
                        help="Directory containing .usdz and .glb from Marble")
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT,
                        help="Output USD filename (written into --input dir)")
    args = parser.parse_args()

    builder = WorldLabsEnvBuilder(args.input, args.output)
    builder.build()

if __name__ == "__main__":
    main()