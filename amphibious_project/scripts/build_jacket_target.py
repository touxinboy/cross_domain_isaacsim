"""Build and visually verify an own jacket target using the full Isaac Sim app."""

import asyncio
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback

import carb
import omni.kit.app
import omni.usd
from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
from river_geometry import RiverGeometry
from robot_assets import box, cylinder, material, project_path, set_camera


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tube(stage, path, a, b, radius, mat, collision=True):
    delta = Gf.Vec3d(*b) - Gf.Vec3d(*a)
    midpoint = (Gf.Vec3d(*a) + Gf.Vec3d(*b)) / 2
    obj = cylinder(stage, path, radius, delta.GetLength(), midpoint, mat, collision=collision)
    q = Gf.Rotation(Gf.Vec3d(0, 0, 1), delta.GetNormalized()).GetQuat()
    obj.AddOrientOp().Set(Gf.Quatf(q.GetReal(), Gf.Vec3f(q.GetImaginary())))
    return obj


def semantic_label(prim, label):
    # This is the semantics schema shipped with Isaac Sim 4.5.
    from pxr import Semantics
    api = Semantics.SemanticsAPI.Apply(prim, "Semantics")
    api.CreateSemanticTypeAttr().Set("class")
    api.CreateSemanticDataAttr().Set(label)


def build_asset(cfg):
    output = project_path(PROJECT, cfg["asset_file"])
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise RuntimeError("Target asset already exists; refusing to replace it implicitly.")
    stage = Usd.Stage.CreateNew(str(output))
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdGeom.SetStageUpAxis(stage, "Z")
    root = UsdGeom.Xform.Define(stage, "/Jacket")
    stage.SetDefaultPrim(root.GetPrim())
    root.GetPrim().SetDocumentation(cfg["model_scope"])
    semantic_label(root.GetPrim(), cfg["semantic_class"])
    yellow = material(stage, "/Jacket/Materials/YellowSteel", (.93, .56, .035), roughness=.32)
    dark = material(stage, "/Jacket/Materials/Braces", (.18, .24, .29), roughness=.4)
    blue = material(stage, "/Jacket/Materials/Deck", (.025, .25, .52), roughness=.45)
    h, base, top = cfg["frame_height_m"], cfg["base_span_m"], cfg["top_span_m"]
    signs = [(-1, -1), (1, -1), (1, 1), (-1, 1)]

    def node(i, z):
        span = base + (top - base) * z / h
        x, y = signs[i % 4]
        return (x * span / 2, y * span / 2, z)

    for i in range(4):
        tube(stage, f"/Jacket/Legs/Leg_{i}", node(i, .08), node(i, h), cfg["leg_radius_m"], yellow)
        box(stage, f"/Jacket/Feet/Plate_{i}", (.48, .48, .08),
            (signs[i][0] * base / 2, signs[i][1] * base / 2, .04), dark, True)
    levels = [.24, h * .3533333333, h * .6733333333, h]
    for k, z in enumerate(levels):
        for face in range(4):
            tube(stage, f"/Jacket/Rings/Level_{k}_Side_{face}", node(face, z), node(face + 1, z), .085, yellow)
            p = node(face, z)
            cylinder(stage, f"/Jacket/Joints/Collar_{k}_{face}", .177, .18, p, yellow)
    for k, (lower, upper) in enumerate(zip(levels, levels[1:])):
        for face in range(4):
            tube(stage, f"/Jacket/Braces/Bay_{k}_Face_{face}_A", node(face, lower),
                 node(face + 1, upper), cfg["brace_radius_m"], dark)
            tube(stage, f"/Jacket/Braces/Bay_{k}_Face_{face}_B", node(face + 1, lower),
                 node(face, upper), cfg["brace_radius_m"], dark)
    deck_span = top + .42
    box(stage, "/Jacket/Deck/Platform", (deck_span, deck_span, .14), (0, 0, h + .075), blue, True)
    rail_half = deck_span / 2 - .08
    corners = [(sx * rail_half, sy * rail_half) for sx, sy in signs]
    for i, (x, y) in enumerate(corners):
        tube(stage, f"/Jacket/Rails/Post_{i}", (x, y, h + .15), (x, y, h + 1.2), .035, yellow)
        nx, ny = corners[(i + 1) % 4]
        for j, dz in enumerate([.66, 1.2]):
            tube(stage, f"/Jacket/Rails/Side_{i}_{j}", (x, y, h + dz), (nx, ny, h + dz), .028, yellow)
    ladder_bottom, ladder_top = (0, base / 2 + .22, .35), (0, top / 2 + .22, h + .15)
    for side in [-1, 1]:
        a = (side * .24, ladder_bottom[1], ladder_bottom[2])
        b = (side * .24, ladder_top[1], ladder_top[2])
        tube(stage, f"/Jacket/Ladder/Rail_{'L' if side < 0 else 'R'}", a, b, .026, yellow, False)
    for i in range(20):
        t = i / 19
        y = ladder_bottom[1] * (1 - t) + ladder_top[1] * t
        z = ladder_bottom[2] * (1 - t) + ladder_top[2] * t
        tube(stage, f"/Jacket/Ladder/Rung_{i:02d}", (-.24, y, z), (.24, y, z), .019, dark, False)
    stage.GetRootLayer().customLayerData = {"owner": "L17", "target_id": cfg["id"], "scope": cfg["model_scope"]}
    stage.GetRootLayer().Save()
    return output


def build_scene(cfg, field, asset, source_hashes):
    output = project_path(PROJECT, cfg["scene_file"])
    if output.exists():
        raise RuntimeError("Target scene already exists; refusing to replace it implicitly.")
    stage = Usd.Stage.CreateNew(str(output))
    stage.GetRootLayer().subLayerPaths = ["river_robot_v1.usda"]
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, "Z")
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
    cx, cy = cfg["center_xy_m"]
    half = cfg["base_span_m"] / 2
    feet = [(cx + sx * half, cy + sy * half) for sx, sy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]]
    # Sample the whole pedestal footprint so every footing intersects the terrain.
    foot_grounds = [[field.mesh_height(x + a, y + b) for a in [-.34, 0, .34] for b in [-.34, 0, .34]] for x, y in feet]
    base_z = max(max(g) for g in foot_grounds) + .04
    UsdGeom.Xform.Define(stage, "/World/NavigationTargets")
    root = UsdGeom.Xform.Define(stage, cfg["prim_path"])
    root.GetPrim().GetReferences().AddReference("../" + cfg["asset_file"])
    root.AddTranslateOp().Set(Gf.Vec3d(cx, cy, base_z))
    root.GetPrim().SetCustomDataByKey("navigation:targetId", cfg["id"])
    concrete = material(stage, "/World/NavigationTargets/Materials/Concrete", (.35, .37, .36), roughness=.85)
    for i, ((x, y), grounds) in enumerate(zip(feet, foot_grounds)):
        bottom = min(grounds) - .12
        box(stage, f"/World/NavigationTargets/JacketFoundations/Pedestal_{i}",
            (.68, .68, base_z - bottom), (x, y, (base_z + bottom) / 2), concrete, True)
    trees, rocks = field.objects()
    cleared = []
    for kind, objects, radius_key, prefix in [("tree", trees, "crown_radius", "/World/Vegetation/Tree_"),
                                               ("rock", rocks, "radii", "/World/Obstacles/Rock_")]:
        for i, obj in enumerate(objects):
            dx = max(0, abs(obj["x"] - cx) - half - .34)
            dy = max(0, abs(obj["y"] - cy) - half - .34)
            radius = obj[radius_key] if kind == "tree" else max(obj["radii"][:2])
            if math.hypot(dx, dy) < radius + .1:
                path = f"{prefix}{i:03d}"
                if not stage.GetPrimAtPath(path):
                    raise RuntimeError("Expected own obstacle missing: " + path)
                stage.OverridePrim(path).SetActive(False)
                cleared.append(path)
    gx, gy = cfg["goal_xy_m"]
    approach = UsdGeom.Xform.Define(stage, "/World/NavigationTargets/ApproachGoal")
    approach.AddTranslateOp().Set(Gf.Vec3d(gx, gy, field.mesh_height(gx, gy)))
    approach.AddRotateZOp().Set(cfg["goal_yaw_deg"])
    approach.GetPrim().SetCustomData({"radius_m": cfg["goal_radius_m"], "evaluator_only": True})
    for name in ["Start", "Goal"]:
        prim = stage.GetPrimAtPath("/World/TaskMarkers/" + name)
        if prim:
            UsdGeom.Imageable(prim).MakeInvisible()
    x, y = cfg["start_xy_m"]
    robot = stage.GetPrimAtPath("/World/Robot")
    robot.GetAttribute("xformOp:translate").Set(Gf.Vec3d(x, y, field.mesh_height(x, y) + .25))
    robot.GetAttribute("xformOp:orient").Set(Gf.Quatf(1))
    set_camera(stage, "/World/Cameras/Target", (3, -34, 8.7), (cx, cy, base_z + 3.3), 32)
    set_camera(stage, "/World/Cameras/Crossing", (-18, -42, 19), (0, -22, 1.8), 29)
    stage.GetRootLayer().customLayerData = {"owner": "L17", "target_config": "configs/jacket_target_v1.json"}
    stage.GetRootLayer().Save()
    colliders = [str(p.GetPath()) for p in stage.Traverse()
                 if p.HasAPI(UsdPhysics.CollisionAPI) and str(p.GetPath()).startswith("/World/NavigationTargets/")]
    bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).ComputeWorldBound(root.GetPrim()).ComputeAlignedRange()
    min_clearance = float("inf")
    for obj in trees + rocks:
        idx = trees.index(obj) if "height" in obj else rocks.index(obj)
        path = ("/World/Vegetation/Tree_" if "height" in obj else "/World/Obstacles/Rock_") + f"{idx:03d}"
        if path in cleared:
            continue
        radius = obj["radius"] if "height" in obj else max(obj["radii"][:2])
        min_clearance = min(min_clearance, math.hypot(obj["x"] - gx, obj["y"] - gy) - radius - .7 - cfg["goal_radius_m"])
    jacket_clearance = cx - half - .34 - gx - .7 - cfg["goal_radius_m"]
    report = {"scene": str(output), "asset": str(asset), "target_prim": cfg["prim_path"],
              "world_origin_m": [cx, cy, base_z], "approach_xy_m": [gx, gy],
              "approach_radius_m": cfg["goal_radius_m"], "target_static_colliders": len(colliders),
              "bounds_min_m": list(bounds.GetMin()), "bounds_max_m": list(bounds.GetMax()),
              "cleared_own_obstacles_in_overlay_only": cleared,
              "approach_disk_clearance_with_robot_radius_0_7m": {"jacket_m": jacket_clearance, "other_obstacles_m": min_clearance},
              "source_hashes_before": source_hashes}
    if len(colliders) < 50 or min_clearance <= 0 or jacket_clearance <= 0:
        raise RuntimeError("Jacket collision/goal-clearance check failed: " + json.dumps(report))
    return output, report


async def wait_png(path, timeout=20):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.is_file() and path.stat().st_size > 1000:
            with path.open("rb") as handle:
                signature = handle.read(8)
                handle.seek(-12, 2)
                tail = handle.read(12)
            if signature == b"\x89PNG\r\n\x1a\n" and tail == b"\x00\x00\x00\x00IEND\xaeB`\x82":
                return
        await omni.kit.app.get_app().next_update_async()
    raise RuntimeError("Screenshot not complete: " + str(path))


async def main():
    app = omni.kit.app.get_app()
    output = project_path(PROJECT, "results/jacket_target_v1")
    output.mkdir(parents=True, exist_ok=True)
    try:
        cfg = json.loads((PROJECT / "configs/jacket_target_v1.json").read_text())
        field = RiverGeometry(json.loads((PROJECT / "configs/river_forest.json").read_text()))
        protected = ["scenes/river_robot_v1.usda", "scenes/river_forest_v1.usda", "robots/tracked_twin_thruster_v1.usda"]
        hashes = {s: digest(PROJECT / s) for s in protected}
        asset = build_asset(cfg)
        scene, report = build_scene(cfg, field, asset, hashes)
        success, error = await omni.usd.get_context().open_stage_async(str(scene))
        if not success:
            raise RuntimeError(str(error))
        settings = carb.settings.get_settings()
        settings.set_int("/persistent/app/viewport/displayOptions", 1 << 10)
        settings.set_bool("/app/viewport/grid/enabled", False)
        viewport = get_active_viewport()
        viewport.set_texture_resolution((1280, 720))
        report["screenshots"] = {}
        for camera, name in [("Crossing", "river_with_jacket.png"), ("Target", "jacket_target.png")]:
            viewport.set_active_camera("/World/Cameras/" + camera)
            for _ in range(65):
                await app.next_update_async()
            path = output / name
            await capture_viewport_to_file(viewport, str(path)).wait_for_result()
            await wait_png(path)
            report["screenshots"][name] = {"bytes": path.stat().st_size, "sha256": digest(path)}
        # Load the same panel the normal project launcher uses and verify its target mission.
        from robot_task_ui import show_robot_tools
        panel = show_robot_tools(PROJECT)
        panel.camera("Target")
        mission = panel.scene_cfg["missions"][panel.mission_index]
        report["default_ui_mission"] = mission
        if mission["id"] != cfg["id"]:
            raise RuntimeError("Target mission missing from the robot panel")
        report["source_hashes_after"] = {s: digest(PROJECT / s) for s in protected}
        if report["source_hashes_after"] != hashes:
            raise RuntimeError("An existing scene or robot asset changed unexpectedly")
        task_file = PROJECT / "configs/task.json"
        old_task = json.loads(task_file.read_text())
        (output / "task_before.json").write_text(json.dumps(old_task, indent=2) + "\n")
        old_task.update(task_name="river_robot_jacket_v1", scene_file=cfg["scene_file"],
                        target_config="configs/jacket_target_v1.json",
                        description="Own river, amphibious robot and static jacket inspection target. Navigation policy is not connected yet.")
        task_file.write_text(json.dumps(old_task, indent=2) + "\n")
        report["passed"] = True
        (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
        print("L17_JACKET_TARGET_VERIFIED " + str(scene), flush=True)
        # No motion in this preview; the regular launcher opens the persisted target scene.
        panel.loop.cancel()
        panel.runtime().shutdown()
        await omni.usd.get_context().close_stage_async()
        app.post_uncancellable_quit(0)
    except Exception:
        (output / "error.txt").write_text(traceback.format_exc())
        print("L17_JACKET_TARGET_ERROR\n" + traceback.format_exc(), flush=True)
        await omni.usd.get_context().close_stage_async()
        app.post_uncancellable_quit(3)


_JACKET_BUILD = asyncio.ensure_future(main())
