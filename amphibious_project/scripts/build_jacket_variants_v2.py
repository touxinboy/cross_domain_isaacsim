"""Build orange land/water jacket targets and leave the verified scene open."""

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
from robot_assets import bind, box, cylinder, material, project_path, set_camera


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tube(stage, path, a, b, radius, mat, collision=True):
    delta = Gf.Vec3d(*b) - Gf.Vec3d(*a)
    midpoint = (Gf.Vec3d(*a) + Gf.Vec3d(*b)) / 2
    obj = cylinder(stage, path, radius, delta.GetLength(), midpoint, mat, collision=collision)
    q = Gf.Rotation(Gf.Vec3d(0, 0, 1), delta.GetNormalized()).GetQuat()
    obj.AddOrientOp().Set(Gf.Quatf(q.GetReal(), Gf.Vec3f(q.GetImaginary())))
    return obj


def new_asset(relative, doc):
    output = project_path(PROJECT, relative)
    if output.exists():
        raise RuntimeError("Refusing to replace an existing target asset: " + str(output))
    stage = Usd.Stage.CreateNew(str(output))
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdGeom.SetStageUpAxis(stage, "Z")
    root = UsdGeom.Xform.Define(stage, "/Jacket")
    stage.SetDefaultPrim(root.GetPrim())
    root.GetPrim().SetDocumentation(doc)
    return output, stage, root


def land_asset(cfg, palette):
    output, stage, root = new_asset(cfg["asset_file"], "Orange flat-top land jacket; original geometry preserved through a reference.")
    root.GetPrim().GetReferences().AddReference("jacket_4leg_v1.usda")
    for name, key in [("YellowSteel", "orange"), ("Braces", "orange_brace"), ("Deck", "deck")]:
        stage.GetPrimAtPath(f"/Jacket/Materials/{name}/Surface").GetAttribute("inputs:diffuseColor").Set(Gf.Vec3f(*palette[key]))
    yellow = material(stage, "/Jacket/Materials/RailYellow", palette["yellow"], roughness=.35)
    red = material(stage, "/Jacket/Materials/UpperRed", palette["red"], roughness=.36)
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if path.startswith("/Jacket/Rails/") or path.startswith("/Jacket/Ladder/Rail_"):
            if prim.IsA(UsdGeom.Gprim):
                bind(prim, yellow)
        if path.startswith("/Jacket/Braces/Bay_2_"):
            bind(prim, red)
    root.GetPrim().SetCustomDataByKey("navigation:shape", "flat_top")
    stage.GetRootLayer().Save()
    return output


def water_asset(cfg, palette, field):
    output, stage, root = new_asset(cfg["asset_file"], "Orange fixed water jacket with yellow railing and a red pointed lattice mast.")
    from pxr import Semantics
    semantics = Semantics.SemanticsAPI.Apply(root.GetPrim(), "Semantics")
    semantics.CreateSemanticTypeAttr().Set("class")
    semantics.CreateSemanticDataAttr().Set("jacket_structure")
    root.GetPrim().SetCustomDataByKey("navigation:shape", "pointed_mast")
    cx, cy = field.cross_section(cfg["center_y_m"])["center_x"], cfg["center_y_m"]
    half = cfg["base_span_m"] / 2
    signs = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    feet = [(cx + sx * half, cy + sy * half) for sx, sy in signs]
    grounds = [[field.mesh_height(x + a, y + b) for a in [-.36, 0, .36] for b in [-.36, 0, .36]] for x, y in feet]
    if max(max(v) for v in grounds) >= field.water_z - .2:
        raise RuntimeError("Water jacket foundations must be submerged")
    base_z = max(max(v) for v in grounds) + .04
    deck_z = field.water_z + cfg["deck_height_above_water_m"]
    h = deck_z - base_z
    orange = material(stage, "/Jacket/Materials/Orange", palette["orange"], roughness=.38)
    brace = material(stage, "/Jacket/Materials/OrangeBrace", palette["orange_brace"], roughness=.4)
    yellow = material(stage, "/Jacket/Materials/Yellow", palette["yellow"], roughness=.35)
    red = material(stage, "/Jacket/Materials/Red", palette["red"], roughness=.32)
    deck_mat = material(stage, "/Jacket/Materials/Deck", palette["deck"], roughness=.6)
    concrete = material(stage, "/Jacket/Materials/Concrete", (.32, .33, .30), roughness=.9)

    def corner(i, z):
        half_width = (cfg["base_span_m"] + (cfg["top_span_m"] - cfg["base_span_m"]) * z / h) / 2
        sx, sy = signs[i % 4]
        return sx * half_width, sy * half_width, z

    for i, ((x, y), samples) in enumerate(zip(feet, grounds)):
        tube(stage, f"/Jacket/Legs/Leg_{i}", corner(i, .04), corner(i, h), .18, orange)
        bottom = min(samples) - .18 - base_z
        box(stage, f"/Jacket/Foundations/Foot_{i}", (.72, .72, .04 - bottom),
            (x - cx, y - cy, (bottom + .04) / 2), concrete, True)
    levels = [.20, max(.6, h * .52), h]
    for k, z in enumerate(levels):
        for side in range(4):
            tube(stage, f"/Jacket/Rings/L{k}_S{side}", corner(side, z), corner(side + 1, z), .105, orange)
    for k, (a, b) in enumerate(zip(levels, levels[1:])):
        for side in range(4):
            tube(stage, f"/Jacket/Braces/B{k}_S{side}_A", corner(side, a), corner(side + 1, b), .072, brace)
            tube(stage, f"/Jacket/Braces/B{k}_S{side}_B", corner(side + 1, a), corner(side, b), .072, brace)
    span = cfg["deck_span_m"]
    box(stage, "/Jacket/Platform/Deck", (span, span, .17), (0, 0, h + .085), deck_mat, True)
    rail_half = span / 2 - .09
    corners = [(sx * rail_half, sy * rail_half) for sx, sy in signs]
    for i, (x, y) in enumerate(corners):
        tube(stage, f"/Jacket/Rails/Post_{i}", (x, y, h + .17), (x, y, h + 1.22), .04, yellow)
        nx, ny = corners[(i + 1) % 4]
        for j, dz in enumerate([.68, 1.22]):
            tube(stage, f"/Jacket/Rails/Side_{i}_{j}", (x, y, h + dz), (nx, ny, h + dz), .03, yellow)
    # Open lattice derrick ending in a narrow pointed tip, rather than a solid cone.
    mast_bottom = h + .18
    mast_h = cfg["mast_height_m"]
    levels = [0, .24 * mast_h, .48 * mast_h, .71 * mast_h, .90 * mast_h, mast_h]
    widths = [.95, .72, .48, .25, .085, 0]

    def mast_node(i, k):
        sx, sy = signs[i % 4]
        return sx * widths[k], sy * widths[k], mast_bottom + levels[k]

    for k in range(len(levels) - 1):
        for side in range(4):
            tube(stage, f"/Jacket/PointedMast/Leg_{k}_{side}", mast_node(side, k), mast_node(side, k + 1), .047, red)
            if k < len(levels) - 2:
                tube(stage, f"/Jacket/PointedMast/X_{k}_{side}_A", mast_node(side, k), mast_node(side + 1, k + 1), .023, red)
                tube(stage, f"/Jacket/PointedMast/X_{k}_{side}_B", mast_node(side + 1, k), mast_node(side, k + 1), .023, red)
            tube(stage, f"/Jacket/PointedMast/Ring_{k}_{side}", mast_node(side, k), mast_node(side + 1, k), .034, red)
    stage.GetRootLayer().Save()
    gx, gy = cfg["goal_xy_m"]
    yaw = math.degrees(math.atan2(cy - gy, cx - gx))
    cfg["goal_yaw_deg"] = yaw
    return output, (cx, cy, base_z), {"ground_z_max": max(max(v) for v in grounds), "deck_world_z": deck_z, "mast_tip_world_z": deck_z + .18 + mast_h}


def scene_asset(cfg, field, land, water, water_pos):
    output = project_path(PROJECT, cfg["scene_file"])
    if output.exists():
        raise RuntimeError("Refusing to replace an existing v2 scene")
    stage = Usd.Stage.CreateNew(str(output))
    stage.GetRootLayer().subLayerPaths = ["river_robot_jacket_v1.usda"]
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, "Z")
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
    land_cfg, water_cfg = cfg["targets"]
    stage.GetPrimAtPath(land_cfg["prim_path"]).GetReferences().SetReferences([Sdf.Reference("../" + land_cfg["asset_file"])])
    obj = UsdGeom.Xform.Define(stage, water_cfg["prim_path"])
    obj.GetPrim().GetReferences().AddReference("../" + water_cfg["asset_file"])
    obj.AddTranslateOp().Set(Gf.Vec3d(*water_pos))
    obj.GetPrim().SetCustomDataByKey("navigation:targetId", water_cfg["id"])
    gx, gy = water_cfg["goal_xy_m"]
    goal = UsdGeom.Xform.Define(stage, "/World/NavigationTargets/WaterApproachGoal")
    goal.AddTranslateOp().Set(Gf.Vec3d(gx, gy, field.water_z + .0354))
    goal.AddRotateZOp().Set(water_cfg["goal_yaw_deg"])
    goal.GetPrim().SetCustomData({"radius_m": water_cfg["goal_radius_m"], "evaluator_only": True, "medium": "water"})
    # Only clear collisions within the new structure's actual footprint, in this overlay.
    trees, rocks = field.objects()
    cleared = []
    half = max(water_cfg["base_span_m"] / 2 + .36, water_cfg["deck_span_m"] / 2)
    for i, rock in enumerate(rocks):
        dx = max(0, abs(rock["x"] - water_pos[0]) - half)
        dy = max(0, abs(rock["y"] - water_pos[1]) - half)
        if math.hypot(dx, dy) < max(rock["radii"][:2]) + .12:
            path = f"/World/Obstacles/Rock_{i:03d}"
            stage.OverridePrim(path).SetActive(False)
            cleared.append(path)
    set_camera(stage, "/World/Cameras/Crossing", (-22, -43, 24), (0, -17, 2.0), 29)
    set_camera(stage, "/World/Cameras/WaterTarget", (water_pos[0] - 7.5, water_pos[1] - 9.5, 6.5),
               (water_pos[0], water_pos[1], 2.0), 30)
    stage.GetRootLayer().customLayerData = {"owner": "L17", "target_config": "configs/jacket_targets_v2.json"}
    stage.GetRootLayer().Save()
    report = {"scene": str(output), "water_origin_m": list(water_pos), "cleared_own_rocks_overlay_only": cleared, "targets": {}}
    for target in cfg["targets"]:
        prim = stage.GetPrimAtPath(target["prim_path"])
        box_range = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).ComputeWorldBound(prim).ComputeAlignedRange()
        target_bounds = [list(box_range.GetMin()), list(box_range.GetMax())]
        colliders = [p for p in Usd.PrimRange(prim) if p.HasAPI(UsdPhysics.CollisionAPI)]
        if len(colliders) < 50:
            raise RuntimeError("Missing target colliders")
        x, y = target["goal_xy_m"]
        lo, hi = target_bounds
        dx, dy = max(lo[0] - x, 0, x - hi[0]), max(lo[1] - y, 0, y - hi[1])
        clearance = math.hypot(dx, dy) - .7 - target["goal_radius_m"]
        # Include concrete pedestals around the land jacket, which live outside its asset.
        if target["medium"] == "land":
            clearance = min(clearance, 12 - 1.8 - .34 - x - .7 - target["goal_radius_m"])
        obstacle_clearance = float("inf")
        for i, rock in enumerate(rocks):
            if f"/World/Obstacles/Rock_{i:03d}" not in cleared:
                obstacle_clearance = min(obstacle_clearance, math.hypot(rock["x"] - x, rock["y"] - y) - max(rock["radii"][:2]) - .7 - target["goal_radius_m"])
        if min(clearance, obstacle_clearance) <= .2:
            raise RuntimeError("Insufficient target goal clearance")
        report["targets"][target["id"]] = {"bounds_m": target_bounds, "static_colliders": len(colliders), "goal_clearance_m": clearance, "goal_rock_clearance_m": obstacle_clearance}
    right = Gf.Vec3d(26, -22, 0).GetNormalized()
    land_pos = Gf.Vec3d(12, -23, 0)
    water_xy = Gf.Vec3d(water_pos[0], water_pos[1], 0)
    report["water_left_of_land_in_crossing_view"] = Gf.Dot(water_xy, right) < Gf.Dot(land_pos, right)
    if not report["water_left_of_land_in_crossing_view"]:
        raise RuntimeError("Water jacket must be left of the land jacket")
    return output, report


async def png_complete(path):
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if path.exists() and path.stat().st_size > 1000:
            with path.open("rb") as handle:
                sig = handle.read(8)
                handle.seek(-12, 2)
                tail = handle.read(12)
            if sig == b"\x89PNG\r\n\x1a\n" and tail == b"\x00\x00\x00\x00IEND\xaeB`\x82":
                return
        await omni.kit.app.get_app().next_update_async()
    raise RuntimeError("Screenshot did not finish writing")


async def main():
    app = omni.kit.app.get_app()
    result = project_path(PROJECT, "results/jacket_targets_v2")
    result.mkdir(parents=True, exist_ok=True)
    try:
        cfg = json.loads((PROJECT / "configs/jacket_targets_v2.json").read_text())
        field = RiverGeometry(json.loads((PROJECT / "configs/river_forest.json").read_text()))
        protected = ["scenes/river_robot_v1.usda", "scenes/river_forest_v1.usda", "scenes/river_robot_jacket_v1.usda", "robots/tracked_twin_thruster_v1.usda", "assets/targets/jacket_4leg_v1.usda"]
        before = {s: sha(PROJECT / s) for s in protected}
        land = land_asset(cfg["targets"][0], cfg["palette"])
        water, position, water_info = water_asset(cfg["targets"][1], cfg["palette"], field)
        scene, report = scene_asset(cfg, field, land, water, position)
        success, error = await omni.usd.get_context().open_stage_async(str(scene))
        if not success:
            raise RuntimeError(str(error))
        settings = carb.settings.get_settings()
        settings.set_int("/persistent/app/viewport/displayOptions", 1 << 10)
        settings.set_bool("/app/viewport/grid/enabled", False)
        viewport = get_active_viewport()
        viewport.set_texture_resolution((1280, 720))
        report["screenshots"] = {}
        for camera, name in [("Crossing", "both_jackets.png"), ("WaterTarget", "water_pointed_jacket.png"), ("Target", "land_orange_jacket.png")]:
            viewport.set_active_camera("/World/Cameras/" + camera)
            for _ in range(65):
                await app.next_update_async()
            output = result / name
            await capture_viewport_to_file(viewport, str(output)).wait_for_result()
            await png_complete(output)
            report["screenshots"][name] = {"bytes": output.stat().st_size, "sha256": sha(output)}
        from robot_task_ui import show_robot_tools
        panel = show_robot_tools(PROJECT)
        panel.camera("Crossing")
        missions = {m["id"] for m in panel.scene_cfg["missions"]}
        if not {t["id"] for t in cfg["targets"]}.issubset(missions):
            raise RuntimeError("Both target missions must be available in the robot panel")
        after = {s: sha(PROJECT / s) for s in protected}
        if before != after:
            raise RuntimeError("Existing source assets were modified")
        task_file = PROJECT / "configs/task.json"
        task = json.loads(task_file.read_text())
        (result / "task_before.json").write_text(json.dumps(task, indent=2) + "\n")
        task.update(task_name="river_robot_jackets_v2", scene_file=cfg["scene_file"], target_config="configs/jacket_targets_v2.json",
                    description="Own river with orange flat-top land jacket and orange/red pointed water jacket; manual controls and two target mission references.")
        task_file.write_text(json.dumps(task, indent=2) + "\n")
        report.update(passed=True, water_geometry=water_info, protected_sources_before=before, protected_sources_after=after,
                      navigation_started=False, gui_left_open=True, ui_target_missions=sorted(missions & {t["id"] for t in cfg["targets"]}))
        (result / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
        print("L17_JACKETS_V2_VERIFIED " + str(scene), flush=True)
        # Keep this verified stage and the manual control panel open for the user.
    except Exception:
        (result / "error.txt").write_text(traceback.format_exc())
        print("L17_JACKETS_V2_ERROR\n" + traceback.format_exc(), flush=True)
        await omni.usd.get_context().close_stage_async()
        app.post_uncancellable_quit(3)


_JACKETS_V2_TASK = asyncio.ensure_future(main())
