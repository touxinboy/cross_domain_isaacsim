"""Rebuild the authorized drop_test scene as a dry river channel in the own GUI."""

import asyncio
import json
import math
from pathlib import Path
import time
import traceback

import carb
import omni.kit.app
import omni.physx
import omni.timeline
import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, PhysxSchema


PROJECT = Path(__file__).resolve().parents[1]
ROOT = Path("/home/ubuntu/sdc/L17").resolve(strict=True)
SOURCE = PROJECT / "scenes/drop_test.usd"
TARGET = PROJECT / "scenes/river_base.usda"
RESULTS = PROJECT / "results"
REPORT = RESULTS / "river_base_verification.json"
PROFILE = [(-10.0, 0.0), (-6.0, 0.0), (-2.0, -1.5), (2.0, -1.5), (6.0, 0.0), (10.0, 0.0)]
SEGMENTS = [
    ("LeftBank", (0.23, 0.34, 0.16), 0.95, 0.80),
    ("LeftRamp", (0.55, 0.39, 0.20), 0.85, 0.65),
    ("Riverbed", (0.30, 0.23, 0.15), 0.70, 0.50),
    ("RightRamp", (0.55, 0.39, 0.20), 0.85, 0.65),
    ("RightBank", (0.23, 0.34, 0.16), 0.95, 0.80),
]


def guarded(path):
    resolved = path.resolve()
    if not resolved.is_relative_to(ROOT):
        raise RuntimeError(f"Write escapes L17: {path}")
    return resolved


def terrain_material(stage, name, color, static_friction, dynamic_friction):
    material = UsdShade.Material.Define(stage, f"/World/Materials/{name}")
    shader = UsdShade.Shader.Define(stage, f"/World/Materials/{name}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.9)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics.CreateStaticFrictionAttr(static_friction)
    physics.CreateDynamicFrictionAttr(dynamic_friction)
    physics.CreateRestitutionAttr(0.0)
    return material


def terrain_segment(stage, index):
    name, color, static_friction, dynamic_friction = SEGMENTS[index]
    (x0, z0), (x1, z1) = PROFILE[index:index + 2]
    bottom, y0, y1 = -2.0, -6.0, 6.0
    points = [(x0, y0, bottom), (x1, y0, bottom), (x1, y1, bottom), (x0, y1, bottom),
              (x0, y0, z0), (x1, y0, z1), (x1, y1, z1), (x0, y1, z0)]
    quads = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
             (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    triangles = []
    normals = []
    for a, b, c, d in quads:
        for tri in [(a, b, c), (a, c, d)]:
            triangles.extend(tri)
            n = Gf.Cross(Gf.Vec3f(*points[tri[1]]) - Gf.Vec3f(*points[tri[0]]),
                         Gf.Vec3f(*points[tri[2]]) - Gf.Vec3f(*points[tri[0]])).GetNormalized()
            normals.extend([n, n, n])
    mesh = UsdGeom.Mesh.Define(stage, f"/World/Terrain/{name}")
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([3] * 12)
    mesh.CreateFaceVertexIndicesAttr(triangles)
    mesh.CreateNormalsAttr(normals)
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateOrientationAttr(UsdGeom.Tokens.rightHanded)
    mesh.CreateExtentAttr([Gf.Vec3f(x0, y0, bottom), Gf.Vec3f(x1, y1, max(z0, z1))])
    mesh.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(True)
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("none")
    material = terrain_material(stage, name, color, static_friction, dynamic_friction)
    bindings = UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim())
    bindings.Bind(material)
    bindings.Bind(material, materialPurpose="physics")
    mesh.GetPrim().SetCustomDataByKey("surface_role", name)
    mesh.GetPrim().SetCustomDataByKey("slope_degrees", math.degrees(math.atan2(abs(z1-z0), x1-x0)))


def build(stage):
    stage.SetEditTarget(stage.GetRootLayer())
    for prim in list(stage.GetPseudoRoot().GetChildren()):
        stage.RemovePrim(prim.GetPath())
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    stage.SetTimeCodesPerSecond(60)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(600)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.Xform.Define(stage, "/World/Terrain")
    UsdGeom.Scope.Define(stage, "/World/Materials")
    UsdGeom.Xform.Define(stage, "/World/Lighting")
    scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    scene.CreateGravityMagnitudeAttr(9.81)
    physx = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
    physx.CreateTimeStepsPerSecondAttr(60)
    physx.CreateEnableGPUDynamicsAttr(False)
    physx.CreateBroadphaseTypeAttr("MBP")
    for index in range(len(SEGMENTS)):
        terrain_segment(stage, index)
    sun = UsdLux.DistantLight.Define(stage, "/World/Lighting/Sun")
    sun.CreateIntensityAttr(2500.0)
    sun.CreateAngleAttr(1.0)
    UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-35, -20, -35))
    sky = UsdLux.DomeLight.Define(stage, "/World/Lighting/Sky")
    sky.CreateIntensityAttr(450.0)
    sky.CreateColorAttr(Gf.Vec3f(0.82, 0.90, 1.0))
    camera = UsdGeom.Camera.Define(stage, "/World/OverviewCamera")
    camera.CreateFocalLengthAttr(27.0)
    camera.CreateHorizontalApertureAttr(36.0)
    camera.CreateVerticalApertureAttr(24.0)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.1, 200.0))
    view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(19, -24, 21), Gf.Vec3d(0, 0, -0.6), Gf.Vec3d(0, 0, 1))
    UsdGeom.Xformable(camera).AddTransformOp().Set(view.GetInverse())
    layer = stage.GetRootLayer()
    layer.comment = "L17 own dry river channel: two banks, two ramps, riverbed; no water dynamics yet."
    layer.customLayerData = {"creator": "L17", "stage_name": "river_base", "units": "meters"}


async def verify_physics(stage):
    app = omni.kit.app.get_app()
    timeline = omni.timeline.get_timeline_interface()
    carb.settings.get_settings().set_bool("/physics/updateToUsd", True)
    tests = [("LeftBank", -8.0, 0.15), ("Riverbed", 0.0, -1.35), ("RightBank", 8.0, 0.15)]
    UsdGeom.Xform.Define(stage, "/World/TemporaryVerification")
    for name, x, _ in tests:
        sphere = UsdGeom.Sphere.Define(stage, f"/World/TemporaryVerification/{name}")
        sphere.CreateRadiusAttr(0.15)
        sphere.AddTranslateOp().Set(Gf.Vec3d(x, 0, 2))
        UsdPhysics.CollisionAPI.Apply(sphere.GetPrim())
        UsdPhysics.RigidBodyAPI.Apply(sphere.GetPrim())
        UsdPhysics.MassAPI.Apply(sphere.GetPrim()).CreateMassAttr(1.0)
    timeline.set_current_time(0.0)
    timeline.play()
    deadline = time.monotonic() + 25
    while timeline.get_current_time() < 3.0:
        await app.next_update_async()
        if time.monotonic() > deadline:
            raise RuntimeError("Physics verification did not advance to 3 simulated seconds.")
    drops = []
    for name, x, expected in tests:
        prim = stage.GetPrimAtPath(f"/World/TemporaryVerification/{name}")
        position = UsdGeom.XformCache().GetLocalToWorldTransform(prim).ExtractTranslation()
        drops.append({"surface": name, "position": list(position), "expected_center_z": expected,
                      "passed": abs(position[2] - expected) < 0.05})
    query = omni.physx.get_physx_scene_query_interface()
    rays = []
    for x, expected in [(-8.0, 0.0), (-4.0, -0.75), (0.0, -1.5), (4.0, -0.75), (8.0, 0.0)]:
        hit = query.raycast_closest((x, 3.0, 5.0), (0.0, 0.0, -1.0), 10.0)
        z = float(hit["position"][2]) if hit.get("hit") else None
        rays.append({"x": x, "surface_z": z, "expected_z": expected,
                     "collision": str(hit.get("collision", "")),
                     "passed": z is not None and abs(z - expected) < 0.02})
    timeline.stop()
    for _ in range(5):
        await app.next_update_async()
    stage.RemovePrim("/World/TemporaryVerification")
    return {"drop_tests": drops, "surface_raycasts": rays,
            "passed": all(x["passed"] for x in drops + rays)}


async def main():
    guarded(PROJECT)
    guarded(TARGET)
    guarded(RESULTS).mkdir(exist_ok=True)
    ctx = omni.usd.get_context()
    timeline = omni.timeline.get_timeline_interface()
    timeline.stop()
    if not SOURCE.is_file():
        raise RuntimeError(f"The authorized source scene is missing: {SOURCE}")
    if TARGET.exists():
        raise RuntimeError(f"Refusing to overwrite an existing destination: {TARGET}")
    success, error = await ctx.open_stage_async(str(SOURCE))
    if not success:
        raise RuntimeError(f"Cannot open drop_test: {error}")
    stage = ctx.get_stage()
    build(stage)
    from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file
    viewport = get_active_viewport()
    if viewport:
        viewport.set_active_camera("/World/OverviewCamera")
    for _ in range(20):
        await omni.kit.app.get_app().next_update_async()
    verification = await verify_physics(stage)
    report = {"source": str(SOURCE), "scene": str(TARGET), "dimensions_m": [20, 12],
              "bank_height_m": 0, "riverbed_height_m": -1.5, "flat_riverbed_width_m": 4,
              "ramp_run_m": 4, "ramp_slope_degrees": math.degrees(math.atan(1.5/4)),
              "cross_section_x_z": PROFILE, "water_dynamics": False,
              "verification": verification}
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if not verification["passed"]:
        raise RuntimeError(f"River collision verification failed; see {REPORT}")
    stage.GetRootLayer().Export(str(TARGET))
    checked = Usd.Stage.Open(str(TARGET))
    if len(list(checked.GetPrimAtPath("/World/Terrain").GetChildren())) != 5:
        raise RuntimeError("Saved terrain has an unexpected number of sections.")
    if checked.GetPrimAtPath("/World/TemporaryVerification"):
        raise RuntimeError("Temporary physics tests leaked into the saved scene.")
    success, error = await ctx.open_stage_async(str(TARGET))
    if not success:
        raise RuntimeError(f"Cannot reopen the finished scene: {error}")
    viewport = get_active_viewport()
    if viewport:
        viewport.set_active_camera("/World/OverviewCamera")
    config_path = guarded(PROJECT / "configs/task.json")
    config = json.loads(config_path.read_text())
    config["scene_file"] = "scenes/river_base.usda"
    config["task_name"] = "river_base"
    config["scene_geometry"] = {"cross_section_x_z": PROFILE, "half_length_y_m": 6,
                                "water_dynamics": False}
    config.pop("river", None)
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    guarded(SOURCE).unlink()
    print(f"L17_RIVER_BUILD_SUCCESS: {TARGET}", flush=True)
    for _ in range(40):
        await omni.kit.app.get_app().next_update_async()
    if viewport:
        capture = capture_viewport_to_file(viewport, str(guarded(RESULTS / "river_base_preview.png")))
        await capture.wait_for_result()
        print("L17_RIVER_PREVIEW_READY", flush=True)


async def run_with_report():
    try:
        await main()
    except Exception:
        error = traceback.format_exc()
        if RESULTS.is_dir():
            guarded(RESULTS / "river_build_error.txt").write_text(error)
        print("L17_RIVER_BUILD_ERROR\n" + error, flush=True)


asyncio.ensure_future(run_with_report())
