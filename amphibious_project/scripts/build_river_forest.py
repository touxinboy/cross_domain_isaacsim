"""Build and verify the L17 forest river scene inside the installed Isaac Sim GUI."""

import asyncio
import hashlib
import json
import math
from pathlib import Path
import random
import sys
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
sys.path.insert(0, str(PROJECT / "scripts"))
from river_geometry import RiverGeometry


def guarded(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT):
        raise RuntimeError(f"Path escapes L17: {path}")
    return path


def material(stage, name, color, static=0.8, dynamic=0.6, vertex_color=False, roughness=0.9, opacity=1.0):
    path = f"/World/Materials/{name}"
    m = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    diffuse = shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
    diffuse.Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
    shader.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(1.333 if name == "Water" else 1.5)
    if vertex_color:
        reader = UsdShade.Shader.Define(stage, path + "/VertexColor")
        reader.CreateIdAttr("UsdPrimvarReader_float3")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("displayColor")
        reader.CreateOutput("result", Sdf.ValueTypeNames.Float3)
        diffuse.ConnectToSource(reader.ConnectableAPI(), "result")
    m.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    physics = UsdPhysics.MaterialAPI.Apply(m.GetPrim())
    physics.CreateStaticFrictionAttr(static)
    physics.CreateDynamicFrictionAttr(dynamic)
    physics.CreateRestitutionAttr(0.0)
    return m


def bind(prim, m):
    api = UsdShade.MaterialBindingAPI.Apply(prim)
    api.Bind(m)
    api.Bind(m, materialPurpose="physics")


def mesh(stage, path, points, triangles, mat, collision=False, approximation="none", colors=None, smooth=False):
    obj = UsdGeom.Mesh.Define(stage, path)
    obj.CreatePointsAttr(points)
    obj.CreateFaceVertexCountsAttr([3] * len(triangles))
    obj.CreateFaceVertexIndicesAttr([i for face in triangles for i in face])
    obj.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    obj.CreateOrientationAttr(UsdGeom.Tokens.rightHanded)
    obj.CreateDoubleSidedAttr(True)
    lo = [min(p[k] for p in points) for k in range(3)]
    hi = [max(p[k] for p in points) for k in range(3)]
    obj.CreateExtentAttr([Gf.Vec3f(*lo), Gf.Vec3f(*hi)])
    if colors:
        pv = obj.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex)
        pv.Set([Gf.Vec3f(*c) for c in colors])
    if smooth:
        normals = [Gf.Vec3f(0) for _ in points]
        for a, b, c in triangles:
            n = Gf.Cross(Gf.Vec3f(*points[b]) - Gf.Vec3f(*points[a]), Gf.Vec3f(*points[c]) - Gf.Vec3f(*points[a]))
            for index in (a, b, c):
                normals[index] += n
        obj.CreateNormalsAttr([n.GetNormalized() if n.GetLength() else Gf.Vec3f(0, 0, 1) for n in normals])
        obj.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    else:
        normals = []
        for a, b, c in triangles:
            n = Gf.Cross(Gf.Vec3f(*points[b]) - Gf.Vec3f(*points[a]), Gf.Vec3f(*points[c]) - Gf.Vec3f(*points[a]))
            n = n.GetNormalized() if n.GetLength() else Gf.Vec3f(0, 0, 1)
            normals.extend([n, n, n])
        obj.CreateNormalsAttr(normals)
        obj.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
    bind(obj.GetPrim(), mat)
    if collision:
        UsdPhysics.CollisionAPI.Apply(obj.GetPrim()).CreateCollisionEnabledAttr(True)
        UsdPhysics.MeshCollisionAPI.Apply(obj.GetPrim()).CreateApproximationAttr(approximation)
    return obj


def cylinder(stage, path, radius, height, xyz, mat, collision=False):
    obj = UsdGeom.Cylinder.Define(stage, path)
    obj.CreateRadiusAttr(radius)
    obj.CreateHeightAttr(height)
    obj.CreateAxisAttr(UsdGeom.Tokens.z)
    obj.AddTranslateOp().Set(Gf.Vec3d(*xyz))
    bind(obj.GetPrim(), mat)
    if collision:
        UsdPhysics.CollisionAPI.Apply(obj.GetPrim()).CreateCollisionEnabledAttr(True)
    return obj


def camera(stage, name, eye, target, focal=27):
    obj = UsdGeom.Camera.Define(stage, f"/World/Cameras/{name}")
    obj.CreateFocalLengthAttr(focal)
    obj.CreateHorizontalApertureAttr(36)
    obj.CreateVerticalApertureAttr(20.25)
    obj.CreateClippingRangeAttr(Gf.Vec2f(0.05, 500))
    matrix = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1)).GetInverse()
    obj.AddTransformOp().Set(matrix)


def make_terrain(stage, field, materials):
    points, groups = field.grid()
    statistics = {}
    for name, faces in groups.items():
        used = sorted({i for face in faces for i in face})
        remap = {old: new for new, old in enumerate(used)}
        local_points = [points[i] for i in used]
        local_faces = [tuple(remap[i] for i in face) for face in faces]
        color = field.config["materials"][name]["color"]
        colors = []
        for x, y, z in local_points:
            variation = 1 + 0.10 * math.sin(0.75 * x + 0.31 * y) * math.cos(0.55 * y - 0.2 * x)
            colors.append(tuple(c * variation for c in color))
        obj = mesh(stage, f"/World/Terrain/{name}", local_points, local_faces, materials[name], True,
                   colors=colors, smooth=True)
        obj.GetPrim().SetCustomDataByKey("surface_class", name)
        statistics[name] = {"triangles": len(faces), **field.config["materials"][name]}
    # A visible perimeter makes the finite terrain tile legible. It is not an extra collision floor.
    boundary = []
    for ix in range(field.nx + 1):
        boundary.append(points[ix])
    for iy in range(1, field.ny + 1):
        boundary.append(points[iy * (field.nx + 1) + field.nx])
    for ix in range(field.nx - 1, -1, -1):
        boundary.append(points[field.ny * (field.nx + 1) + ix])
    for iy in range(field.ny - 1, 0, -1):
        boundary.append(points[iy * (field.nx + 1)])
    skirt_points, skirt_faces = [], []
    for i, p in enumerate(boundary):
        q = boundary[(i + 1) % len(boundary)]
        k = len(skirt_points)
        skirt_points.extend([p, q, (q[0], q[1], -3.0), (p[0], p[1], -3.0)])
        skirt_faces.extend([(k, k + 1, k + 2), (k, k + 2, k + 3)])
    mesh(stage, "/World/Terrain/Perimeter", skirt_points, skirt_faces, materials["EarthSide"])
    return statistics


def make_tree(stage, tree, index, mats):
    root = f"/World/Vegetation/Tree_{index:03d}"
    UsdGeom.Xform.Define(stage, root)
    x, y, z, h = tree["x"], tree["y"], tree["z"], tree["height"]
    cylinder(stage, root + "/Trunk", tree["radius"], h * 0.88,
             (x, y, z + h * 0.44), mats["Bark"], True)
    rng = random.Random(300 + tree["variant"])
    points, faces, colors = [], [], []
    yaw = math.radians(tree["yaw"])
    for level in range(8):
        fraction = level / 8
        bottom = z + h * (0.18 + 0.095 * level)
        radius = tree["crown_radius"] * (1 - fraction) ** 0.70
        peak = bottom + h * (0.22 - 0.06 * fraction)
        k = len(points)
        points.append((x, y, peak))
        colors.append((0.15, 0.27, 0.09))
        n = 18
        for j in range(n):
            angle = j * 2 * math.pi / n + yaw + level * 0.45
            r = radius * rng.uniform(0.77, 1.15)
            points.append((x + r * math.cos(angle), y + r * math.sin(angle),
                           bottom + h * rng.uniform(-0.025, 0.025)))
            light = rng.uniform(0.85, 1.20)
            colors.append((0.15 * light, 0.26 * light, 0.085 * light))
        points.append((x, y, bottom + 0.07))
        colors.append((0.10, 0.17, 0.06))
        for j in range(n):
            a, b = k + 1 + j, k + 1 + (j + 1) % n
            faces.extend([(k, a, b), (k + n + 1, b, a)])
    mesh(stage, root + "/Crown", points, faces, mats["Needles"], colors=colors)


def make_rock(stage, rock, index, mat):
    rng = random.Random(900 + rock["variant"])
    x, y, z = rock["x"], rock["y"], rock["z"]
    rx, ry, rz = rock["radii"]
    yaw = math.radians(rock["yaw"])
    points, colors = [], []
    n, rings = 14, 7
    points.append((x, y, z + 1.55 * rz))
    colors.append((0.36, 0.37, 0.33))
    for j in range(1, rings):
        phi = math.pi * j / rings
        for i in range(n):
            theta = i * 2 * math.pi / n
            jitter = rng.uniform(0.89, 1.10)
            a, b = rx * math.sin(phi) * math.cos(theta) * jitter, ry * math.sin(phi) * math.sin(theta) * jitter
            points.append((x + a * math.cos(yaw) - b * math.sin(yaw), y + a * math.sin(yaw) + b * math.cos(yaw),
                           z + 0.55 * rz + rz * math.cos(phi) * jitter))
            brightness = rng.uniform(0.86, 1.14)
            colors.append((0.37 * brightness, 0.38 * brightness, 0.34 * brightness))
    points.append((x, y, z - 0.45 * rz))
    colors.append((0.25, 0.26, 0.23))
    faces = []
    for i in range(n):
        faces.append((0, 1 + i, 1 + (i + 1) % n))
    for j in range(rings - 2):
        for i in range(n):
            a, b = 1 + j * n + i, 1 + j * n + (i + 1) % n
            c, d = a + n, b + n
            faces.extend([(a, c, d), (a, d, b)])
    bottom = len(points) - 1
    last = 1 + (rings - 2) * n
    for i in range(n):
        faces.append((bottom, last + (i + 1) % n, last + i))
    obj = mesh(stage, f"/World/Obstacles/Rock_{index:03d}", points, faces, mat,
               True, "convexHull", colors=colors, smooth=True)
    obj.GetPrim().SetCustomDataByKey("obstacle_type", "rock")


def make_reeds(stage, field, material):
    rng = random.Random(720)
    points, triangles, colors = [], [], []
    for i in range(90):
        y = rng.uniform(-field.half_y + 1, field.half_y - 1)
        s = field.cross_section(y)
        sign = rng.choice([-1, 1])
        x = s["center_x"] + sign * (s["shore_half_width"] + rng.uniform(0.12, 0.55))
        z = field.mesh_height(x, y)
        for _ in range(7):
            angle = rng.uniform(0, 2 * math.pi)
            a, b = x + rng.uniform(-0.17, 0.17), y + rng.uniform(-0.17, 0.17)
            height, width = rng.uniform(0.25, 0.7), rng.uniform(0.025, 0.05)
            k = len(points)
            points.extend([(a - width * math.sin(angle), b + width * math.cos(angle), z),
                           (a + width * math.sin(angle), b - width * math.cos(angle), z),
                           (a + 0.15 * math.cos(angle), b + 0.15 * math.sin(angle), z + height)])
            triangles.append((k, k + 1, k + 2))
            colors.extend([(0.27, 0.32, 0.11), (0.26, 0.32, 0.10), (0.47, 0.49, 0.23)])
    mesh(stage, "/World/Vegetation/ShoreReeds", points, triangles, material, colors=colors)


def make_markers(stage, field, mission, mats):
    for name, key, matname in [("Start", "start_xy_m", "Start"), ("Goal", "goal_xy_m", "Goal")]:
        x, y = mission[key]
        z = field.mesh_height(x, y)
        root = UsdGeom.Xform.Define(stage, f"/World/TaskMarkers/{name}")
        root.AddTranslateOp().Set(Gf.Vec3d(x, y, z))
        cylinder(stage, str(root.GetPath()) + "/Disc", 0.8, 0.04, (0, 0, 0.04), mats[matname])
        cylinder(stage, str(root.GetPath()) + "/Pole", 0.025, 1.8, (0, 0, 0.9), mats["MarkerPole"])
        mesh(stage, str(root.GetPath()) + "/Flag", [(0, 0, 1.8), (0.9, 0, 1.6), (0, 0, 1.2)],
             [(0, 1, 2)], mats[matname])


def build_scene(stage, config, field):
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    stage.SetTimeCodesPerSecond(60)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(1200)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    for name in ["Terrain", "Vegetation", "Obstacles", "Water", "Lighting", "Cameras", "TaskMarkers"]:
        UsdGeom.Xform.Define(stage, "/World/" + name)
    UsdGeom.Scope.Define(stage, "/World/Materials")
    scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    scene.CreateGravityMagnitudeAttr(9.81)
    physics = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
    physics.CreateTimeStepsPerSecondAttr(60)
    physics.CreateEnableGPUDynamicsAttr(False)
    physics.CreateBroadphaseTypeAttr("MBP")
    materials = {}
    for name, m in config["materials"].items():
        materials[name] = material(stage, name, m["color"], m["static_friction"], m["dynamic_friction"], True)
    for name, color, use_color in [("EarthSide", (0.24, 0.20, 0.13), False), ("Bark", (0.22, 0.16, 0.09), False),
                                   ("Needles", (0.17, 0.28, 0.10), False), ("Rock", (0.37, 0.38, 0.34), True),
                                   ("Reeds", (0.30, 0.35, 0.12), True), ("Start", (0.08, 0.50, 0.24), False),
                                   ("Goal", (0.80, 0.16, 0.08), False), ("MarkerPole", (0.7, 0.72, 0.7), False)]:
        materials[name] = material(stage, name, color, vertex_color=use_color)
    terrain_stats = make_terrain(stage, field, materials)
    trees, rocks = field.objects()
    for i, tree in enumerate(trees):
        make_tree(stage, tree, i, materials)
    for i, rock in enumerate(rocks):
        make_rock(stage, rock, i, materials["Rock"])
    make_reeds(stage, field, materials["Reeds"])
    water_material = material(stage, "Water", (0.035, 0.29, 0.35), roughness=0.14, opacity=0.78)
    water_points, water_faces = field.water_mesh()
    water = mesh(stage, "/World/Water/Surface", water_points, water_faces, water_material)
    water.GetPrim().CreateAttribute("river:level", Sdf.ValueTypeNames.Double).Set(config["water_level_m"])
    water.GetPrim().CreateAttribute("river:fluidDensity", Sdf.ValueTypeNames.Double).Set(config["freshwater_density_kg_m3"])
    water.GetPrim().SetCustomDataByKey("water_dynamics_enabled", False)
    water.GetPrim().SetCustomDataByKey("purpose_note", "Visual free surface only. Buoyancy and hydrodynamics are not active.")
    mission = next(m for m in config["missions"] if m["id"] == config["default_mission"])
    make_markers(stage, field, mission, materials)
    sun = UsdLux.DistantLight.Define(stage, "/World/Lighting/Sun")
    sun.CreateIntensityAttr(1800)
    sun.CreateAngleAttr(0.7)
    sun.CreateColorAttr(Gf.Vec3f(1.0, 0.96, 0.88))
    UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-45, -22, -25))
    sky = UsdLux.DomeLight.Define(stage, "/World/Lighting/Sky")
    sky.CreateIntensityAttr(300)
    sky.CreateColorAttr(Gf.Vec3f(0.82, 0.9, 1.0))
    camera(stage, "Overview", (43, -52, 44), (-3, -2, -8), 25)
    camera(stage, "Crossing", (12, -34, 10), (-1.4, -23, 0.1), 24)
    s = field.cross_section(-13)
    camera(stage, "Riverbed", (s["center_x"], -19, -0.20), (s["center_x"], -8, -1.1), 20)
    stage.GetRootLayer().comment = "L17 own procedural river forest. Colliders active; water dynamics and robot are not implemented."
    stage.GetRootLayer().customLayerData = {"project": "L17", "scene": config["scene_name"], "seed": config["seed"]}
    return {"terrain_materials": terrain_stats, "trees": trees, "rocks": rocks,
            "terrain_triangles": sum(m["triangles"] for m in terrain_stats.values())}


async def verify_scene(stage, field, objects):
    app = omni.kit.app.get_app()
    timeline = omni.timeline.get_timeline_interface()
    carb.settings.get_settings().set_bool("/physics/updateToUsd", True)
    test_points = [field.probes()[i] for i in [0, 4, 7, 17, 27]]
    root = "/World/TemporaryVerification"
    UsdGeom.Xform.Define(stage, root)
    for i, point in enumerate(test_points):
        sphere = UsdGeom.Sphere.Define(stage, f"{root}/Drop_{i}")
        sphere.CreateRadiusAttr(0.12)
        sphere.AddTranslateOp().Set(Gf.Vec3d(point["x"], point["y"], point["z"] + 2.0))
        UsdPhysics.CollisionAPI.Apply(sphere.GetPrim())
        UsdPhysics.RigidBodyAPI.Apply(sphere.GetPrim())
        UsdPhysics.MassAPI.Apply(sphere.GetPrim()).CreateMassAttr(1.0)
    timeline.set_current_time(0)
    timeline.play()
    deadline = time.monotonic() + 40
    while timeline.get_current_time() < 3.0:
        await app.next_update_async()
        if time.monotonic() > deadline:
            raise RuntimeError("Physics did not reach three simulated seconds.")
    drops = []
    for i, point in enumerate(test_points):
        p = UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(f"{root}/Drop_{i}")).ExtractTranslation()
        # A free sphere on an inclined riverbed can drift; compare against its final XY mesh height.
        expected = field.mesh_height(float(p[0]), float(p[1])) + 0.12
        drops.append({"position_m": list(p), "expected_center_z_m": expected,
                      "passed": abs(float(p[2]) - expected) < 0.075})
    query = omni.physx.get_physx_scene_query_interface()
    rays = []
    for p in field.probes():
        hit = query.raycast_closest((p["x"], p["y"], 8.0), (0, 0, -1), 14)
        z = float(hit["position"][2]) if hit.get("hit") else None
        path = str(hit.get("collision", ""))
        rays.append({"x": p["x"], "y": p["y"], "expected_z_m": p["z"], "actual_z_m": z,
                     "collider": path, "passed": z is not None and abs(z - p["z"]) < 0.025
                     and path.startswith("/World/Terrain/")})
    obstacle_tests = []
    for item, rootpath in [(objects["trees"][0], "/World/Vegetation/Tree_000/Trunk"),
                           (objects["rocks"][0], "/World/Obstacles/Rock_000")]:
        hit = query.raycast_closest((item["x"], item["y"], 9.0), (0, 0, -1), 15)
        path = str(hit.get("collision", ""))
        obstacle_tests.append({"expected_collider": rootpath, "actual_collider": path,
                               "passed": bool(hit.get("hit")) and path == rootpath})
    water = stage.GetPrimAtPath("/World/Water/Surface")
    water_has_no_collider = not water.HasAPI(UsdPhysics.CollisionAPI)
    timeline.stop()
    for _ in range(5):
        await app.next_update_async()
    stage.RemovePrim(root)
    return {"drop_tests": drops, "terrain_raycasts": rays, "obstacle_raycasts": obstacle_tests,
            "water_has_no_collision_floor": water_has_no_collider,
            "passed": all(t["passed"] for t in drops + rays + obstacle_tests) and water_has_no_collider}


async def main():
    guarded(PROJECT)
    config_path = guarded(PROJECT / "configs/river_forest.json")
    config = json.loads(config_path.read_text())
    field = RiverGeometry(config)
    scene_path = guarded(PROJECT / "scenes" / (config["scene_name"] + ".usda"))
    if scene_path.exists():
        raise RuntimeError(f"Refusing to overwrite an existing scene: {scene_path}")
    original = guarded(PROJECT / "scenes/river_base.usda")
    original_hash = hashlib.sha256(original.read_bytes()).hexdigest()
    results = guarded(PROJECT / "results/river_forest_v1")
    results.mkdir(exist_ok=True)
    ctx = omni.usd.get_context()
    await ctx.new_stage_async()
    stage = ctx.get_stage()
    print("L17_FOREST_BUILD_STARTED", flush=True)
    objects = build_scene(stage, config, field)
    from omni.kit.viewport.utility import get_active_viewport
    viewport = get_active_viewport()
    if viewport:
        viewport.set_active_camera("/World/Cameras/Overview")
    for _ in range(15):
        await omni.kit.app.get_app().next_update_async()
    print("L17_FOREST_PHYSICS_CHECK_STARTED", flush=True)
    verification = await verify_scene(stage, field, objects)
    report = {"scene": str(scene_path), "seed": config["seed"], "size_m": [config["width_m"], config["length_m"]],
              "water_surface_z_m": config["water_level_m"], "water_dynamics_enabled": False,
              "robot_present": False, "tree_count": len(objects["trees"]), "rock_count": len(objects["rocks"]),
              "terrain_triangles": objects["terrain_triangles"], "physics_verification": verification,
              "previous_base_scene_sha256": original_hash}
    guarded(results / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    if not verification["passed"]:
        raise RuntimeError("Physics verification failed; see results/river_forest_v1/verification.json")
    stage.GetRootLayer().Export(str(scene_path))
    success, error = await ctx.open_stage_async(str(scene_path))
    if not success:
        raise RuntimeError(f"Could not reopen the new scene: {error}")
    if hashlib.sha256(original.read_bytes()).hexdigest() != original_hash:
        raise RuntimeError("The previous base scene changed unexpectedly.")
    task_path = guarded(PROJECT / "configs/task.json")
    task = json.loads(task_path.read_text())
    task.update({"task_name": config["scene_name"], "scene_file": "scenes/" + scene_path.name,
                 "scene_config": "configs/river_forest.json",
                 "description": "Forest river terrain and mission markers; robot, control and water dynamics are not implemented."})
    task.pop("scene_geometry", None)
    task_path.write_text(json.dumps(task, indent=2) + "\n")
    from river_task_ui import show_scene_tools
    tools = show_scene_tools(PROJECT)
    await tools.capture_views()
    report["preview_files"] = ["overview.png", "crossing.png", "riverbed.png"]
    guarded(results / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"L17_FOREST_BUILD_SUCCESS: {scene_path}", flush=True)


async def run_with_report():
    try:
        await main()
    except Exception:
        error = traceback.format_exc()
        p = guarded(PROJECT / "results/river_forest_v1")
        p.mkdir(exist_ok=True)
        guarded(p / "build_error.txt").write_text(error)
        print("L17_FOREST_BUILD_ERROR\n" + error, flush=True)


asyncio.ensure_future(run_with_report())
