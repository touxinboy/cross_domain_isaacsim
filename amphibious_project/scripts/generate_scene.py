"""Create a self-contained USD scene using the installed Isaac Sim Python."""

import json
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade


PROJECT = Path(__file__).resolve().parents[1]


def cube(stage, path, position, dimensions, color, collision=False):
    shape = UsdGeom.Cube.Define(stage, path)
    shape.CreateSizeAttr(1.0)
    shape.AddTranslateOp().Set(Gf.Vec3d(*position))
    shape.AddScaleOp().Set(Gf.Vec3f(*dimensions))
    shape.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])
    if collision:
        UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
    return shape


def generate():
    config = json.loads((PROJECT / "configs/task.json").read_text())
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    stage.SetTimeCodesPerSecond(60.0)

    physics = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    physics.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
    physics.CreateGravityMagnitudeAttr(config["gravity_m_s2"])

    material = UsdShade.Material.Define(stage, "/World/Materials/Ground")
    friction = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    friction.CreateStaticFrictionAttr(0.8)
    friction.CreateDynamicFrictionAttr(0.65)
    friction.CreateRestitutionAttr(0.0)

    profile = config["river"]["terrain_profile_xz_m"]
    half_y = config["river"]["half_length_y_m"]
    names = ["LeftBank", "LeftShore", "Riverbed", "RightShore", "RightBank"]
    colors = [(0.30, 0.46, 0.22), (0.57, 0.43, 0.25), (0.32, 0.30, 0.27),
              (0.57, 0.43, 0.25), (0.30, 0.46, 0.22)]
    for index, ((x0, z0), (x1, z1)) in enumerate(zip(profile, profile[1:])):
        mesh = UsdGeom.Mesh.Define(stage, f"/World/Terrain/{names[index]}")
        mesh.CreatePointsAttr([(x0, -half_y, z0), (x1, -half_y, z1),
                               (x1, half_y, z1), (x0, half_y, z0)])
        mesh.CreateFaceVertexCountsAttr([4])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.GetDisplayColorAttr().Set([Gf.Vec3f(*colors[index])])
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material, materialPurpose="physics")

    water = cube(stage, "/World/WaterVisual", (0.0, 0.0, -0.02),
                 (9.0, 2.0 * half_y, 0.04), (0.10, 0.42, 0.72))
    water.GetDisplayOpacityAttr().Set([0.55])
    water.GetPrim().SetCustomDataByKey("simulation_note", "Visual surface only: no fluid forces implemented yet.")

    robot = UsdGeom.Xform.Define(stage, "/World/Robot")
    robot.AddTranslateOp().Set(Gf.Vec3d(*config["robot"]["initial_position_m"]))
    UsdPhysics.RigidBodyAPI.Apply(robot.GetPrim())
    UsdPhysics.MassAPI.Apply(robot.GetPrim()).CreateMassAttr(config["robot"]["mass_kg"])
    robot.GetPrim().SetCustomDataByKey("model_status", "Compound rigid-body placeholder; actuator models pending.")
    cube(stage, "/World/Robot/Chassis", (0.0, 0.0, 0.28), (1.10, 0.64, 0.32),
         (0.83, 0.48, 0.12), collision=True)
    for side, y in (("Left", 0.43), ("Right", -0.43)):
        cube(stage, f"/World/Robot/{side}Track", (0.0, y, 0.12), (1.18, 0.20, 0.24),
             (0.08, 0.09, 0.10), collision=True)
        thruster = UsdGeom.Cylinder.Define(stage, f"/World/Robot/{side}ThrusterVisual")
        thruster.CreateAxisAttr("X")
        thruster.CreateRadiusAttr(0.10)
        thruster.CreateHeightAttr(0.22)
        thruster.AddTranslateOp().Set(Gf.Vec3d(-0.66, y * 0.6, 0.23))
        thruster.GetDisplayColorAttr().Set([Gf.Vec3f(0.12, 0.16, 0.22)])

    goal_x, goal_y = config["robot"]["goal_xy_m"]
    goal = UsdGeom.Cylinder.Define(stage, "/World/GoalMarker")
    goal.CreateRadiusAttr(0.42)
    goal.CreateHeightAttr(0.025)
    goal.AddTranslateOp().Set(Gf.Vec3d(goal_x, goal_y, 0.52))
    goal.GetDisplayColorAttr().Set([Gf.Vec3f(0.10, 0.85, 0.20)])

    light = UsdLux.DistantLight.Define(stage, "/World/Sun")
    light.CreateIntensityAttr(2500.0)
    light.AddRotateXYZOp().Set(Gf.Vec3f(-35.0, -25.0, 0.0))
    dome = UsdLux.DomeLight.Define(stage, "/World/Sky")
    dome.CreateIntensityAttr(500.0)

    camera = UsdGeom.Camera.Define(stage, "/World/OverviewCamera")
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.1, 500.0))
    camera.CreateFocalLengthAttr(24.0)
    view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(16.0, -20.0, 15.0),
                                 Gf.Vec3d(0.0, 0.0, -0.1), Gf.Vec3d(0.0, 0.0, 1.0))
    camera.AddTransformOp().Set(view.GetInverse())

    target = PROJECT / "scenes/river_stage_01.usda"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not stage.GetRootLayer().Export(str(target)):
        raise RuntimeError(f"Could not export {target}")
    report = {"scene": str(target), "prim_count": len(list(stage.Traverse())),
              "external_references": list(stage.GetRootLayer().GetExternalReferences()),
              "water_dynamics_implemented": False, "navigation_implemented": False}
    print(json.dumps(report, indent=2))
    return target


if __name__ == "__main__":
    generate()
