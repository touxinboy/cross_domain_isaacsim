"""USD construction for the own amphibious robot and its isolated validation stages."""

import hashlib
import json
import math
import os
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, PhysxSchema


def project_path(project, relative):
    root = Path("/home/ubuntu/sdc/L17").resolve(strict=True)
    path = (Path(project) / relative).resolve()
    if not path.is_relative_to(root):
        raise RuntimeError("Robot file path escapes L17.")
    return path


def material(stage, path, color, friction=0.6, track=False, roughness=0.5):
    obj = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.15)
    obj.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    physics = UsdPhysics.MaterialAPI.Apply(obj.GetPrim())
    physics.CreateStaticFrictionAttr(friction)
    physics.CreateDynamicFrictionAttr(friction)
    physics.CreateRestitutionAttr(0.0)
    if track:
        PhysxSchema.PhysxMaterialAPI.Apply(obj.GetPrim()).CreateFrictionCombineModeAttr("min")
    return obj


def bind(prim, mat):
    binding = UsdShade.MaterialBindingAPI.Apply(prim)
    binding.Bind(mat)
    binding.Bind(mat, materialPurpose="physics")


def box(stage, path, size, position, mat, collision=False, rotate=None):
    obj = UsdGeom.Cube.Define(stage, path)
    obj.CreateSizeAttr(1)
    obj.AddTranslateOp().Set(Gf.Vec3d(*position))
    if rotate is not None:
        obj.AddRotateXYZOp().Set(Gf.Vec3f(*rotate))
    obj.AddScaleOp().Set(Gf.Vec3f(*size))
    bind(obj.GetPrim(), mat)
    if collision:
        UsdPhysics.CollisionAPI.Apply(obj.GetPrim())
    return obj


def cylinder(stage, path, radius, height, xyz, mat, axis="Z", collision=False):
    obj = UsdGeom.Cylinder.Define(stage, path)
    obj.CreateRadiusAttr(radius)
    obj.CreateHeightAttr(height)
    obj.CreateAxisAttr(axis)
    obj.AddTranslateOp().Set(Gf.Vec3d(*xyz))
    bind(obj.GetPrim(), mat)
    if collision:
        UsdPhysics.CollisionAPI.Apply(obj.GetPrim())
    return obj


def triangle_mesh(stage, path, points, faces, mat, collision=False):
    obj = UsdGeom.Mesh.Define(stage, path)
    obj.CreatePointsAttr(points)
    obj.CreateFaceVertexCountsAttr([3] * len(faces))
    obj.CreateFaceVertexIndicesAttr([i for face in faces for i in face])
    obj.CreateSubdivisionSchemeAttr("none")
    obj.CreateDoubleSidedAttr(True)
    normals = []
    for a, b, c in faces:
        n = Gf.Cross(Gf.Vec3f(*points[b])-Gf.Vec3f(*points[a]), Gf.Vec3f(*points[c])-Gf.Vec3f(*points[a]))
        n = n.GetNormalized() if n.GetLength() else Gf.Vec3f(0, 0, 1)
        normals.extend([n, n, n])
    obj.CreateNormalsAttr(normals)
    obj.SetNormalsInterpolation("faceVarying")
    bind(obj.GetPrim(), mat)
    if collision:
        UsdPhysics.CollisionAPI.Apply(obj.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(obj.GetPrim()).CreateApproximationAttr("convexHull")
    return obj


def stadium(distance, half_straight, radius, zcenter):
    straight, arc = 2 * half_straight, math.pi * radius
    length = 2 * (straight + arc)
    s = distance % length
    if s < straight:
        return (-half_straight+s, zcenter-radius, 0.0)
    s -= straight
    if s < arc:
        a = -math.pi/2 + s/radius
        return (half_straight+radius*math.cos(a), zcenter+radius*math.sin(a), a+math.pi/2)
    s -= arc
    if s < straight:
        return (half_straight-s, zcenter+radius, math.pi)
    a = math.pi/2 + (s-straight)/radius
    return (-half_straight+radius*math.cos(a), zcenter+radius*math.sin(a), a+math.pi/2)


def track_geometry(stage, root, y, cfg, mats):
    tr = cfg["tracks"]
    UsdGeom.Xform.Define(stage, root)
    r, half = tr["height_m"]/2, (tr["length_m"]-tr["height_m"])/2
    n, width, zc = 40, tr["width_m"], tr["center_z_m"]
    perimeter = 4*half + 2*math.pi*r
    outer = [stadium(i*perimeter/n, half, r, zc)[:2] for i in range(n)]
    inside = [stadium(i*(4*half+2*math.pi*(r-.042))/n, half, r-.042, zc)[:2] for i in range(n)]
    points = [(x, y+side*width/2, z) for side in [-1, 1] for x, z in outer]
    faces = []
    for i in range(n):
        j = (i+1) % n
        faces.extend([(i, j, n+j), (i, n+j, n+i)])
    for i in range(1, n-1):
        faces.extend([(0, i, i+1), (n, n+i+1, n+i)])
    collider = triangle_mesh(stage, root + "/Collider", points, faces, mats["TrackContact"], True)
    UsdGeom.Imageable(collider).MakeInvisible()
    # A ring belt is visible; its convex outer envelope supplies normal contact to PhysX.
    ring_points = []
    for side in [-1, 1]:
        ring_points += [(x, y+side*width/2, z) for x, z in outer]
        ring_points += [(x, y+side*width/2, z) for x, z in inside]
    ring_faces = []
    for i in range(n):
        j = (i+1) % n
        for a, b, c, d in [(i, j, n+j, n+i), (2*n+i, 3*n+i, 3*n+j, 2*n+j),
                            (i, 2*n+i, 2*n+j, j), (n+i, n+j, 3*n+j, 3*n+i)]:
            ring_faces.extend([(a, b, c), (a, c, d)])
    triangle_mesh(stage, root + "/Belt", ring_points, ring_faces, mats["Rubber"])
    for i in range(32):
        x, z, tangent = stadium(i*perimeter/32, half, r+.006, zc)
        box(stage, f"{root}/Tread_{i:02d}", (.045, width+.008, .012), (x, y, z), mats["Tread"],
            rotate=(0, -math.degrees(tangent), 0))
    for i, x in enumerate([-.36, -.18, 0., .18, .36]):
        cylinder(stage, f"{root}/Roller_{i}", .073, width*.88, (x, y, zc), mats["Metal"], "Y")
        cylinder(stage, f"{root}/Hub_{i}", .031, width*1.02, (x, y, zc), mats["Bolt"], "Y")


def make_thruster(stage, root, center, cfg, mats):
    UsdGeom.Xform.Define(stage, root)
    x, y, z = center
    radius = cfg["thrusters"]["propeller_radius_m"]
    n = 28
    points = []
    for xoff in [-.09, .05]:
        for r in [radius+.014, radius+.003]:
            points += [(x+xoff, y+r*math.cos(i*2*math.pi/n), z+r*math.sin(i*2*math.pi/n)) for i in range(n)]
    faces = []
    for i in range(n):
        j = (i+1) % n
        for a, b, c, d in [(i,j,2*n+j,2*n+i),(n+i,3*n+i,3*n+j,n+j),
                            (i,n+i,n+j,j),(2*n+i,2*n+j,3*n+j,3*n+i)]:
            faces.extend([(a,b,c),(a,c,d)])
    triangle_mesh(stage, root+"/Duct", points, faces, mats["Metal"])
    physical = cylinder(stage, root+"/Collision", radius+.014, .14, (x-.02,y,z), mats["Metal"], "X", True)
    UsdGeom.Imageable(physical).MakeInvisible()
    cylinder(stage, root+"/Hub", .018, .10, (x-.04,y,z), mats["Bolt"], "X")
    for i in range(4):
        angle = i*math.pi/2
        box(stage, f"{root}/Blade_{i}", (.012, radius*.86, .022),
            (x-.065, y+radius*.45*math.cos(angle), z+radius*.45*math.sin(angle)), mats["Propeller"],
            rotate=(math.degrees(angle),0,0))
    box(stage, root+"/Mount", (.15,.05,.06), (x+.06,y,z+.08), mats["Hull"])


def build_robot_asset(project, cfg):
    output = project_path(project, "robots/tracked_twin_thruster_v1.usda")
    canonical = {k:cfg[k] for k in ["mass_kg","inertia_diagonal_kg_m2","hull"]}
    canonical["tracks"]={k:cfg["tracks"][k] for k in ["length_m","width_m","height_m","center_separation_m","center_z_m"]}
    canonical["thrusters"]={k:cfg["thrusters"][k] for k in ["positions_body_m","propeller_radius_m"]}
    digest = hashlib.sha256(json.dumps(canonical,sort_keys=True).encode()).hexdigest()
    if output.exists():
        existing = Usd.Stage.Open(str(output))
        if existing.GetRootLayer().customLayerData.get("config_sha256") != digest:
            raise RuntimeError("Existing robot asset has a different configuration; refusing an implicit replacement.")
        return output
    output.parent.mkdir(exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output))
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdGeom.SetStageUpAxis(stage, "Z")
    root = UsdGeom.Xform.Define(stage, "/Amphibian")
    stage.SetDefaultPrim(root.GetPrim())
    root.AddTranslateOp().Set(Gf.Vec3d(0))
    root.AddOrientOp().Set(Gf.Quatf(1))
    body = UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    body.CreateRigidBodyEnabledAttr(True)
    mass = UsdPhysics.MassAPI.Apply(root.GetPrim())
    mass.CreateMassAttr(cfg["mass_kg"])
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*cfg["inertia_diagonal_kg_m2"]))
    mass.CreatePrincipalAxesAttr(Gf.Quatf(1))
    physx = PhysxSchema.PhysxRigidBodyAPI.Apply(root.GetPrim())
    physx.CreateLinearDampingAttr(0)
    physx.CreateAngularDampingAttr(0)
    physx.CreateSleepThresholdAttr(0)
    physx.CreateSolverPositionIterationCountAttr(16)
    physx.CreateSolverVelocityIterationCountAttr(8)
    report = PhysxSchema.PhysxContactReportAPI.Apply(root.GetPrim())
    report.CreateThresholdAttr(0)
    UsdGeom.Scope.Define(stage, "/Amphibian/Looks")
    mats = {}
    for name,color in [("Hull",(.82,.24,.045)),("Deck",(.20,.23,.25)),("Rubber",(.035,.042,.045)),
                       ("Tread",(.08,.085,.09)),("Metal",(.16,.20,.23)),("Bolt",(.55,.58,.60)),
                       ("Propeller",(.07,.09,.11)),("Glass",(.025,.08,.12))]:
        mats[name]=material(stage,"/Amphibian/Looks/"+name,color)
    mats["TrackContact"]=material(stage,"/Amphibian/Looks/TrackContact",(.03,.03,.03),0,True)
    box(stage,"/Amphibian/Hull",cfg["hull"]["size_m"],cfg["hull"]["center_body_m"],mats["Hull"],True)
    box(stage,"/Amphibian/Deck",(.80,.48,.035),(0,0,.245),mats["Deck"])
    box(stage,"/Amphibian/Hatch",(.32,.34,.035),(-.05,0,.277),mats["Hull"])
    box(stage,"/Amphibian/FrontWindow",(.08,.28,.065),(.36,0,.27),mats["Glass"])
    cylinder(stage,"/Amphibian/SensorMast",.018,.20,(.16,0,.35),mats["Bolt"])
    cylinder(stage,"/Amphibian/SensorHead",.055,.05,(.16,0,.465),mats["Metal"])
    UsdGeom.Xform.Define(stage,"/Amphibian/Tracks")
    half=cfg["tracks"]["center_separation_m"]/2
    track_geometry(stage,"/Amphibian/Tracks/Left",half,cfg,mats)
    track_geometry(stage,"/Amphibian/Tracks/Right",-half,cfg,mats)
    UsdGeom.Xform.Define(stage,"/Amphibian/Thrusters")
    for name,pos in zip(["Left","Right"],cfg["thrusters"]["positions_body_m"]):
        make_thruster(stage,"/Amphibian/Thrusters/"+name,pos,cfg,mats)
    root.GetPrim().SetCustomDataByKey("drive_model","equivalent two-track contact patches; no discrete belt links")
    stage.GetRootLayer().customLayerData={"creator":"L17","config_sha256":digest,"mass_kg":cfg["mass_kg"]}
    stage.GetRootLayer().Save()
    return output


def add_robot(stage, project, asset, xyz, roll=0, pitch=0, yaw=0):
    root=stage.DefinePrim("/World/Robot","Xform")
    root.GetReferences().AddReference(os.path.relpath(str(asset),str(Path(stage.GetRootLayer().realPath).parent)))
    root.GetAttribute("xformOp:translate").Set(Gf.Vec3d(*xyz))
    rotation=Gf.Rotation(Gf.Vec3d(0,0,1),math.degrees(yaw))*Gf.Rotation(Gf.Vec3d(0,1,0),math.degrees(pitch))*Gf.Rotation(Gf.Vec3d(1,0,0),math.degrees(roll))
    q=rotation.GetQuat()
    root.GetAttribute("xformOp:orient").Set(Gf.Quatf(q.GetReal(),Gf.Vec3f(q.GetImaginary())))
    return root


def set_camera(stage, path, eye, target, focal=28):
    camera=UsdGeom.Camera.Define(stage,path)
    camera.CreateFocalLengthAttr(focal)
    camera.CreateHorizontalApertureAttr(36)
    camera.CreateVerticalApertureAttr(20.25)
    camera.CreateClippingRangeAttr(Gf.Vec2f(.03,250))
    transform=Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye),Gf.Vec3d(*target),Gf.Vec3d(0,0,1)).GetInverse()
    ops=camera.GetOrderedXformOps()
    if ops:ops[0].Set(transform)
    else:camera.AddTransformOp().Set(transform)


def set_physics(stage,cfg):
    scene=UsdPhysics.Scene.Define(stage,"/World/PhysicsScene")
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0,0,-1))
    scene.CreateGravityMagnitudeAttr(cfg["gravity_m_s2"])
    p=PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
    p.CreateTimeStepsPerSecondAttr(cfg["physics_hz"])
    p.CreateEnableGPUDynamicsAttr(False)
    p.CreateBroadphaseTypeAttr("MBP")
    stage.SetTimeCodesPerSecond(60)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(36000)


def create_validation_stage(project,cfg,asset,kind):
    output=project_path(project,"scenes/robot_validation_"+kind+".usda")
    if output.exists():
        layer=Sdf.Layer.FindOrOpen(str(output))
        layer.startTimeCode=0;layer.endTimeCode=36000;layer.Save()
        return output
    stage=Usd.Stage.CreateNew(str(output))
    UsdGeom.SetStageMetersPerUnit(stage,1);UsdGeom.SetStageUpAxis(stage,"Z")
    world=UsdGeom.Xform.Define(stage,"/World");stage.SetDefaultPrim(world.GetPrim())
    set_physics(stage,cfg)
    UsdGeom.Scope.Define(stage,"/World/Looks")
    ground=material(stage,"/World/Looks/Ground",(.37,.39,.33),.8)
    water=material(stage,"/World/Looks/Water",(.03,.29,.36),.0,roughness=.14)
    water_shader=UsdShade.Shader(stage.GetPrimAtPath("/World/Looks/Water/Surface"))
    water_shader.CreateInput("opacity",Sdf.ValueTypeNames.Float).Set(.8)
    z=0 if kind=="land" else -3
    box(stage,"/World/Ground",(32,24,.5),(0,0,z-.25),ground,True)
    if kind!="land":
        triangle_mesh(stage,"/World/Water",[(-16,-12,0),(16,-12,0),(16,12,0),(-16,12,0)],[(0,1,2),(0,2,3)],water)
    initial=(-4,0,.75) if kind=="land" else (-4,0,.45)
    add_robot(stage,project,asset,initial)
    sky=UsdLux.DomeLight.Define(stage,"/World/Sky");sky.CreateIntensityAttr(400)
    sun=UsdLux.DistantLight.Define(stage,"/World/Sun");sun.CreateIntensityAttr(1800)
    UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-40,-20,-35))
    set_camera(stage,"/World/Cameras/Overview",(9,-13,9),(0,0,-.1),28)
    set_camera(stage,"/World/Cameras/Robot",(-5.9,-2.3,1.5),(-4,0,.1),28)
    stage.GetRootLayer().customLayerData={"creator":"L17","validation_kind":kind,"water_dynamics_requires_runtime":True}
    stage.GetRootLayer().Save()
    return output
