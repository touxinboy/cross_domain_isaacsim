"""Instanced circulating shoes, guide horns, pins and rotating wheel groups."""
import math
import numpy as np
from pxr import Gf, Usd, UsdGeom, Vt

from robot_assets import box, cylinder, material


class TrackVisuals:
    ROOT='/World/Robot/TrackMotion'

    @classmethod
    def clear(cls,stage):
        with Usd.EditContext(stage,stage.GetSessionLayer()):
            if stage.GetPrimAtPath(cls.ROOT):
                stage.RemovePrim(cls.ROOT)
            UsdGeom.Imageable(stage.GetPrimAtPath('/World/Robot/Tracks')).GetVisibilityAttr().Set('inherited')

    def __init__(self,stage,motion):
        self.stage,self.motion = stage,motion
        self.next_update_s=0.
        self.instancers=[]
        self.wheels=[]
        self.clear(stage)
        cfg,tr=motion.settings,motion.tracks
        with Usd.EditContext(stage,stage.GetSessionLayer()):
            UsdGeom.Imageable(stage.GetPrimAtPath('/World/Robot/Tracks')).GetVisibilityAttr().Set('invisible')
            UsdGeom.Xform.Define(stage,self.ROOT)
            mats={name:material(stage,self.ROOT+'/Looks/'+name,color) for name,color in (
                ('Shoe',(.16,.19,.21)),('Pad',(.045,.05,.055)),('Pin',(.42,.46,.49)),
                ('Rubber',(.025,.03,.035)),('Hub',(.22,.26,.29)),('Mark',(.9,.30,.045)))}
            for side,name in enumerate(('Left','Right')):
                root=self.ROOT+'/'+name
                UsdGeom.Xform.Define(stage,root)
                proto=root+'/ShoePrototype'
                UsdGeom.Xform.Define(stage,proto)
                plate=cfg['shoe_thickness_m']
                box(stage,proto+'/Plate',(cfg['shoe_length_m'],tr['width_m'],plate),(0,0,0),mats['Shoe'])
                box(stage,proto+'/Grouser',(cfg['grouser_length_m'],tr['width_m'],cfg['grouser_height_m']),
                    (0,0,-plate/2-cfg['grouser_height_m']/2),mats['Pad'])
                box(stage,proto+'/GuideHorn',(.018,.016,cfg['guide_horn_height_m']),
                    (0,0,plate/2+cfg['guide_horn_height_m']/2),mats['Shoe'])
                # Pins are visual geometry; the band path supplies connectivity.
                for sign in (-1,1):
                    cylinder(stage,proto+('/PinL' if sign==1 else '/PinR'),.004,.012,
                        (cfg['shoe_length_m']/2,sign*(tr['width_m']/2+.003),0),mats['Pin'],'Y')
                links=UsdGeom.PointInstancer.Define(stage,root+'/Shoes')
                links.CreatePrototypesRel().SetTargets([proto])
                links.CreateProtoIndicesAttr([0]*motion.count)
                links.CreateIdsAttr(list(range(motion.count)))
                self.instancers.append(links)
                y=tr['center_separation_m']/2*(1 if side==0 else -1)
                inner_bottom=tr['center_z_m']-motion.radius+plate/2
                for i,x in enumerate(cfg['road_wheel_x_m']):
                    self._wheel(root+f'/RoadWheel_{i}',(x,y,inner_bottom+cfg['road_wheel_radius_m']),
                        cfg['road_wheel_radius_m'],side,cfg['road_wheel_radius_m'],mats)
                for name,x in (('DriveSprocket',-motion.half_straight),('FrontIdler',motion.half_straight)):
                    self._wheel(root+'/'+name,(x,y,tr['center_z_m']),motion.radius-plate/2-.001,
                        side,motion.radius,mats,sprocket=name=='DriveSprocket')
                for i,x in enumerate(cfg['return_roller_x_m']):
                    self._wheel(root+f'/ReturnRoller_{i}',(x,y,tr['center_z_m']+motion.radius-plate/2-cfg['return_roller_radius_m']),
                        cfg['return_roller_radius_m'],side,cfg['return_roller_radius_m'],mats)
        self.update(0.,force=True)

    def _wheel(self,path,position,radius,side,rolling_radius,mats,sprocket=False):
        root=UsdGeom.Xform.Define(self.stage,path)
        root.AddTranslateOp().Set(Gf.Vec3d(*position))
        orient=root.AddOrientOp()
        width=self.motion.tracks['width_m']
        for sign in (-1,1):
            suffix='L' if sign==1 else 'R'
            cylinder(self.stage,path+'/Rim'+suffix,radius,.045,(0,sign*.0475,0),mats['Hub'] if sprocket else mats['Rubber'],'Y')
            cylinder(self.stage,path+'/Hub'+suffix,radius*.48,.014,(0,sign*(width/2+.004),0),mats['Hub'],'Y')
            box(self.stage,path+'/Mark'+suffix,(radius*.75,.003,.009),(radius*.28,sign*(width/2+.013),0),mats['Mark'])
        if sprocket:
            for k in range(12):
                angle=k*2*math.pi/12
                box(self.stage,path+f'/Tooth_{k}',(.010,.11,.010),
                    (radius*math.cos(angle),0,radius*math.sin(angle)),mats['Shoe'],rotate=(0,-math.degrees(angle),0))
        self.wheels.append((orient,side,rolling_radius))

    def update(self,sim_time,force=False):
        if not force and sim_time<self.next_update_s:
            return
        self.next_update_s=sim_time+1/30
        with Usd.EditContext(self.stage,self.stage.GetSessionLayer()):
            for side,links in enumerate(self.instancers):
                positions,angles,_=self.motion.poses(side)
                links.CreatePositionsAttr().Set(Vt.Vec3fArray.FromNumpy(positions.astype(np.float32)))
                orientations=Vt.QuathArray([Gf.Quath(math.cos(a/2),Gf.Vec3h(0,-math.sin(a/2),0)) for a in angles])
                links.CreateOrientationsAttr().Set(orientations)
            for orient,side,radius in self.wheels:
                angle=self.motion.distance[side]/radius
                orient.Set(Gf.Quatf(math.cos(angle/2),Gf.Vec3f(0,math.sin(angle/2),0)))
