"""Own robot-mounted RGB-D rig and static calibration captures in Isaac 4.5.

RGB/depth are public Replicator annotators on the same render product. Camera
is used for the installed version's pinhole calibration helpers. The first
interface check freezes simulation time and records the actual capture pose.
"""
import asyncio
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import uuid

import numpy as np

_RIG=None


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


class RGBDRig:
    def __init__(self,project):
        import omni.usd
        import omni.kit.app
        from pxr import UsdGeom,Usd
        import omni.replicator.core as rep
        manager=omni.kit.app.get_app().get_extension_manager()
        manager.set_extension_enabled_immediate('isaacsim.sensors.camera',True)
        from isaacsim.sensors.camera import Camera
        import carb
        carb.settings.get_settings().set_bool('/physics/updateToUsd',True)
        self.project=Path(project).resolve()
        allowed=Path('/home/ubuntu/sdc/L17').resolve(strict=True)
        if not self.project.is_relative_to(allowed):raise RuntimeError('RGB-D output must stay in own L17')
        self.stage=omni.usd.get_context().get_stage()
        if not np.isclose(UsdGeom.GetStageMetersPerUnit(self.stage),1.):raise RuntimeError('This rig requires metre stage units')
        if str(UsdGeom.GetStageUpAxis(self.stage))!='Z':raise RuntimeError('This rig requires a Z-up stage')
        if not self.stage.GetPrimAtPath('/World/Robot'):raise RuntimeError('Own robot is missing')
        self.camera_config=json.loads((self.project/'configs/navigation_camera.json').read_text())
        self.camera_config_sha256=sha(self.project/'configs/navigation_camera.json')
        cfg=self.camera_config
        resolution=tuple(int(x) for x in cfg['resolution_wh'])
        if len(resolution)!=2 or min(resolution)<16 or max(resolution)>4096:raise ValueError('Invalid navigation camera resolution')
        hfov=float(cfg['horizontal_fov_deg'])
        if not 0<hfov<180:raise ValueError('Invalid horizontal field of view')
        aperture=float(cfg['horizontal_aperture_mm'])
        focal=aperture/(2*math.tan(math.radians(hfov)/2))
        self.rep=rep;self.cameras={};self.products={};self.annotators={}
        with Usd.EditContext(self.stage,self.stage.GetSessionLayer()):
            UsdGeom.Xform.Define(self.stage,'/World/Robot/Sensors')
            for name in ['front','left','right']:
                yaw=float(cfg['yaw_degrees'][name])
                angle=math.radians(yaw)/2
                camera=Camera('/World/Robot/Sensors/Nav_'+name,name='l17_nav_'+name,
                              translation=np.array(cfg['translation_body_m']),orientation=np.array([math.cos(angle),0.,0.,math.sin(angle)]),
                              resolution=resolution)
                usd=UsdGeom.Camera(camera.prim)
                usd.GetFocalLengthAttr().Set(focal)
                usd.GetHorizontalApertureAttr().Set(aperture)
                usd.GetVerticalApertureAttr().Set(aperture*resolution[1]/resolution[0])
                usd.GetClippingRangeAttr().Set(tuple(cfg['clip_m']))
                camera.set_projection_type('pinhole')
                self._attach(name,camera,resolution)
            near=cfg.get('mapping_nearfield')
            if near:
                pitch=math.radians(float(near['pitch_down_degrees']))/2
                resolution=tuple(near['resolution_wh'])
                camera=Camera('/World/Robot/Sensors/MappingNearfield',name='l17_mapping_nearfield',
                    translation=np.array(near['translation_body_m']),orientation=np.array([math.cos(pitch),0.,math.sin(pitch),0.]),
                    resolution=resolution)
                usd=UsdGeom.Camera(camera.prim)
                usd.GetFocalLengthAttr().Set(aperture/(2*math.tan(math.radians(near['horizontal_fov_deg'])/2)))
                usd.GetHorizontalApertureAttr().Set(aperture)
                usd.GetVerticalApertureAttr().Set(aperture*resolution[1]/resolution[0])
                usd.GetClippingRangeAttr().Set(tuple(cfg['clip_m']))
                camera.set_projection_type('pinhole')
                self._attach('mapping_nearfield',camera,resolution)
        self.busy=False
        self.initialized=False
        self.warmup=None
        self.capture_runtime=None

    def _attach(self,name,camera,resolution):
        from isaacsim.sensors.camera import Camera
        product=self.rep.create.render_product(camera.prim_path,resolution)
        # The 4.5 calibration API reads resolution through its render-product
        # path. Associate a wrapper with the already-created product without
        # duplicating camera acquisition callbacks or changing the USD pose.
        camera=Camera(camera.prim_path,name='l17_calibration_'+name,resolution=resolution,
                      render_product_path=product.path)
        annotators={kind:self.rep.AnnotatorRegistry.get_annotator(kind) for kind in ['rgb','distance_to_image_plane','ReferenceTime']}
        for item in annotators.values():item.attach([product.path])
        self.cameras[name]=camera;self.products[name]=product;self.annotators[name]=annotators

    async def _render(self):
        import omni.kit.app
        import omni.timeline
        # Camera.initialize installs the native rendering callback that evaluates
        # the SDG pipeline. Plain app updates render products without the
        # Replicator step helper's timeline start/stop transitions.
        omni.timeline.get_timeline_interface().pause()
        for _ in range(8):await omni.kit.app.get_app().next_update_async()

    async def _initialize_capture(self):
        if self.initialized:return
        import omni.kit.app
        import omni.timeline
        from isaacsim.core.nodes.bindings import _isaacsim_core_nodes
        timeline=omni.timeline.get_timeline_interface()
        clock=_isaacsim_core_nodes.acquire_interface()
        started=float(clock.get_sim_time())
        timeline.play()
        for _ in range(24):await omni.kit.app.get_app().next_update_async()
        timeline.pause()
        self.warmup={'start_sim_s':started,'end_sim_s':float(clock.get_sim_time()),
                     'purpose':'initialize physics and let the uncommanded robot settle before static sensor validation',
                     'pose_or_velocity_overwrite':False}
        for camera in self.cameras.values():camera.initialize()
        self.initialized=True
        for _ in range(8):await omni.kit.app.get_app().next_update_async()

    async def _capture(self,name,folder):
        import omni.timeline
        from pxr import UsdGeom
        from isaacsim.core.nodes.bindings import _isaacsim_core_nodes
        from PIL import Image
        camera=self.cameras[name]
        before=np.asarray(np.linalg.inv(camera.get_view_matrix_ros()),dtype=float)
        await self._render()
        after=np.asarray(np.linalg.inv(camera.get_view_matrix_ros()),dtype=float)
        if not np.allclose(before,after,atol=1e-7):raise RuntimeError('Camera moved during static RGB-D capture')
        data=self.annotators[name]
        rgb=np.asarray(data['rgb'].get_data())
        depth=np.asarray(data['distance_to_image_plane'].get_data()).squeeze().astype(np.float32)
        if rgb.ndim!=3 or rgb.shape[2] not in [3,4] or depth.shape!=rgb.shape[:2] or rgb.dtype!=np.uint8:
            raise RuntimeError('RGB and depth annotator shapes/types are inconsistent')
        reference=data['ReferenceTime'].get_data()
        frame=int(reference['referenceTimeNumerator'])
        # USD timeline time can remain zero while native physics advances.
        # Compare renderer and pose stamps in the same native simulation clock;
        # retain timeline time separately for diagnosing clock-domain mistakes.
        clock=_isaacsim_core_nodes.acquire_interface()
        render_time=float(clock.get_sim_time_at_swh_frame(frame))
        pose_time=float(clock.get_sim_time())
        folder.mkdir(parents=True,exist_ok=False)
        Image.fromarray(rgb[:,:,:3]).save(folder/'rgb.png')
        np.save(folder/'depth_m.npy',depth,allow_pickle=False)
        k=np.asarray(camera.get_intrinsics_matrix(),dtype=float)
        h,w=depth.shape
        body=np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(self.stage.GetPrimAtPath('/World/Robot'))).T
        self_pose=[float(body[0,3]),float(body[1,3]),math.atan2(float(body[1,0]),float(body[0,0]))]
        runtime=self.capture_runtime
        controller_time=float(runtime.sim_time) if runtime is not None else None
        controller_epoch=runtime.clock_epoch if runtime is not None else None
        pose_delta=None
        if runtime is not None:
            if runtime.body_view is None:raise RuntimeError('Native robot body is not initialized')
            physics_pose=runtime.body_view.get_transforms()[0]
            pose_delta=float(np.linalg.norm(np.asarray(physics_pose[:3])-body[:3,3]))
            if pose_delta>.05:raise RuntimeError('Rendered robot pose does not match native physics')
        meta={'schema':'L17.rgbd.v1','frame_id':name+':'+str(frame)+':'+uuid.uuid4().hex[:8],
              'renderer_reference_frame':frame,'camera_name':name,
              'camera_prim_path':str(camera.prim_path),'camera_role':'robot_mounted' if name!='reference_water' else 'fixed_reference_for_sensor_validation',
              'width':w,'height':h,'rgb_file':'rgb.png','depth_file':'depth_m.npy',
              'rgb_sha256':sha(folder/'rgb.png'),'depth_sha256':sha(folder/'depth_m.npy'),
              'intrinsics':k.tolist(),'world_from_optical':after.tolist(),
              'optical_axes':'x_right_y_down_z_forward','depth_type':'distance_to_image_plane','depth_unit':'m',
              'render_time_s':render_time,'pose_time_s':pose_time,'capture_mode':'paused_static_interface_validation',
              'timestamp_clock':'isaac_core_nodes_sim_time',
              'robot_pose_xyyaw':self_pose,'self_pose_source':'simulator_self_pose_baseline',
              'controller_time_s':controller_time,'controller_epoch':controller_epoch,
              'navigation_camera_config_sha256':self.camera_config_sha256,
              'render_physics_position_delta_m':pose_delta,
              'usd_timeline_time_s':float(omni.timeline.get_timeline_interface().get_current_time()),
              'pose_change_matrix_max':float(np.max(np.abs(after-before))),
              'depth_is_underwater_optics_model':False}
        (folder/'frame.json').write_text(json.dumps(meta,indent=2)+'\n')
        from depth_localization import DepthFrame,unproject
        DepthFrame(meta,depth).validate()
        uv=np.array([[w//2,h//2],[w//2-17,h//2],[w//2,h//2+13]],dtype=float)
        d=depth[uv[:,1].astype(int),uv[:,0].astype(int)]
        valid=np.isfinite(d)&(d>.08)&(d<100)
        if valid.any():
            expected=np.asarray(camera.get_world_points_from_image_coords(uv[valid],d[valid]),dtype=float)
            actual=unproject(uv[valid],d[valid],k,after)
            comparison={'reference':'installed Isaac Camera.get_world_points_from_image_coords',
                        'checked_pixels':uv[valid].tolist(),'max_world_difference_m':float(np.linalg.norm(actual-expected,axis=1).max())}
            (folder/'api_projection_check.json').write_text(json.dumps(comparison,indent=2)+'\n')
        finite=depth[np.isfinite(depth)&(depth>.08)&(depth<100)]
        return {'name':name,'frame_file':str(folder/'frame.json'),'rgb_file':str(folder/'rgb.png'),
                'valid_depth_fraction':float(len(finite)/depth.size),'depth_min_m':float(finite.min()) if len(finite) else None,
                'depth_max_m':float(finite.max()) if len(finite) else None}

    async def calibration(self,root):
        from pxr import Usd,UsdGeom,Gf
        from depth_localization import DepthFrame,localize_surface
        camera=self.cameras['front']
        position,quaternion=camera.get_world_pose(camera_axes='world')
        # This known cube is a separate calibration fixture, never a hidden
        # navigation target. The estimator receives only its RGB/depth pixel.
        from isaacsim.core.utils.rotations import quat_to_rot_matrix
        rotation=np.asarray(quat_to_rot_matrix(quaternion),dtype=float)
        center=np.asarray(position,dtype=float)+3.*rotation[:,0]
        path='/World/L17_RGBD_CalibrationFixture'
        with Usd.EditContext(self.stage,self.stage.GetSessionLayer()):
            cube=UsdGeom.Cube.Define(self.stage,path)
            cube.CreateSizeAttr(.6)
            cube.AddTranslateOp().Set(Gf.Vec3d(*center))
            cube.AddOrientOp().Set(Gf.Quatf(float(quaternion[0]),Gf.Vec3f(*quaternion[1:])))
            cube.CreateDisplayColorAttr([Gf.Vec3f(.9,.05,.6)])
        try:
            await self._capture('front',root/'calibration')
            frame=DepthFrame.load(root/'calibration/frame.json')
            h,w=frame.depth_m.shape
            estimate=localize_surface(frame,{'visible':True,'format_valid':True,'image_sha256':frame.metadata['rgb_sha256'],
                                            'point_uv':[w//2,h//2],'provider':'known_calibration_pixel_only'})
            expected=np.asarray(position,dtype=float)+2.7*rotation[:,0]
            error=float(np.linalg.norm(np.asarray(estimate['world_point_m'])-expected)) if estimate['accepted'] else None
            result={'fixture':'0.6 m cube centred 3.0 m ahead of the optical centre',
                    'expected_axial_depth_m':2.7,'estimate':estimate,'world_surface_error_m':error,
                    'passed':bool(estimate['accepted'] and abs(estimate['depth_m']-2.7)<.005 and error<.015),
                    'target_world_truth_used_by_estimator':False}
            (root/'calibration_report.json').write_text(json.dumps(result,indent=2)+'\n')
            return result
        finally:
            with Usd.EditContext(self.stage,self.stage.GetSessionLayer()):self.stage.RemovePrim(path)

    async def capture_all(self,run_calibration=False,include_reference=False,runtime=None):
        import omni.timeline
        if self.busy:raise RuntimeError('RGB-D capture is already in progress')
        self.busy=True
        self.capture_runtime=runtime
        timeline=omni.timeline.get_timeline_interface()
        was_playing=timeline.is_playing();timeline.pause()
        root=self.project/'results/rgbd'/('run_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8])
        root.mkdir(parents=True,exist_ok=False)
        result={'directory':str(root),'capture_mode':'paused_static_interface_validation','robot_motion_commanded':False,
                'frame_files':[],'mapping_frame_files':[],'diagnostic_frame_files':[],'calibration':None}
        try:
            if include_reference and 'reference_water' not in self.cameras and self.stage.GetPrimAtPath('/World/Cameras/WaterTarget'):
                from isaacsim.sensors.camera import Camera
                from pxr import Usd
                with Usd.EditContext(self.stage,self.stage.GetSessionLayer()):
                    camera=Camera('/World/Cameras/WaterTarget',name='l17_reference_water',resolution=(1280,720))
                    camera.set_projection_type('pinhole');self._attach('reference_water',camera,(1280,720))
                if self.initialized:self.cameras['reference_water'].initialize()
            await self._initialize_capture()
            result['initialization_warmup']=self.warmup
            if run_calibration:result['calibration']=await self.calibration(root)
            for name in self.cameras:
                if name=='reference_water':
                    if include_reference:result['diagnostic_frame_files'].append(await self._capture(name,root/name))
                elif name=='mapping_nearfield':result['mapping_frame_files'].append(await self._capture(name,root/name))
                else:result['frame_files'].append(await self._capture(name,root/name))
            (root/'capture_summary.json').write_text(json.dumps(result,indent=2)+'\n')
            (self.project/'results/rgbd/latest_capture.json').write_text(json.dumps(result,indent=2)+'\n')
            return result
        finally:
            self.busy=False
            self.capture_runtime=None
            if was_playing:timeline.play()


def get_rig(project):
    global _RIG
    import omni.usd
    if _RIG is None or _RIG.stage!=omni.usd.get_context().get_stage():_RIG=RGBDRig(project)
    return _RIG


def dispose_rig():
    global _RIG
    if _RIG is None:return
    if _RIG.busy:raise RuntimeError('Do not reload while a sensor capture is running')
    for camera in _RIG.cameras.values():
        try:camera.pause()
        except Exception:pass
    for name,product in _RIG.products.items():
        for annotator in _RIG.annotators[name].values():
            try:annotator.detach([product.path])
            except Exception:pass  # Legacy step cleanup can already have removed SDG nodes.
        try:product.destroy()
        except Exception:pass
    _RIG=None
