"""RGB-D geometry independent of Isaac. No target Stage/mesh positions are read.

Output is an observed surface point, not a collision-free robot arrival pose.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np


def calibration(intrinsics, world_from_optical):
    k=np.asarray(intrinsics,dtype=float)
    t=np.asarray(world_from_optical,dtype=float)
    if k.shape!=(3,3) or not np.isfinite(k).all() or min(k[0,0],k[1,1])<=0 or not np.allclose(k[2],[0,0,1]):
        raise ValueError('Invalid pinhole intrinsics')
    if t.shape!=(4,4) or not np.isfinite(t).all() or not np.allclose(t[3],[0,0,0,1]):
        raise ValueError('Invalid world_from_optical transform')
    r=t[:3,:3]
    if not np.allclose(r.T@r,np.eye(3),atol=1e-5) or not np.isclose(np.linalg.det(r),1,atol=1e-5):
        raise ValueError('Camera rotation must be proper and orthonormal')
    return k,t


def unproject(uv,depth_m,intrinsics,world_from_optical,depth_type='distance_to_image_plane'):
    k,t=calibration(intrinsics,world_from_optical)
    points=np.asarray(uv,dtype=float)
    depth=np.asarray(depth_m,dtype=float)
    if points.ndim!=2 or points.shape[1]!=2 or depth.shape!=(len(points),) or not np.isfinite(points).all() or not np.isfinite(depth).all() or np.any(depth<=0):
        raise ValueError('Expected finite UV points and positive metric depths')
    rays=(np.linalg.inv(k)@np.c_[points,np.ones(len(points))].T).T
    if depth_type=='distance_to_image_plane':
        camera=rays/rays[:,2,None]*depth[:,None]
    elif depth_type=='distance_to_camera':
        camera=rays/np.linalg.norm(rays,axis=1)[:,None]*depth[:,None]
    else: raise ValueError('Declare axial image-plane depth or Euclidean camera range')
    return (t[:3,:3]@camera.T).T+t[:3,3]


def project(world_points,intrinsics,world_from_optical):
    k,t=calibration(intrinsics,world_from_optical)
    p=np.asarray(world_points,dtype=float)
    if p.ndim!=2 or p.shape[1]!=3 or not np.isfinite(p).all(): raise ValueError('Expected finite world XYZ')
    camera=(t[:3,:3].T@(p-t[:3,3]).T).T
    if np.any(camera[:,2]<=0): raise ValueError('World point is behind the camera')
    image=(k@camera.T).T
    return image[:,:2]/image[:,2,None],camera[:,2]


@dataclass
class DepthFrame:
    metadata: dict
    depth_m: np.ndarray

    def validate(self,max_pose_skew_s=.035):
        meta=self.metadata
        if meta.get('schema')!='L17.rgbd.v1' or meta.get('depth_unit')!='m':
            raise ValueError('Use L17.rgbd.v1 with metric depth')
        if meta.get('optical_axes')!='x_right_y_down_z_forward': raise ValueError('Unexpected optical axes')
        if self.depth_m.ndim!=2 or list(self.depth_m.shape)!=[meta['height'],meta['width']]:
            raise ValueError('Depth must have shape H,W, matching RGB')
        calibration(meta['intrinsics'],meta['world_from_optical'])
        times=np.array([meta['render_time_s'],meta['pose_time_s']],dtype=float)
        if not np.isfinite(times).all() or abs(times[0]-times[1])>max_pose_skew_s:
            raise ValueError('Camera pose and rendered observation are not synchronized')
        if meta.get('depth_type') not in ['distance_to_image_plane','distance_to_camera']:
            raise ValueError('Unknown depth convention')
        return self

    @classmethod
    def load(cls,path):
        path=Path(path).resolve()
        meta=json.loads(path.read_text(encoding='utf-8'))
        root=path.parent
        def file(name):
            value=(root/meta[name]).resolve()
            if not value.is_relative_to(root): raise ValueError('Frame files must remain inside the capture directory')
            return value
        rgb=file('rgb_file');depth=file('depth_file')
        if hashlib.sha256(rgb.read_bytes()).hexdigest()!=meta['rgb_sha256']:
            raise ValueError('RGB file identity mismatch')
        if hashlib.sha256(depth.read_bytes()).hexdigest()!=meta['depth_sha256']:
            raise ValueError('Depth file identity mismatch')
        return cls(meta,np.load(depth,allow_pickle=False)).validate()


def localize_surface(frame,grounding,radius=3,min_samples=6,depth_bounds=(.08,100.),mask=None):
    frame.validate()
    result={'accepted':False,'source':'observed_rgbd_surface','frame_id':frame.metadata['frame_id'],
            'rgb_sha256':frame.metadata['rgb_sha256'],'render_time_s':frame.metadata['render_time_s'],
            'camera_role':frame.metadata.get('camera_role','unclassified'),
            'world_point_m':None,'goal_xy_m':None,'navigation_goal_valid':False,
            'semantic_association':'depends_on_the_2d_grounding_provider',
            'note':'A visible surface estimate still requires free-space/standoff planning to form an arrival pose.'}
    if grounding.get('image_sha256')!=frame.metadata['rgb_sha256']:
        raise ValueError('Grounding belongs to another RGB frame; do not reuse old pixels on new depth')
    if not grounding.get('visible') or grounding.get('format_valid') is False:
        result['reason']='target_not_grounded';return result
    uv=np.asarray(grounding.get('point_uv'),dtype=float)
    h,w=frame.depth_m.shape
    if uv.shape!=(2,) or not np.isfinite(uv).all() or not 0<=uv[0]<w or not 0<=uv[1]<h:
        raise ValueError('Grounded pixel is outside the image')
    u,v=np.rint(uv).astype(int)
    u=min(w-1,u);v=min(h-1,v)
    z0=float(frame.depth_m[v,u])
    if not np.isfinite(z0) or not depth_bounds[0]<=z0<=depth_bounds[1]:
        result['reason']='invalid_depth_at_grounded_pixel';return result
    y0,y1=max(0,v-radius),min(h,v+radius+1)
    x0,x1=max(0,u-radius),min(w,u+radius+1)
    patch=frame.depth_m[y0:y1,x0:x1]
    valid=np.isfinite(patch)&(patch>=depth_bounds[0])&(patch<=depth_bounds[1])
    # A local cluster around the actual grounded pixel avoids taking the median
    # of all background seen through a hollow framework's entire bounding box.
    selected=valid&(np.abs(patch-z0)<=max(.08,.02*z0))
    if mask is not None:
        mask=np.asarray(mask,dtype=bool)
        if mask.shape!=(h,w) or not mask[v,u]: raise ValueError('Mask must contain the grounded pixel and match the frame')
        selected&=mask[y0:y1,x0:x1]
        result['semantic_association']='filtered_by_supplied_target_mask'
    n=int(selected.sum())
    result.update(sampled_pixel_uv=[int(u),int(v)],seed_depth_m=z0,local_sample_count=n,
                  local_cluster_fraction=float(n/patch.size))
    if n<min_samples:
        result['reason']='insufficient_same_surface_depth_support';return result
    values=patch[selected].astype(float)
    spread=float(np.percentile(values,90)-np.percentile(values,10))
    if spread>max(.12,.02*z0):
        result['reason']='local_depth_discontinuity';return result
    depth=float(np.median(values))
    xyz=unproject([[u,v]],[depth],frame.metadata['intrinsics'],frame.metadata['world_from_optical'],frame.metadata['depth_type'])[0]
    ys,xs=np.where(selected)
    cloud=unproject(np.c_[xs+x0,ys+y0],values,frame.metadata['intrinsics'],frame.metadata['world_from_optical'],frame.metadata['depth_type'])
    result.update(accepted=True,reason='metric_surface_observation',world_point_m=xyz.tolist(),
                  depth_m=depth,depth_type=frame.metadata['depth_type'],depth_p90_minus_p10_m=spread,
                  local_surface_covariance_m2=np.cov(cloud,rowvar=False).tolist(),
                  covariance_scope='local_surface_dispersion_not_calibrated_sensor_uncertainty')
    return result
