"""Incremental geometry from calibrated robot RGB-D, without scene geometry.

Ground normals and continuous observed image quads provide visible support.
Unknown space stays blocked. Cyan-water classification is an explicit rendered
RGB heuristic, not a learned segmentation model or a bathymetry measurement.
"""
import heapq
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from depth_localization import DepthFrame, unproject
from navigation_io import file_sha, write_json


class OnlineTerrainMap:
    def __init__(self, center_xy, resolution=.25, extent_m=100., footprint_radius=.68):
        self.resolution=float(resolution)
        self.origin=np.floor(np.asarray(center_xy,dtype=float)/self.resolution)*self.resolution-extent_m/2
        n=int(math.ceil(extent_m/self.resolution));self.shape=(n,n)
        self.footprint_radius=float(footprint_radius)
        self.ground=np.zeros(self.shape,bool);self.water=np.zeros(self.shape,bool)
        self.floor=np.full(self.shape,np.inf,np.float32)
        self.low_steep=np.full(self.shape,np.inf,np.float32)
        self.high=np.full(self.shape,-np.inf,np.float32)
        self.steep_count=np.zeros(self.shape,np.uint32)
        self.measured_support=np.zeros(self.shape,np.int8)
        self.measured_height=np.zeros(self.shape,np.float32)
        self.seen_frames=set();self.state=np.full(self.shape,-1,np.int8)
        self.safe=np.zeros(self.shape,bool);self.last_pose=None;self.integration=[]

    def cell(self,xy):
        x,y=np.floor((np.asarray(xy)-self.origin)/self.resolution).astype(int)
        return int(y),int(x)

    def xy(self,cell):return (self.origin+(np.asarray(cell[::-1])+.5)*self.resolution).tolist()

    def _indices(self,points):
        ij=np.floor((points[:,:2]-self.origin)/self.resolution).astype(int)
        inside=(ij[:,0]>=0)&(ij[:,1]>=0)&(ij[:,0]<self.shape[1])&(ij[:,1]<self.shape[0])
        return ij[inside,1],ij[inside,0],inside

    def integrate(self,frame_path,stride=6):
        frame=DepthFrame.load(frame_path);meta=frame.metadata
        if meta.get('camera_role')!='robot_mounted':raise ValueError('Only robot-mounted RGB-D may build the navigation map')
        if meta['frame_id'] in self.seen_frames:return
        with Image.open(Path(frame_path).parent/meta['rgb_file']) as source:rgb=np.asarray(source.convert('RGB'))
        depth=frame.depth_m[::stride,::stride]
        vv,uu=np.mgrid[0:meta['height']:stride,0:meta['width']:stride]
        valid=np.isfinite(depth)&(depth>.08)&(depth<40.)
        xyz=np.full((*depth.shape,3),np.nan)
        xyz[valid]=unproject(np.c_[uu[valid],vv[valid]],depth[valid],meta['intrinsics'],meta['world_from_optical'],meta['depth_type'])
        pose=np.asarray(meta['robot_pose_xyyaw'],float);self.last_pose=pose.tolist()
        delta=xyz[:,:,:2]-pose[:2]
        forward=delta[:,:,0]*math.cos(pose[2])+delta[:,:,1]*math.sin(pose[2])
        lateral=-delta[:,:,0]*math.sin(pose[2])+delta[:,:,1]*math.cos(pose[2])
        # Known robot envelope is removed from its own observations.
        valid&=~((np.abs(forward)<.72)&(np.abs(lateral)<.57))
        normal_z=np.zeros(depth.shape)
        a=xyz[1:-1,2:]-xyz[1:-1,:-2];b=xyz[2:,1:-1]-xyz[:-2,1:-1]
        cross=np.cross(a,b);norm=np.linalg.norm(cross,axis=2)
        with np.errstate(invalid='ignore',divide='ignore'):
            normal_z[1:-1,1:-1]=np.abs(cross[:,:,2])/norm
        ground=valid&np.isfinite(normal_z)&(normal_z>=math.cos(math.radians(35)))
        color=rgb[::stride,::stride].astype(float)
        water=(color[:,:,2]>color[:,:,0]*1.15)&(color[:,:,1]>color[:,:,0]*1.12)&(color[:,:,2]>20)
        points=xyz[valid];ys,xs,inside=self._indices(points)
        np.maximum.at(self.high,(ys,xs),points[inside,2])
        points=xyz[ground];ys,xs,inside=self._indices(points)
        np.minimum.at(self.floor,(ys,xs),points[inside,2])
        steep=valid&~ground
        points=xyz[steep];ys,xs,inside=self._indices(points)
        np.minimum.at(self.low_steep,(ys,xs),points[inside,2]);np.add.at(self.steep_count,(ys,xs),1)
        ground_image=Image.new('1',(self.shape[1],self.shape[0]));water_image=Image.new('1',ground_image.size)
        draw=ImageDraw.Draw(ground_image);wet_draw=ImageDraw.Draw(water_image)
        quads=ground[:-1,:-1]&ground[:-1,1:]&ground[1:,:-1]&ground[1:,1:]
        # Fill only continuous depth-supported quads, never a whole visible box.
        for y,x in np.argwhere(quads):
            q=np.array([xyz[y,x],xyz[y,x+1],xyz[y+1,x+1],xyz[y+1,x]])
            if np.max(np.linalg.norm(q-np.roll(q,1,axis=0),axis=1))>.9:continue
            poly=np.floor((q[:,:2]-self.origin)/self.resolution).astype(int)
            if np.any(poly<0) or np.any(poly>=np.array(self.shape[::-1])):continue
            corners=[tuple(int(v) for v in point) for point in poly]
            draw.polygon(corners,fill=1)
            if np.count_nonzero(water[y:y+2,x:x+2])>=3:wet_draw.polygon(corners,fill=1)
        self.ground|=np.asarray(ground_image,dtype=bool)
        self.water|=np.asarray(water_image,dtype=bool)
        self.seen_frames.add(meta['frame_id'])
        self.integration.append({'frame_id':meta['frame_id'],'rgb_sha256':meta['rgb_sha256'],
            'depth_sha256':meta['depth_sha256'],'valid_points':int(valid.sum()),'ground_points':int(ground.sum())})

    def update_state(self,row=None):
        finite=np.isfinite(self.floor)
        if finite.any():
            distance,indices=ndimage.distance_transform_edt(~finite,return_indices=True)
            reference=self.floor[indices[0],indices[1]]
            nearby=distance*self.resolution<1.5
        else:reference=np.zeros(self.shape);nearby=np.zeros(self.shape,bool)
        occupied=(self.steep_count>=2)&nearby&(self.low_steep<reference+1.05)&(self.high>reference+.12)
        self.state.fill(-1);self.state[self.ground]=1;self.state[self.ground&self.water]=2
        self.state[occupied]=0
        self.height=np.where(finite,self.floor,reference)
        if row is not None:
            weight=48.*9.81
            supported=float(row['left_normal_n'])+float(row['right_normal_n'])>.2*weight
            floating=float(row['buoyancy_n'])>.7*weight
            if supported or floating:
                cy,cx=self.cell([row['x_m'],row['y_m']]);yaw=float(row['yaw_rad'])
                for dy in range(-4,5):
                    for dx in range(-4,5):
                        y,x=cy+dy,cx+dx
                        if not 0<=y<self.shape[0] or not 0<=x<self.shape[1]:continue
                        world=np.asarray(self.xy((y,x)))-[row['x_m'],row['y_m']]
                        f=world[0]*math.cos(yaw)+world[1]*math.sin(yaw)
                        l=-world[0]*math.sin(yaw)+world[1]*math.cos(yaw)
                        if abs(f)<.56 and abs(l)<.44:
                            self.measured_support[y,x]=1 if supported else 2
                            self.measured_height[y,x]=float(row.get('z_m',0))-.22 if supported else 0.
        supported=self.measured_support>0
        self.state[supported]=self.measured_support[supported]
        self.height[supported]=self.measured_height[supported]
        free=np.isin(self.state,[1,2])
        self.safe=free&(ndimage.distance_transform_edt(free)*self.resolution>
                        self.footprint_radius+self.resolution*math.sqrt(2)/2)
        return self.state

    def is_safe(self,cell):
        y,x=cell
        return 0<=y<self.shape[0] and 0<=x<self.shape[1] and bool(self.safe[y,x])

    def segment_safe(self,start,end,allow_current_start=False):
        a=np.asarray(start,float);b=np.asarray(end,float)
        count=max(2,int(math.ceil(np.linalg.norm(b-a)/(self.resolution/4)))+1)
        start_cell=self.cell(a)
        return all(self.is_safe(self.cell(point)) or (allow_current_start and self.cell(point)==start_cell and self.state[start_cell]>0)
                   for point in np.linspace(a,b,count))

    def reachable(self,robot_xy):
        start=self.cell(robot_xy)
        if not 0<=start[0]<self.shape[0] or not 0<=start[1]<self.shape[1] or self.state[start]<=0:
            raise ValueError('Robot footprint has no observed or measured support')
        costs={start:0.};parents={};queue=[(0.,start)]
        while queue:
            cost,cell=heapq.heappop(queue)
            if cost>costs[cell]+1e-9:continue
            for dy,dx in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                other=cell[0]+dy,cell[1]+dx
                if not self.is_safe(other):continue
                if dy and dx and not (self.is_safe((cell[0]+dy,cell[1])) and self.is_safe((cell[0],cell[1]+dx))):continue
                length=math.hypot(dx,dy)*self.resolution
                if self.state[cell]==1 and self.state[other]==1:
                    if abs(float(self.height[cell]-self.height[other]))>length*math.tan(math.radians(35))+.10:continue
                elif self.state[cell]!=self.state[other]:
                    if abs(float(self.height[cell]-self.height[other]))>.35:continue
                new=cost+length*(1.15 if self.state[other]==2 else 1.)
                if new<costs.get(other,float('inf')):
                    costs[other]=new;parents[other]=cell;heapq.heappush(queue,(new,other))
        return start,costs,parents

    def plan(self,robot_xy,target_xy=None,visited=None,min_standoff=1.8,max_standoff=3.0):
        start,costs,parents=self.reachable(robot_xy)
        choices=[]
        if target_xy is not None:
            for cell,cost in costs.items():
                distance=math.dist(self.xy(cell),target_xy)
                if self.state[cell]==1 and min_standoff<=distance<=max_standoff:
                    choices.append((cost,cell))
        if choices:
            _,goal=min(choices);kind='observed_semantic_approach'
        else:
            unknown=self.state<0
            near_unknown=ndimage.maximum_filter(unknown,size=11)
            candidates=[]
            for cell,cost in costs.items():
                if cost<.8 or not near_unknown[cell]:continue
                xy=self.xy(cell)
                remaining=math.dist(xy,target_xy) if target_xy is not None else -min(cost,12.)
                repeated=sum(max(0.,1.-math.dist(xy,p)) for p in (visited or [])[-20:])
                candidates.append((remaining+.18*cost+2.*repeated,cell))
            if not candidates:raise ValueError('No reachable observed frontier; obtain another physical viewpoint')
            _,goal=min(candidates);kind='observed_frontier_exploration'
        cells=[goal]
        while cells[-1]!=start:cells.append(parents[cells[-1]])
        path=[list(robot_xy)]+[self.xy(c) for c in reversed(cells[:-1])]
        # Use a visible short path prefix, not the target behind unknown space.
        local=path[0]
        for point in path[1:]:
            if math.dist(robot_xy,point)>2.5:break
            if self.segment_safe(robot_xy,point,allow_current_start=True):local=point
            else:break
        if math.dist(local,robot_xy)<.25:raise ValueError('No footprint-safe local route prefix')
        return {'source':'RGBD_observed_grid_Dijkstra','kind':kind,'path_xy_m':path,
                'local_goal_xy_m':local,'arrival_xy_m':self.xy(goal),'reachable_cells':len(costs),
                'target_xy_from_observation':list(target_xy) if target_xy is not None else None}

    def save(self,directory,plan=None,target_xy=None,robot_xy=None):
        root=Path(directory);root.mkdir(parents=True,exist_ok=True)
        np.save(root/'state.npy',self.state,allow_pickle=False);np.save(root/'height.npy',self.height,allow_pickle=False)
        meta={'schema':'L17.traversability.v1','source':'perception','state_file':'state.npy','state_sha256':file_sha(root/'state.npy'),
              'origin_xy_m':self.origin.tolist(),'resolution_m':self.resolution,'integrated_frames':len(self.seen_frames),
              'water_classification':'observed_RGB_cyan_heuristic_not_learned_segmentation','unknown_is_blocked':True,
              'footprint_radius_m':self.footprint_radius,'robot_self_pose_source':'simulator_self_pose_baseline',
              'hidden_scene_geometry_used':False}
        meta['self_footprint_evidence']='measured_contact_or_float_support_at_actual_visited_poses_static_scene'
        write_json(root/'grid.json',meta)
        palette={-1:(90,90,90),0:(15,15,15),1:(160,190,125),2:(45,160,190)}
        pixels=np.zeros((*self.shape,3),np.uint8)
        for code,color in palette.items():pixels[self.state==code]=color
        image=Image.fromarray(pixels);draw=ImageDraw.Draw(image)
        if plan:
            points=[self.cell(p)[::-1] for p in plan['path_xy_m']]
            if len(points)>1:draw.line(points,fill=(255,255,255),width=2)
        for xy,color in [(target_xy,(255,130,20)),(robot_xy,(230,30,210))]:
            if xy is not None:
                y,x=self.cell(xy);draw.ellipse((x-3,y-3,x+3,y+3),fill=color)
        image.transpose(Image.Transpose.FLIP_TOP_BOTTOM).save(root/'map.png')
        return meta
