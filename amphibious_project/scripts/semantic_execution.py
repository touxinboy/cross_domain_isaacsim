"""Own semantic baseline execution from proprioception, without a terrain map oracle."""
import math
import numpy as np

from crossing_control import load_ratio,wrap_angle
from navigation_execution import PhysicalNavigationBridge


class SensedMediumSwitcher:
    policy_name='contact_buoyancy_reference_without_scene_depth'
    def __init__(self,robot):
        self.robot=robot;self.mode='LAND';self.filtered_load=1.;self.candidate='LAND';self.candidate_since=0.
    def update(self,row,dt):
        self.filtered_load+=(1-math.exp(-dt/.25))*(load_ratio(row,self.robot)-self.filtered_load)
        buoy=row['buoyancy_n']/(self.robot['mass_kg']*self.robot['gravity_m_s2'])
        if self.filtered_load>.38 and buoy<.06:candidate='LAND'
        elif self.filtered_load<.08 and buoy>.72:candidate='WATER'
        else:candidate='HYBRID'
        now=row['time_s']
        if candidate!=self.candidate:self.candidate=candidate;self.candidate_since=now
        if now-self.candidate_since>=(.10 if candidate=='HYBRID' else .4):self.mode=candidate
        return self.mode


class SemanticPhysicalBridge(PhysicalNavigationBridge):
    def __init__(self,robot,control,max_plan_age_s=2.5,max_step_m=1.5):
        super().__init__(robot,None,control,max_plan_age_s,max_step_m)
        self.switcher=SensedMediumSwitcher(robot);self.scan_heading=None
    def clear_plan(self):
        self.scan_heading=None;super().clear_plan()
    def scan_to(self,yaw):
        self.clear_plan();self.scan_heading=float(yaw)
    def tick(self,row,dt):
        if self.scan_heading is None:return super().tick(row,dt)
        mode=self.switcher.update(row,dt);error=wrap_angle(self.scan_heading-row['yaw_rad'])
        tracking=abs(error)>.06 or abs(row['wz_rad_s'])>.06
        if tracking:
            w=float(np.clip(1.4*error-.4*row['wz_rad_s'],-.4,.4))
            command=self.allocator.commands([0.,w/2],mode,row,dt)
        else:command={'mode':mode,'tracks':[0.,0.],'thrusters':[0.,0.]}
        return {'status':'tracking' if tracking else 'short_waypoint_reached','plan_age_s':None,'target_xy_m':None,
                'switch_policy':self.switcher.policy_name,'command':command,'active_perception_rotation':True}
