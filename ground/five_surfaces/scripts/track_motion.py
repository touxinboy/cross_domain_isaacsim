"""Shared moving-shoe geometry for rendering and SCM contact (no Kit imports).

The chain follows a fixed stadium path; shoes are not independent dynamic
bodies. Positive belt speed moves the bottom run backward relative to chassis.
"""
import math
import numpy as np


class CirculatingTracks:
    def __init__(self,robot,settings):
        self.settings,self.tracks = settings,robot['tracks']
        self.count = int(settings['shoes_per_side'])
        self.half_straight = (self.tracks['length_m']-self.tracks['height_m'])/2
        self.radius = self.tracks['height_m']/2-settings['shoe_thickness_m']/2-settings['grouser_height_m']
        if self.radius <= 0 or self.count < 12:
            raise ValueError('Invalid shoe path dimensions')
        self.straight = 2*self.half_straight
        self.perimeter = 2*self.straight+2*math.pi*self.radius
        self.pitch = self.perimeter/self.count
        if settings['shoe_length_m'] >= self.pitch:
            raise ValueError('Shoe length must be smaller than chain pitch')
        self.reset()

    def reset(self):
        self.distance = np.zeros(2)
        self._speeds = np.zeros(2)

    def advance(self,speeds,dt):
        if dt < 0 or not np.isfinite(dt):
            raise ValueError('Track timestep must be finite and nonnegative')
        self.distance += np.asarray(speeds)*dt

    @property
    def phase(self):
        return np.remainder(-self.distance,self.perimeter)

    def path(self,s):
        s = np.remainder(np.asarray(s,dtype=float),self.perimeter)
        x,z,t = np.zeros_like(s),np.zeros_like(s),np.zeros_like(s)
        a = s < self.straight
        x[a],z[a],t[a] = -self.half_straight+s[a],-self.radius,0.
        b = (s >= self.straight)&(s < self.straight+math.pi*self.radius)
        angle = -math.pi/2+(s[b]-self.straight)/self.radius
        x[b],z[b],t[b] = self.half_straight+self.radius*np.cos(angle),self.radius*np.sin(angle),angle+math.pi/2
        c = (s >= self.straight+math.pi*self.radius)&(s < 2*self.straight+math.pi*self.radius)
        x[c],z[c],t[c] = self.half_straight-(s[c]-self.straight-math.pi*self.radius),self.radius,math.pi
        d = ~(a|b|c)
        angle = math.pi/2+(s[d]-2*self.straight-math.pi*self.radius)/self.radius
        x[d],z[d],t[d] = -self.half_straight+self.radius*np.cos(angle),self.radius*np.sin(angle),angle+math.pi/2
        curvature = np.where(b|d,1/self.radius,0.)
        return x,z+self.tracks['center_z_m'],t,curvature

    def poses(self,side):
        s = (np.arange(self.count)+.5)*self.pitch+self.phase[side]
        x,z,t,k = self.path(s)
        y = self.tracks['center_separation_m']/2*(1 if side==0 else -1)
        return np.column_stack((x,np.full(self.count,y),z)),t,k

    def nearest_parameter(self,body_points):
        x,z = body_points[:,0],body_points[:,2]-self.tracks['center_z_m']
        s = np.where(z <= 0,x+self.half_straight,
                     self.straight+math.pi*self.radius+self.half_straight-x)
        right = x > self.half_straight
        s[right] = self.straight+self.radius*(np.arctan2(z[right],x[right]-self.half_straight)+math.pi/2)
        left = x < -self.half_straight
        angle = np.arctan2(z[left],x[left]+self.half_straight)
        angle = np.where(angle < math.pi/2,angle+2*math.pi,angle)
        s[left] = 2*self.straight+math.pi*self.radius+self.radius*(angle-math.pi/2)
        return np.remainder(s,self.perimeter)

    def ray_hits(self,side,origins_body,direction_body,guess_body,all_shoes=False):
        """Vertical world rays in chassis coordinates; return height and motion.

        A nearby-shoe search is used in normal operation; the exhaustive option
        provides a geometry reference for tests. Each shoe has a plate and lug.
        """
        origin = np.asarray(origins_body)
        direction = np.asarray(direction_body)
        n = len(origin)
        if all_shoes:
            indices = np.broadcast_to(np.arange(self.count),(n,self.count))
        else:
            s = self.nearest_parameter(guess_body)
            nearest = np.rint((s-self.phase[side])/self.pitch-.5).astype(int)
            r = self.settings['contact_neighbor_shoes']
            indices = np.remainder(nearest[:,None]+np.arange(-r,r+1),self.count)
        centers,angles,curvature = self.poses(side)
        c = centers[indices]
        theta = angles[indices]
        co,si = np.cos(theta),np.sin(theta)
        delta = origin[:,None,:]-c
        local_origin = np.stack((delta[:,:,0]*co+delta[:,:,2]*si,delta[:,:,1],
                                 -delta[:,:,0]*si+delta[:,:,2]*co),axis=-1)
        local_direction = np.stack((direction[0]*co+direction[2]*si,
                                    np.full_like(co,direction[1]),
                                    -direction[0]*si+direction[2]*co),axis=-1)
        plate = self.settings['shoe_thickness_m']
        grouser = self.settings['grouser_height_m']
        sizes = ((self.settings['shoe_length_m'],self.tracks['width_m'],plate,0.),
                 (self.settings['grouser_length_m'],self.tracks['width_m'],grouser,-plate/2-grouser/2))
        best = np.full(n,np.inf)
        best_shoe = np.zeros(n,dtype=int)
        best_lug = np.zeros(n,dtype=bool)
        for lug,(length,width,height,offset) in enumerate(sizes):
            q = local_origin.copy()
            q[:,:,2] -= offset
            half = np.array([length,width,height])/2
            parallel = np.abs(local_direction)<1e-10
            low,high = np.zeros_like(q),np.zeros_like(q)
            np.divide(-half-q,local_direction,out=low,where=~parallel)
            np.divide(half-q,local_direction,out=high,where=~parallel)
            enter = np.max(np.where(parallel,-np.inf,np.minimum(low,high)),axis=-1)
            leave = np.min(np.where(parallel,np.inf,np.maximum(low,high)),axis=-1)
            valid = (enter <= leave+1e-9)&np.all(~parallel|(np.abs(q)<=half+1e-9),axis=-1)
            enter = np.where(valid,enter,np.inf)
            local_index = np.argmin(enter,axis=1)
            hit = enter[np.arange(n),local_index]
            choose = hit < best
            best[choose] = hit[choose]
            best_shoe[choose] = indices[np.arange(n),local_index][choose]
            best_lug[choose] = bool(lug)
        valid = np.isfinite(best)
        hit_body = origin+np.where(valid,best,0.)[:,None]*direction
        contact_motion = np.zeros_like(hit_body)
        # Different points on a shoe have the same rigid shoe velocity. Curved
        # runs include the relative angular velocity of the rotating plate.
        speed = self._speeds[side] if hasattr(self,'_speeds') else 0.
        t = angles[best_shoe]
        center_v = np.column_stack((-speed*np.cos(t),np.zeros(n),-speed*np.sin(t)))
        omega = np.column_stack((np.zeros(n),speed*curvature[best_shoe],np.zeros(n)))
        contact_motion[valid] = (center_v+np.cross(omega,hit_body-centers[best_shoe]))[valid]
        return best,contact_motion,best_lug,best_shoe

    def set_speeds(self,speeds):
        self._speeds = np.asarray(speeds,dtype=float).copy()

    def telemetry(self):
        result={}
        for i,name in enumerate(('left','right')):
            result.update({name+'_belt_distance_m':float(self.distance[i]),
                name+'_belt_phase_m':float(self.phase[i]),
                name+'_sprocket_angle_rad':float(self.distance[i]/self.radius)})
        return result
