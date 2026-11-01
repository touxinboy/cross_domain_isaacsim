"""Vectorized, fixed-world-grid SCM-style contact for equivalent rigid tracks.

Bekker virgin loading + elastic unloading, and a bounded vector Janosi bristle
law. This is an engineering approximation, not a port of Chrono's SCM. Soil
motion is along world Z; no bulk flow, pore pressure, or slip-induced digging.
"""
import json
import math
from pathlib import Path
import time

import numpy as np

from amphibian_dynamics import AmphibianForces, quaternion_matrix_xyzw


class SCMTerrain:
    def __init__(self, ground, settings, robot, collider_points):
        self.ground, self.settings, self.robot = ground, settings, robot
        self.spacing = float(settings['grid_spacing_m'])
        if self.spacing <= 0:
            raise ValueError('SCM grid spacing must be positive')
        tiles = ground['surfaces']
        self.origin = np.array([min(t['x_min_m'] for t in tiles), min(t['y_min_m'] for t in tiles)])
        upper = np.array([max(t['x_max_m'] for t in tiles), max(t['y_max_m'] for t in tiles)])
        self.shape = tuple(np.ceil((upper-self.origin)/self.spacing).astype(int))
        self.size = int(np.prod(self.shape))
        self.h0 = float(ground['top_z_m'])
        self.material = np.full(self.size, -1, dtype=np.int16)
        ix, iy = np.indices(self.shape)
        x = self.origin[0] + (ix.ravel()+.5)*self.spacing
        y = self.origin[1] + (iy.ravel()+.5)*self.spacing
        for i, tile in enumerate(tiles):
            mask = (x >= tile['x_min_m']) & (x < tile['x_max_m']) & (y >= tile['y_min_m']) & (y < tile['y_max_m'])
            self.material[mask] = i
        keys = ('kc', 'kphi', 'n', 'cohesion_pa', 'friction_deg', 'janosi_m',
                'elastic_pa_per_m', 'damping_pa_s_per_m', 'shear_damping_pa_s_per_m')
        self.parameters = {key:np.array([settings['soil_parameters'][t['id']][key] for t in tiles]) for key in keys}
        if (np.any(self.parameters['n'] <= 0) or np.any(self.parameters['janosi_m'] <= 0)
                or np.any(self.parameters['elastic_pa_per_m'] <= 0)
                or np.any(self.parameters['kphi'] <= 0)):
            raise ValueError('SCM exponents, stiffnesses and Janosi lengths must be positive')
        self.plastic = np.zeros(self.size)
        self.elastic = np.zeros(self.size)
        self.pressure = np.zeros(self.size)
        self.peak_sinkage = np.zeros(self.size)
        self.shear = np.zeros((self.size, 2))
        self.shear_travel = np.zeros(self.size)
        self.last_contact = np.full(self.size, -2, dtype=np.int64)
        self.step = 0
        self.active = np.empty(0, dtype=np.int64)
        self.dirty_materials = set(range(len(tiles)))
        self.max_rut = 0.0
        self.points = [np.asarray(p, dtype=float) for p in collider_points]
        self.planes = [self._planes(p) for p in self.points]
        self.slip_distance = [0.0, 0.0]
        self.track_surface = None
        self.surface_motion = {}
        self.grouser_fraction = {}

    @staticmethod
    def _planes(points):
        # The asset is an extrusion of its first half of XZ vertices along Y.
        polygon = points[:len(points)//2, :][:, [0, 2]]
        edge = np.roll(polygon, -1, axis=0)-polygon
        normals = np.column_stack((edge[:, 1], np.zeros(len(edge)), -edge[:, 0]))
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        bounds = normals[:, 0]*polygon[:, 0]+normals[:, 2]*polygon[:, 1]
        normals = np.vstack((normals, [0, 1, 0], [0, -1, 0]))
        bounds = np.r_[bounds, np.max(points[:, 1]), -np.min(points[:, 1])]
        # Collinear asset edges create redundant planes on the straight belt.
        _,unique = np.unique(np.round(np.column_stack((normals,bounds)),12),axis=0,return_index=True)
        return normals[unique], bounds[unique]

    def _footprint(self, side, position, rotation):
        self.surface_motion[side] = np.empty((0,3))
        self.grouser_fraction[side] = np.empty(0)
        world = self.points[side] @ rotation.T + position
        if self.track_surface is not None:
            # Shoe corners extend slightly beyond the nominal smooth envelope.
            world = np.vstack((world-.005,world+.005))
        lower = np.floor((world[:, :2].min(axis=0)-self.origin)/self.spacing-.5).astype(int)
        upper = np.ceil((world[:, :2].max(axis=0)-self.origin)/self.spacing+.5).astype(int)
        lower = np.maximum(lower, 0)
        upper = np.minimum(upper, self.shape)
        if np.any(lower >= upper):
            return np.empty(0, dtype=np.int64), np.empty((0,3)), np.empty(0)
        ix, iy = np.meshgrid(np.arange(lower[0], upper[0]), np.arange(lower[1], upper[1]), indexing='ij')
        ids = (ix.ravel()*self.shape[1]+iy.ravel()).astype(np.int64)
        xy = self.origin + (np.column_stack((ix.ravel(), iy.ravel()))+.5)*self.spacing
        # Conservative local-box rejection before the expensive convex rays.
        # This is especially important for a long narrow track at 45 degrees:
        # its world AABB contains several times the actual footprint area.
        middle_z = (world[:,2].min()+world[:,2].max())/2
        body_middle = (np.column_stack((xy,np.full(len(xy),middle_z)))-position) @ rotation
        allowance = ((world[:,2].max()-world[:,2].min())/2*np.abs(rotation[2,:2])
                     +self.spacing*.25*(np.abs(rotation[0,:2])+np.abs(rotation[1,:2])))
        local_min,local_max = self.points[side][:,:2].min(axis=0),self.points[side][:,:2].max(axis=0)
        if self.track_surface is not None:
            local_min,local_max = local_min-.005,local_max+.005
        candidate = np.all((body_middle[:,:2]+allowance >= local_min-1e-9)
                           & (body_middle[:,:2]-allowance <= local_max+1e-9),axis=1)
        ids,xy = ids[candidate],xy[candidate]
        if not len(ids):
            return np.empty(0,dtype=np.int64),np.empty((0,3)),np.empty(0)
        # Four subcell samples reduce footprint aliasing without changing soil
        # material width b when the numerical grid is refined.
        offsets = np.array([[-.25,-.25],[-.25,.25],[.25,-.25],[.25,.25]])*self.spacing
        subxy = (xy[:,None,:]+offsets).reshape(-1,2)
        origins = np.column_stack((subxy, np.zeros(len(subxy))))
        body = (origins-position) @ rotation
        if self.track_surface is not None:
            # Locate the lower run with the enclosing convex hull, rather than
            # the undeformed ground plane (which can lie inside/above a tilted
            # belt). Pad by 5 mm for the corners of the rectangular shoes.
            envelope_lo,envelope_valid = self.envelope_rays(side,body,rotation[2,:],padding=.005)
            guess = body+np.where(envelope_valid,envelope_lo,0.)[:,None]*rotation[2,:]
            lo,motion,lug,shoe = self.track_surface.ray_hits(side,body,rotation[2,:],guess)
            valid = np.isfinite(lo).reshape(-1,4)
            count = valid.sum(axis=1)
            z = np.where(valid,lo.reshape(-1,4),0.).sum(axis=1)/np.maximum(count,1)
            contact_xy = (np.where(valid[:,:,None],subxy.reshape(-1,4,2),0.).sum(axis=1)
                          /np.maximum(count,1)[:,None])
            averaged_motion = (np.where(valid[:,:,None],motion.reshape(-1,4,3),0.).sum(axis=1)
                               /np.maximum(count,1)[:,None]) @ rotation.T
            lug_fraction = (lug.reshape(-1,4)&valid).sum(axis=1)/np.maximum(count,1)
            keep = (count>0)&(self.material[ids]>=0)
            self.surface_motion[side] = averaged_motion[keep]
            self.grouser_fraction[side] = lug_fraction[keep]
            return ids[keep],np.column_stack((contact_xy[keep],z[keep])),self.spacing**2*count[keep]/4
        lo,valid = self.envelope_rays(side,body,rotation[2,:])
        valid = valid.reshape(-1,4)
        count = valid.sum(axis=1)
        z = np.where(valid, lo.reshape(-1,4), 0.).sum(axis=1)/np.maximum(count,1)
        contact_xy = (np.where(valid[:,:,None], subxy.reshape(-1,4,2), 0.).sum(axis=1)
                      /np.maximum(count,1)[:,None])
        keep = (count > 0) & (self.material[ids] >= 0)
        return ids[keep], np.column_stack((contact_xy[keep],z[keep])), self.spacing**2*count[keep]/4

    def envelope_rays(self,side,body,direction,padding=0.):
        normals, bounds = self.planes[side]
        bounds = bounds+padding
        margins = bounds[None,:]-np.einsum('ni,pi->np', body, normals)
        coeff = normals @ direction
        parallel = np.abs(coeff) < 1e-10
        divisions = np.zeros_like(margins)
        np.divide(margins, coeff, out=divisions, where=~parallel)
        lo = np.max(np.where(coeff < -1e-10, divisions, -np.inf), axis=1)
        hi = np.min(np.where(coeff > 1e-10, divisions, np.inf), axis=1)
        valid = (lo <= hi+1e-9) & np.all(margins[:,parallel] >= -1e-9, axis=1)
        return lo,valid

    def release_contacts(self):
        if len(self.active):
            self.dirty_materials.update(self.material[self.active].tolist())
            self.pressure[self.active] = 0.
            self.elastic[self.active] = 0.
            self.shear[self.active] = 0.
            self.last_contact[self.active] = -2
        self.active = np.empty(0, dtype=np.int64)

    def heights(self, xy, unloaded=False):
        xy = np.asarray(xy, dtype=float)
        index = np.floor((xy-self.origin)/self.spacing).astype(int)
        valid = np.all((index >= 0) & (index < self.shape), axis=1)
        clipped = np.minimum(np.maximum(index,0),np.array(self.shape)-1)
        ids = clipped[:,0]*self.shape[1]+clipped[:,1]
        valid &= self.material[ids] >= 0
        height = self.h0-self.plastic[ids]
        if not unloaded:
            height = height-self.elastic[ids]
        return np.where(valid,height,np.nan)

    def compute(self, position, quaternion, velocity, belt_speed, dt):
        started = time.perf_counter()
        if dt <= 0 or not np.isfinite(dt):
            raise ValueError('SCM requires a finite positive timestep')
        position, velocity = np.asarray(position), np.asarray(velocity)
        rotation = quaternion_matrix_xyzw(quaternion)
        forward = rotation[:2,0].copy()
        forward /= max(1e-9,np.linalg.norm(forward))
        lateral = np.array([-forward[1],forward[0]])
        old = self.active
        self.pressure[old] = 0.
        self.elastic[old] = 0.
        force, torque = np.zeros(3), np.zeros(3)
        active, stats = [], {}
        total_cells, total_area, dissipated, reentered = 0, 0., 0., 0
        shear_integral = 0.
        pressure_peaks, shear_peaks = [], []
        for side, name in enumerate(('left','right')):
            ids, points, area = self._footprint(side,position,rotation)
            z = self.h0-points[:,2]
            contact = z > self.plastic[ids]+1e-10
            ids, points, area, z = ids[contact], points[contact], area[contact], z[contact]
            load, traction, vx, lateral_rms, sinkage, rut = 0., 0., 0., 0., 0., 0.
            slip_velocity, ratio, valid_slip = float(belt_speed[side]), math.nan, False
            stress_stats = dict(pressure_mean_pa=math.nan, pressure_max_pa=math.nan,
                                shear_stress_mean_pa=math.nan, shear_stress_max_pa=math.nan,
                                shear_longitudinal_mean_pa=math.nan, shear_lateral_mean_pa=math.nan)
            if len(ids):
                reentered += int(np.count_nonzero((self.last_contact[ids] != self.step-1) & (self.plastic[ids] > 1e-4)))
                material = self.material[ids]
                param = {k:v[material] for k,v in self.parameters.items()}
                elastic_k = param['elastic_pa_per_m']
                envelope = (param['kc']/self.robot['tracks']['width_m']+param['kphi'])*z**param['n']
                trial = elastic_k*np.maximum(z-self.plastic[ids],0.)
                static_pressure = np.minimum(trial,envelope)
                self.plastic[ids] = np.maximum(self.plastic[ids],z-static_pressure/elastic_k)
                self.elastic[ids] = static_pressure/elastic_k
                self.peak_sinkage[ids] = np.maximum(self.peak_sinkage[ids],z)
                lever = points-position
                body_point_velocity = velocity[:3]+np.cross(velocity[3:],lever)
                point_velocity = body_point_velocity.copy()
                if self.track_surface is not None:
                    point_velocity += self.surface_motion[side][contact]
                # No tension: damping acts only while the soil is compressed.
                pressure = np.maximum(0., static_pressure-param['damping_pa_s_per_m']*point_velocity[:,2])
                self.pressure[ids] = pressure
                continuous = self.last_contact[ids] == self.step-1
                j = np.where(continuous[:,None],self.shear[ids],0.)
                relative = point_velocity[:,:2].copy()
                if self.track_surface is None:
                    relative -= belt_speed[side]*forward
                j += relative*dt
                magnitude = np.linalg.norm(j,axis=1)
                cap = self.settings['shear_memory_limit_K']*param['janosi_m']
                j *= np.minimum(1., cap/np.maximum(magnitude,1e-12))[:,None]
                magnitude = np.linalg.norm(j,axis=1)
                limit = param['cohesion_pa']+pressure*np.tan(np.radians(param['friction_deg']))
                stress = -j/np.maximum(magnitude,1e-12)[:,None]*(limit*(-np.expm1(-magnitude/param['janosi_m'])))[:,None]
                stress -= param['shear_damping_pa_s_per_m'][:,None]*relative
                stress *= np.minimum(1.,limit/np.maximum(np.linalg.norm(stress,axis=1),1e-12))[:,None]
                tangential = stress*area[:,None]
                normal = pressure*area
                covered = float(area.sum())
                stress_norm = np.linalg.norm(stress, axis=1)
                stress_stats = dict(
                    pressure_mean_pa=float(normal.sum()/covered), pressure_max_pa=float(pressure.max()),
                    shear_stress_mean_pa=float(stress_norm @ area/covered),
                    shear_stress_max_pa=float(stress_norm.max()),
                    shear_longitudinal_mean_pa=float((stress @ forward) @ area/covered),
                    shear_lateral_mean_pa=float((stress @ lateral) @ area/covered))
                shear_integral += float(stress_norm @ area)
                pressure_peaks.append(stress_stats['pressure_max_pa'])
                shear_peaks.append(stress_stats['shear_stress_max_pa'])
                forces = np.column_stack((tangential,normal))
                force += forces.sum(axis=0)
                torque += np.cross(lever,forces).sum(axis=0)
                self.shear[ids] = j
                self.shear_travel[ids] += np.linalg.norm(relative,axis=1)*dt
                self.last_contact[ids] = self.step
                load = float(normal.sum())
                traction = float(tangential.sum(axis=0) @ forward)
                weights = normal/max(load,1e-12)
                patch_vx = body_point_velocity[:,:2] @ forward
                vx = float(weights @ patch_vx)
                lateral_rms = float(np.sqrt(weights @ (relative @ lateral)**2))
                slip_velocity = float(-weights @ (relative @ forward))
                valid_slip = load > .5 and max(abs(belt_speed[side]),abs(vx)) >= .05
                ratio = slip_velocity/max(abs(belt_speed[side]),abs(vx),.05) if valid_slip else math.nan
                if load > .5:
                    self.slip_distance[side] += float(weights @ np.abs(relative @ forward))*dt
                sinkage, rut = float(z.max()), float(self.plastic[ids].max())
                self.max_rut = max(self.max_rut,rut)
                dissipated += float(np.sum(param['damping_pa_s_per_m']*point_velocity[:,2]**2*area))
                total_cells += len(ids)
                total_area += float(area.sum())
                active.append(ids)
            stats.update({
                name+'_normal_n':load, name+'_traction_n':traction,
                name+'_contact_cells':len(ids), name+'_contact_area_m2':float(area.sum()),
                name+'_patch_vx_m_s':vx, name+'_slip_velocity_m_s':slip_velocity,
                name+'_slip_ratio':ratio, name+'_slip_valid':int(valid_slip),
                name+'_ground_contact':int(load>.5), name+'_lateral_slip_rms_m_s':lateral_rms,
                name+'_slip_distance_m':self.slip_distance[side], name+'_sinkage_m':sinkage,
                name+'_rut_depth_m':rut, name+'_sinkage_valid':int(bool(len(ids))),
                name+'_grouser_contact_cells':int(np.count_nonzero(self.grouser_fraction[side][contact]>0))
                    if self.track_surface is not None else 0,
            })
            stats.update({name+'_'+key:value for key,value in stress_stats.items()})
        self.active = np.concatenate(active) if active else np.empty(0,dtype=np.int64)
        self.dirty_materials.update(self.material[old].tolist())
        self.dirty_materials.update(self.material[self.active].tolist())
        self.step += 1
        stats.update(
            soil_pressure_mean_pa=float(force[2]/total_area) if total_area else math.nan,
            soil_pressure_max_pa=max(pressure_peaks, default=math.nan),
            soil_shear_stress_mean_pa=shear_integral/total_area if total_area else math.nan,
            soil_shear_stress_max_pa=max(shear_peaks, default=math.nan))
        stats.update(scm_contact_cells=total_cells, scm_contact_area_m2=total_area,
                     scm_reentered_rut_cells=reentered,
                     soil_rut_depth_max_m=self.max_rut, soil_normal_force_n=float(force[2]),
                     soil_damping_power_w=dissipated, soil_force_x_n=float(force[0]),
                     soil_force_y_n=float(force[1]), soil_torque_z_nm=float(torque[2]),
                     scm_step_ms=(time.perf_counter()-started)*1000)
        return force,torque,stats

    def save(self,path):
        np.savez_compressed(Path(path), shape=self.shape, origin=self.origin,
            spacing=self.spacing, h0=self.h0, plastic=self.plastic.reshape(self.shape),
            elastic=self.elastic.reshape(self.shape), pressure=self.pressure.reshape(self.shape),
            peak_sinkage=self.peak_sinkage.reshape(self.shape), shear_travel=self.shear_travel.reshape(self.shape),
            settings_json=json.dumps(self.settings), ground_json=json.dumps(self.ground))


class SCMTrackForces(AmphibianForces):
    """Reuse actuator/non-ground terms; replace all original track forces."""
    def __init__(self,config,terrain):
        self.terrain = terrain
        self.soil_metrics = {}
        super().__init__(config)

    def reset(self):
        super().reset()
        self.terrain.release_contacts()
        self.terrain.slip_distance = [0.,0.]
        if self.terrain.track_surface is not None:
            self.terrain.track_surface.reset()

    def compute(self,position,quaternion,velocity,commands,contacts,environment,dt,buoyancy_enabled=True):
        zero_contacts = [{'normal_n':0.,'normal':[0.,0.,1.],'mu':0.} for _ in range(2)]
        force,torque,state = super().compute(position,quaternion,velocity,commands,
            zero_contacts,environment,dt,buoyancy_enabled)
        if self.terrain.track_surface is not None:
            self.terrain.track_surface.set_speeds(self.belt_speed)
            self.terrain.track_surface.advance(self.belt_speed,dt)
        soil_force,soil_torque,self.soil_metrics = self.terrain.compute(position,quaternion,velocity,self.belt_speed,dt)
        force += soil_force
        torque += soil_torque
        state.update(self.soil_metrics)
        if self.terrain.track_surface is not None:
            state.update(self.terrain.track_surface.telemetry())
        state['native_track_normal_n'] = sum(c['normal_n'] for c in contacts)
        state['track_mechanical_power_w'] = sum(abs(state[n+'_traction_n']*self.belt_speed[i])
                                                for i,n in enumerate(('left','right')))
        state.update(force_x_n=float(force[0]),force_y_n=float(force[1]),force_z_n=float(force[2]),
                     torque_x_nm=float(torque[0]),torque_y_nm=float(torque[1]),torque_z_nm=float(torque[2]))
        return force,torque,state
