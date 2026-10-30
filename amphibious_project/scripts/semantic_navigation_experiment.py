"""Native-physics semantic episode with independent perception and waypoint workers."""
import asyncio
import csv
import math
from pathlib import Path
import time
import traceback
import uuid

import omni.kit.app
import omni.timeline
import omni.usd

from approach_planning import ObservedGrid
from navigation_experiment import NavigationExperiment,NavigationRuntime
from navigation_io import own_path,read_json,write_json,file_sha
from navigation_loop import NavigationSession
from navigation_execution import safe_polyline_prefix
from navigation_observations import ObservationBuffer
from semantic_execution import SemanticPhysicalBridge
import robot_runtime


class SemanticNavigationExperiment(NavigationExperiment):
    def __init__(self,project,request,rig_factory=None):
        self.project=Path(project).resolve();self.request=request
        self.rig_factory=rig_factory
        if request.get('schema')!='L17.semantic_navigation_run.v1' or request.get('mode')!='execute':raise ValueError('Invalid semantic mission request')
        if float(request.get('start_deadline_wall_unix_s',0))<time.time():raise ValueError('Semantic start request expired')
        self.cfg=read_json(self.project/'configs/navigation_closed_loop.json')
        self.cfg.update(read_json(self.project/'configs/semantic_navigation.json'))
        if request.get('target_id')!=self.cfg['target_id'] or request.get('instruction')!=self.cfg['instruction']:
            raise ValueError('Use the configured target appearance instruction')
        self.output=own_path(self.project,request['output_relative']);self.stop_file=own_path(self.project,request['stop_relative'])
        self.queue=own_path(self.project,self.cfg['worker_queue']);self.semantic_queue=own_path(self.project,self.cfg['semantic_queue'])
        self.runtime=self.session=self.future=None;self.reason=self.error=None;self.phase='prepared';self.cycles=0
        self.started_wall=time.monotonic();self.output_created=False;self.cleanup_errors=[]
        self.buffer=ObservationBuffer(self.project);self.semantic_events=[];self.pending_semantic=None
        self.last_plan=None;self.scan_count=0;self.unsafe_head_count=0;self.motion_positions=[]

    def semantic_ready(self):
        ready=read_json(self.semantic_queue/'ready.json');beat=read_json(self.semantic_queue/'heartbeat.json')
        if not ready.get('model_loaded') or ready.get('project')!=str(self.project):raise RuntimeError('Own semantic model worker is not ready')
        if beat['worker_id']!=ready['worker_id'] or not -1<=time.time()-beat['wall_unix_s']<=5:raise RuntimeError('Semantic worker heartbeat is stale')
        for name,digest in ready['code_sha256'].items():
            if file_sha(own_path(self.project,name))!=digest:raise RuntimeError('Semantic worker has older source; restart it')
        stopped=self.semantic_queue/'stopped.json'
        if stopped.exists() and read_json(stopped)['worker_id']==ready['worker_id']:raise RuntimeError('Semantic worker stopped')
        return ready

    async def await_semantic_plan(self,path,observation):
        ready=self.semantic_ready();request_id=uuid.uuid4().hex
        request={'schema':'L17.semantic_request.v1','request_id':request_id,'worker_id':ready['worker_id'],
                 'run_id':self.request['run_id'],'step':self.cycles,'controller_epoch':self.runtime.clock_epoch,
                 'observation_file':path.relative_to(self.project).as_posix(),'observation_sha256':file_sha(path),
                 'instruction':self.request['instruction'],'state':dict(self.runtime.latest),
                 'output_relative':self.request['output_relative'],'deadline_wall_unix_s':time.time()+self.cfg['worker_timeout_s']}
        reply=self.semantic_queue/(request_id+'.result.json');self.pending_semantic=request_id
        write_json(self.semantic_queue/(request_id+'.request.json'),request)
        started=time.monotonic();sim=float(self.runtime.sim_time)
        self.phase='semantic_perception_and_route_planning_paused'
        while not reply.exists():
            if time.monotonic()-started>self.cfg['worker_timeout_s']:raise RuntimeError('Semantic model/planner timed out')
            stopped=self.semantic_queue/'stopped.json'
            if stopped.exists() and read_json(stopped)['worker_id']==ready['worker_id']:raise RuntimeError('Semantic worker exited during request')
            await self.update()
            if abs(self.runtime.sim_time-sim)>1e-6:raise RuntimeError('Physics advanced during semantic inference')
        result=read_json(reply);self.pending_semantic=None
        write_json(self.output/f'semantic_{self.cycles:04d}.json',result)
        for key in ['request_id','worker_id','controller_epoch','observation_sha256']:
            if result.get(key)!=request[key]:raise RuntimeError('Semantic result identity mismatch: '+key)
        if not result.get('ok'):raise RuntimeError('Semantic worker error: '+str(result.get('error')))
        self.semantic_events.append({'step':self.cycles,'event':'semantic_plan_accepted','wall_s':time.monotonic()-started,
                                    'advanced_sim_s':self.runtime.sim_time-sim,'target_distance_m':result['result']['target_distance_m'],
                                    'route_kind':(result['result'].get('route') or {}).get('kind')})
        return result['result']

    async def waypoint_plan(self,path,observation,goal):
        worker=self.worker_ready();self.session.submit(path,observation,goal,self.queue,time.monotonic())
        self.phase='waypoint_model_inference_paused';sim=float(self.runtime.sim_time)
        reply=self.session.pending['reply_path']
        while not reply.exists():
            if self.session.check_timeout(time.monotonic()):raise RuntimeError('Waypoint worker timeout')
            stopped=self.queue/'stopped.json'
            if stopped.exists() and read_json(stopped)['worker_id']==worker['worker_id']:raise RuntimeError('Waypoint worker exited')
            await self.update()
            if abs(self.runtime.sim_time-sim)>1e-6:raise RuntimeError('Physics advanced during waypoint inference')
        result=read_json(reply);write_json(self.output/f'head_{self.cycles:04d}.json',result)
        self.session.events.append({'event':'inference_physics_pause_verified','start_sim_s':sim,
                                    'end_sim_s':self.runtime.sim_time,'advanced_sim_s':self.runtime.sim_time-sim})
        return self.session.accept(result,self.runtime.latest,time.monotonic())

    async def scan(self,rig):
        self.scan_count+=1
        if self.scan_count>self.cfg['maximum_active_perception_scans']:raise RuntimeError('Observed map still disconnected after active perception scans')
        self.phase='active_perception_scan'
        self.runtime.bridge.scan_to(self.runtime.latest['yaw_rad']+math.pi/2)
        await self.advance(6.,drive=True)
        self.runtime.bridge.clear_plan()
        await self.advance(.5)
        await self.capture(rig)
        self.semantic_events.append({'step':self.cycles,'event':'physical_active_perception_scan','scan_number':self.scan_count})

    async def run(self):
        timeline=omni.timeline.get_timeline_interface()
        try:
            self.output.mkdir(parents=True,exist_ok=False);self.output_created=True
            write_json(self.output/'request.json',self.request)
            if not self.request.get('synthetic_test_fixture_only'):
                from navigation_audit import verify_baseline_freeze
                write_json(self.output/'model_integrity_at_start.json',verify_baseline_freeze(self.project))
            write_json(self.output/'worker_at_start.json',self.worker_ready())
            write_json(self.output/'semantic_worker_at_start.json',self.semantic_ready())
            stage=omni.usd.get_context().get_stage();task=read_json(self.project/'configs/task.json')
            if not stage or Path(stage.GetRootLayer().realPath).resolve()!=own_path(self.project,task['scene_file']):
                raise RuntimeError('Open the configured own robot scene before semantic navigation')
            # Remove privileged task-marker visuals from perception. Target
            # geometry remains visible; marker transforms are never inspected.
            from pxr import Usd,UsdGeom
            with Usd.EditContext(stage,stage.GetSessionLayer()):
                for name in ['/World/TaskMarkers','/World/NavigationTargets/WaterApproachGoal','/World/L17_RGBD_CalibrationFixture']:
                    prim=stage.GetPrimAtPath(name)
                    if prim:UsdGeom.Imageable(prim).MakeInvisible()
            current=robot_runtime.get_robot_runtime()
            if current and any(abs(v)>1e-8 for k in ['tracks','thrusters'] for v in current.commands[k]):raise RuntimeError('Own manual commands are nonzero')
            timeline.pause();robot_runtime.detach_robot_runtime()
            robot=read_json(self.project/'configs/robot_v1.json');control=read_json(self.project/'configs/crossing_experiment.json')
            control['short_waypoint_tolerance_m']=self.cfg['short_waypoint_tolerance_m']
            bridge=SemanticPhysicalBridge(robot,control,self.cfg['max_plan_age_s'],self.cfg['max_step_m'])
            self.runtime=NavigationRuntime(self.project,bridge,self);robot_runtime._RUNTIME=self.runtime
            self.session=NavigationSession(self.project,self.runtime.clock_epoch,bridge,self.cfg['worker_timeout_s'])
            if self.rig_factory is None:
                import rgbd_sensor_rig
                rgbd_sensor_rig.dispose_rig();source=self.project/'scripts/rgbd_sensor_rig.py'
                exec(compile(source.read_text(),str(source),'exec'),rgbd_sensor_rig.__dict__)
                rig=rgbd_sensor_rig.get_rig(self.project)
            else:rig=self.rig_factory(self.project)
            self.phase='settling';await self.advance(self.cfg['settle_s'])
            for i in range(5):
                if i:await self.advance(self.cfg['history_spacing_s'])
                await self.capture(rig)
            for self.cycles in range(1,self.cfg['max_inference_cycles']+1):
                self.check()
                observation=self.buffer.observation(self.request['instruction'])
                path=self.output/f'observation_{self.cycles:04d}.json';write_json(path,observation)
                self.last_plan=await self.await_semantic_plan(path,observation)
                self.motion_positions.append(observation['anchor_pose_xyyaw'][:2])
                if self.last_plan['semantic_arrival_candidate']:
                    speed=math.hypot(self.runtime.latest['vx_m_s'],self.runtime.latest['vy_m_s'])
                    if speed<self.cfg['arrival_speed_m_s']:
                        self.reason='semantic_flat_top_jacket_reached';break
                    self.phase='arrival_braking';await self.advance(.75);await self.capture(rig);continue
                if self.last_plan['need_scan']:
                    await self.scan(rig)
                    write_json(self.output/'progress.json',{**self.status(),'target_distance_m':self.last_plan['target_distance_m'],
                               'scans':self.scan_count,'planner_status':'active_perception_required'})
                    continue
                grid_path=own_path(self.project,self.last_plan['grid_file'])
                grid=ObservedGrid.load(grid_path,footprint_radius_m=self.last_plan['footprint_radius_m'])
                # The mapper can seed a route from its measured-support cell,
                # even when its fully inflated footprint is not yet observed.
                # Check the identical execution footprint before requesting a
                # model waypoint; no model can repair an invalid start state.
                if not grid.is_safe(grid.cell(observation['anchor_pose_xyyaw'][:2])):
                    self.semantic_events.append({'step':self.cycles,'event':'unvalidated_route_start',
                        'robot_xy_m':observation['anchor_pose_xyyaw'][:2],
                        'grid_file':self.last_plan['grid_file'],'waypoint_model_called':False})
                    await self.scan(rig)
                    continue
                route=self.last_plan['route']
                goal={'kind':'robot_arrival_pose','source':'observed_traversability','approach_validated':True,
                      'goal_xy_m':route['local_goal_xy_m'],'instruction':self.request['instruction'],
                      'grid_file':grid_path.relative_to(self.project).as_posix(),'semantic_route_subgoal':True}
                plan=await self.waypoint_plan(path,observation,goal)
                polyline=plan.get('path_xy_m') or [observation['anchor_pose_xyyaw'][:2],plan['goal_xy_m']]
                start_xy=polyline[0]
                if plan.get('path_length_m',0.)>.08:
                    prefix,kept=safe_polyline_prefix(polyline,grid.segment_is_safe)
                    if len(prefix)<2 or kept<.08:
                        self.session.cancel('head_segment_outside_observed_safe_space');self.unsafe_head_count+=1
                        self.semantic_events.append({'step':self.cycles,'event':'unsafe_head_rejected','head_goal_xy_m':plan['goal_xy_m']})
                        if self.unsafe_head_count>=3:raise RuntimeError('Dedicated head repeatedly proposes a segment outside observed safe space')
                        await self.scan(rig);continue
                    if len(prefix)<len(polyline):
                        original=plan.get('path_length_m',0.)
                        plan=self.runtime.bridge.truncate_plan(prefix)
                        self.semantic_events.append({'step':self.cycles,'event':'head_polyline_truncated',
                                                     'original_m':original,'kept_m':kept})
                elif not grid.is_safe(grid.cell(start_xy)):
                    self.session.cancel('head_segment_outside_observed_safe_space');self.unsafe_head_count+=1
                    self.semantic_events.append({'step':self.cycles,'event':'unsafe_head_rejected','head_goal_xy_m':plan['goal_xy_m']})
                    if self.unsafe_head_count>=3:raise RuntimeError('Dedicated head repeatedly proposes a segment outside observed safe space')
                    await self.scan(rig);continue
                self.phase='executing_model_waypoint';await self.advance(self.cfg['execution_slice_s'],drive=True)
                bridge.clear_plan();await self.advance(self.cfg['history_spacing_s']);await self.capture(rig)
                write_json(self.output/'progress.json',{**self.status(),'target_distance_m':self.last_plan['target_distance_m'],
                           'last_pose_xyyaw':self.buffer.frames[-1]['anchor_pose_xyyaw'],'route_kind':route['kind'],
                           'active_perception_scans':self.scan_count})
                if len(self.motion_positions)>30 and math.dist(self.motion_positions[-1],self.motion_positions[-25])<.25:
                    raise RuntimeError('No physical navigation progress over 25 model decisions')
            if not self.reason:self.reason='semantic_inference_cycle_limit'
        except Exception:
            self.error=traceback.format_exc()
            if not self.reason:self.reason='semantic_navigation_stopped'
        finally:
            if self.runtime:
                self.runtime.enabled=False;self.runtime.zero()
                if self.runtime.stage==omni.usd.get_context().get_stage():
                    try:await self.pause_physics()
                    except Exception:self.cleanup_errors.append(traceback.format_exc())
            if self.pending_semantic:
                write_json(self.semantic_queue/(self.pending_semantic+'.cancelled.json'),{'reason':self.reason})
            if self.session and self.session.pending:self.session.cancel(self.reason or 'semantic_finished')
            try:
                if self.output_created:
                    if self.runtime and self.runtime.history:self.runtime.save_csv(self.output/'telemetry.csv')
                    if self.runtime and self.runtime.control_rows:
                        with (self.output/'control.csv').open('w',newline='') as handle:
                            writer=csv.DictWriter(handle,fieldnames=list(self.runtime.control_rows[0]));writer.writeheader();writer.writerows(self.runtime.control_rows)
                    write_json(self.output/'semantic_events.json',self.semantic_events)
                    write_json(self.output/'events.json',self.session.events if self.session else [])
            except Exception:self.cleanup_errors.append(traceback.format_exc())
            finally:
                if self.runtime and robot_runtime.get_robot_runtime() is self.runtime:
                    robot_runtime.detach_robot_runtime()
                    if self.runtime.stage==omni.usd.get_context().get_stage():
                        try:robot_runtime.attach_robot_runtime(self.project,'river')
                        except Exception:self.cleanup_errors.append(traceback.format_exc())
            self.phase='finished'
            if self.output_created:
                summary={**self.status(),'target_instruction':self.request['instruction'],
                         'semantic_visual_arrival_verified':self.reason=='semantic_flat_top_jacket_reached',
                         'independent_ground_truth_evaluation_pending':True,'manual_target_coordinate_used':False,
                         'hidden_scene_map_used_for_planning':False,'pose_or_velocity_overwrite_during_navigation':False,
                         'capture_and_inference_pause_physics':True,'switch_policy':'contact_buoyancy_reference_without_scene_depth',
                         'scans':self.scan_count,'unsafe_heads_rejected':self.unsafe_head_count,'cleanup_errors':self.cleanup_errors,
                         'last_observed_target_xy_m':self.last_plan.get('target_surface_xy_m') if self.last_plan else None}
                write_json(self.output/'summary.json',summary)
                try:
                    from navigation_audit import write_audit
                    write_audit(self.output)
                except Exception:
                    write_json(self.output/'mistake_audit_error.json',{'error':traceback.format_exc()})
            print('L17_SEMANTIC_NAVIGATION_FINISHED',self.reason,str(self.output),flush=True)
