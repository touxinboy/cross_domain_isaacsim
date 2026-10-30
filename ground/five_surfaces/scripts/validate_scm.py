"""Native PhysX validation: soil response and optional circulating-shoe model."""
import asyncio
import csv
import faulthandler
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import omni.kit.app
import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'scripts'))


async def validate():
    app=omni.kit.app.get_app()
    track_settings=json.loads((PROJECT/'configs/tracks.json').read_text())
    moving=track_settings['enabled']
    output=PROJECT/('results/tracks_validation' if moving else 'results/scm_validation')/datetime.now().strftime('run_%Y%m%d_%H%M%S')
    output.mkdir(parents=True,exist_ok=False)
    diagnostic=(output/'diagnostic.txt').open('w',buffering=1)
    faulthandler.dump_traceback_later(60,repeat=True,file=diagnostic)
    diagnostic.write('Opening scene\n')
    report={'passed':False,'tested_project':str(PROJECT),'checks':{},'experiments':{}}
    tools=None

    def check(name,condition):
        report['checks'][name]=bool(condition)
        if not condition:
            raise AssertionError(name)

    async def advance(duration):
        target=tools.runtime.sim_time+duration
        deadline=time.perf_counter()+120
        while tools.runtime.sim_time < target:
            if tools.runtime.error:
                raise RuntimeError(tools.runtime.error)
            if time.perf_counter()>deadline:
                raise RuntimeError('Native physics advance timeout')
            await app.next_update_async()

    def save_trial(name):
        tools.runtime.save_csv(output/(name+'.csv'))
        report['experiments'][name]={k:tools.runtime.latest[k] for k in (
            'x_m','y_m','z_m','vz_m_s','yaw_rad','soil_normal_force_n','soil_rut_depth_max_m',
            'scm_contact_cells','rtf_average','scm_step_ms')}
        print('SCM_EXPERIMENT '+name+' '+json.dumps(report['experiments'][name]),flush=True)

    async def capture_robot(name):
        from robot_assets import set_camera
        from omni.kit.viewport.utility import capture_viewport_to_file,get_active_viewport
        tools.runtime.timeline.pause()
        p=np.array([tools.runtime.latest[k] for k in ('x_m','y_m','z_m')])
        if tools.track_view:
            tools.track_view.update(tools.runtime.sim_time,force=True)
        with Usd.EditContext(stage,stage.GetSessionLayer()):
            set_camera(stage,'/World/Cameras/Side',p+np.array([1.35,-1.6,.85]),p+np.array([0.,0.,-.02]),35)
        tools.camera('Side')
        for _ in range(10):
            await app.next_update_async()
        capture=capture_viewport_to_file(get_active_viewport(),str(output/(name+'.png')))
        await capture.wait_for_result()

    try:
        from open_stage import open_scene
        tools=await open_scene(scm_enabled=True)
        diagnostic.write('Scene and controls ready\n')
        stage=omni.usd.get_context().get_stage()
        check('single_robot_body',sum(p.HasAPI(UsdPhysics.RigidBodyAPI) for p in Usd.PrimRange(stage.GetPrimAtPath('/World/Robot')))==1)
        check('robot_mass_preserved',UsdPhysics.MassAPI(stage.GetPrimAtPath('/World/Robot')).GetMassAttr().Get()==48.)
        if moving:
            check('two_instanced_bands',len(tools.track_view.instancers)==2)
            check('forty_eight_shoes_per_side',all(len(i.GetProtoIndicesAttr().Get())==48 for i in tools.track_view.instancers))
            check('no_new_rigid_bodies_or_colliders',not any(p.HasAPI(UsdPhysics.RigidBodyAPI) or p.HasAPI(UsdPhysics.CollisionAPI)
                for p in Usd.PrimRange(stage.GetPrimAtPath('/World/Robot/TrackMotion'))))
            check('original_static_belt_hidden',stage.GetPrimAtPath('/World/Robot/Tracks').GetAttribute('visibility').Get()=='invisible')
        check('all_original_ground_colliders_disabled',all(not UsdPhysics.CollisionAPI(
            stage.GetPrimAtPath('/World/Ground/'+t['id'])).GetCollisionEnabledAttr().Get() for t in tools.config['surfaces']))
        check('deformed_visual_meshes_have_no_collision',all(not stage.GetPrimAtPath(
            '/World/SCM/'+t['id']).HasAPI(UsdPhysics.CollisionAPI) for t in tools.config['surfaces']))

        await tools.reset(position=[-4.,0.,.24])
        diagnostic.write('Compression reset complete\n')
        tools.runtime.timeline.play()
        await advance(4.)
        runtime=tools.runtime
        tail=runtime.history[-60:]
        weight=runtime.cfg['mass_kg']*runtime.cfg['gravity_m_s2']
        check('compression_supports_weight',abs(np.mean([r['soil_normal_force_n'] for r in tail])/weight-1)<.08)
        check('compression_reaches_stable_height',abs(runtime.latest['vz_m_s'])<.02)
        check('compression_generates_sinkage',.12<runtime.latest['z_m']<.215)
        check('distributed_contact_exceeds_four_patches',runtime.latest['scm_contact_cells']>100)
        check('native_ground_does_not_duplicate_support',max(r['native_track_normal_n'] for r in runtime.history)<1.)
        save_trial('compression')
        if moving:
            check('grousers_carry_real_soil_contact',runtime.latest['left_grouser_contact_cells']>10 and runtime.latest['right_grouser_contact_cells']>10)
            await capture_robot('tracks_closeup')
            runtime.timeline.play()
        tools.pause()
        await app.next_update_async()
        soil=runtime.terrain
        plastic=soil.plastic.copy()
        old_ids=np.flatnonzero(plastic>.001)
        check('compression_leaves_plastic_state',len(old_ids)>100)
        loaded_elastic=float(soil.elastic.max())
        await tools.reset(clear_soil=False,position=[-4.,1.5,.24])
        check('vehicle_reset_preserves_soil_object',tools.runtime.terrain is soil)
        check('unload_retains_ruts',np.array_equal(soil.plastic,plastic))
        check('unload_releases_elastic_pressure',soil.elastic.max()==0 and soil.pressure.max()==0)
        check('loaded_state_had_elastic_component',loaded_elastic>0)
        soil.save(output/'unloaded_soil.npz')

        await tools.reset(clear_soil=False,position=[-4.,0.,.24])
        tools.runtime.timeline.play()
        await advance(3.)
        check('revisit_uses_previously_deformed_cells',np.count_nonzero(tools.runtime.terrain.plastic[old_ids]>=plastic[old_ids]-1e-9)==len(old_ids))
        check('revisit_remains_stable',abs(tools.runtime.latest['vz_m_s'])<.03)
        save_trial('repeated_loading')

        tools.drive(.45,.45)
        start=tools.runtime.latest['x_m']
        first=len(tools.runtime.history)
        await advance(3.)
        travel=tools.runtime.history[first:]
        check('traction_moves_vehicle_forward',tools.runtime.latest['x_m']-start>.5)
        check('traction_from_belt_shear',any(r['left_traction_n']+r['right_traction_n']>10. for r in travel))
        check('traction_records_finite_slip',any(r['left_slip_valid'] and math.isfinite(r['left_slip_ratio']) for r in travel))
        save_trial('straight_traction')
        if moving:
            motion=tools.runtime.track_motion
            check('belt_phase_integrates_filtered_speed',abs(motion.distance[0]-sum(r['left_belt_m_s']/120 for r in tools.runtime.history))<1e-8)
            tools.track_view.update(tools.runtime.sim_time,force=True)
            check('render_contact_share_shoe_positions',all(np.allclose(np.array(links.GetPositionsAttr().Get()),motion.poses(i)[0],atol=1e-6)
                for i,links in enumerate(tools.track_view.instancers)))
            check('sprocket_rotates',abs(stage.GetPrimAtPath('/World/Robot/TrackMotion/Left/DriveSprocket').GetAttribute('xformOp:orient').Get().GetImaginary()[1])>.01)
            await capture_robot('tracks_after_forward')
            runtime.timeline.play()
            start=runtime.latest['x_m']
            tools.drive(-.35,-.35)
            await advance(2.)
            check('reverse_drive_moves_backward',runtime.latest['x_m']<start-.15)
            save_trial('reverse_traction')

        await tools.reset(clear_soil=False,position=[-4.,0.,.24])
        tools.runtime.timeline.play()
        await advance(2.)
        first=len(tools.runtime.history)
        start=tools.runtime.latest['x_m']
        tools.drive(.45,.45)
        await advance(3.)
        check('repeated_pass_moves_over_same_route',tools.runtime.latest['x_m']-start>.5)
        check('repeated_pass_reenters_retained_ruts',sum(r['scm_reentered_rut_cells'] for r in tools.runtime.history[first:])>100)
        save_trial('repeated_pass')

        tools.drive(0.,0.)
        await advance(1.)
        yaw=tools.runtime.latest['yaw_rad']
        first=len(tools.runtime.history)
        tools.drive(-.35,.35)
        await advance(3.)
        angle=tools.runtime.latest['yaw_rad']-yaw
        check('differential_belts_turn_vehicle',math.atan2(math.sin(angle),math.cos(angle))>.2)
        check('turn_includes_lateral_slip',max(r['left_lateral_slip_rms_m_s'] for r in tools.runtime.history[first:])>.01)
        save_trial('turning')
        tools.pause()
        await app.next_update_async()
        export=tools.export()
        check('export_has_soil_history',(export/'soil_state.npz').is_file())
        summary=json.loads((export/'summary.json').read_text())
        check('export_identifies_uncalibrated_model',summary['scm_settings']['parameters_are_calibrated'] is False)
        if moving:
            check('export_records_shoe_phase',all(k in summary['final_metrics'] for k in ('left_belt_phase_m','right_belt_phase_m','left_sprocket_angle_rad')))
            check('export_identifies_approximation',not summary['track_settings']['independent_link_dynamics'] and not summary['track_settings']['suspension_dynamics'])

        # Complete a fresh route with all five material parameter sets.
        await tools.run_route()
        deadline=time.perf_counter()+240
        while tools.runtime.running_route:
            if tools.runtime.error:
                raise RuntimeError(tools.runtime.error)
            if time.perf_counter()>deadline:
                raise RuntimeError('SCM five-surface route timeout')
            await app.next_update_async()
        runtime=tools.runtime
        check('five_surface_route_completed',runtime.route_reason=='completed')
        check('five_surface_contact',runtime.visited=={t['id'] for t in tools.config['surfaces']})
        check('route_keeps_native_support_zero',max(r['native_track_normal_n'] for r in runtime.history)<1.)
        check('route_schema_consistent',all(set(r)==set(runtime.history[0]) for r in runtime.history))
        check('route_stable',max(abs(r['roll_deg']) for r in runtime.history)<15 and max(abs(r['pitch_deg']) for r in runtime.history)<15)
        check('panel_displays_ruts','Ruts max' in tools.metrics_label.text)
        save_trial('five_surfaces')
        runtime.terrain.save(output/'five_surfaces_soil.npz')
        (output/'panel_text.txt').write_text(tools.metrics_label.text+'\n')
        # Leave the vehicle elsewhere so an oblique camera can see unloaded ruts.
        await tools.reset(clear_soil=False,position=[8.5,1.5,.24])
        check('retained_meshes_reconstruct_all_material_ruts',all(np.allclose(points[:,2],tools.runtime.terrain.heights(query),atol=1e-6)
            for mesh,points,query,color,base in tools.terrain_view.meshes))
        check('vertex_colors_match_geometry',all(len(color.Get())==len(points)
            for mesh,points,query,color,base in tools.terrain_view.meshes))
        from robot_assets import set_camera
        with Usd.EditContext(stage,stage.GetSessionLayer()):
            set_camera(stage,'/World/Cameras/Side',(5.,-2.0,1.6),(3.8,0.,-.02),35)
        tools.camera('Side')
        from omni.kit.viewport.utility import capture_viewport_to_file,get_active_viewport
        for _ in range(15):
            await app.next_update_async()
        capture=capture_viewport_to_file(get_active_viewport(),str(output/'ruts.png'))
        await capture.wait_for_result()
        report['performance']={'simulated_s':runtime.sim_time,'active_wall_s':runtime.latest['active_wall_time_s'],
            'rtf_average':runtime.latest['rtf_average'],'mean_scm_ms':float(np.mean([r['scm_step_ms'] for r in runtime.history])),
            'p95_scm_ms':float(np.percentile([r['scm_step_ms'] for r in runtime.history],95))}

        # Explicitly verify that the switch restores the actual rigid colliders.
        await tools.reset(scm_enabled=False)
        check('rigid_mode_restores_ground_collision',all(UsdPhysics.CollisionAPI(stage.GetPrimAtPath(
            '/World/Ground/'+t['id'])).GetCollisionEnabledAttr().Get() for t in tools.config['surfaces']))
        check('rigid_mode_removes_scm_visuals',not stage.GetPrimAtPath('/World/SCM'))
        if moving:
            check('rigid_mode_removes_moving_bands',not stage.GetPrimAtPath('/World/Robot/TrackMotion'))
            check('rigid_mode_restores_original_belt',stage.GetPrimAtPath('/World/Robot/Tracks').GetAttribute('visibility').Get()=='inherited')
        tools.runtime.timeline.play()
        await advance(1.)
        check('rigid_mode_still_supports_vehicle',tools.runtime.latest['left_normal_n']+tools.runtime.latest['right_normal_n']>100)
        check('baseline_scene_file_unchanged',hashlib.sha256((PROJECT/'scenes/five_surfaces.usda').read_bytes()).hexdigest()==
              'ddfacfcbd00994b3e5f1e0dd6136ed898f9874e1e7c55e6ac96bab711183774c')
        manifest=json.loads((PROJECT/'source_manifest.json').read_text())
        check('robot_assets_and_original_force_model_preserved',all(hashlib.sha256((PROJECT/p).read_bytes()).hexdigest()==h
              for p,h in manifest['unchanged_copies_sha256'].items()))
        report['passed']=True
        print('SCM_VALIDATION '+json.dumps(report),flush=True)
    except Exception:
        report['error']=traceback.format_exc()
        if tools and tools.runtime.latest:
            report['last_state']=tools.runtime.latest
        print(report['error'],flush=True)
    finally:
        faulthandler.cancel_dump_traceback_later()
        diagnostic.close()
        (output/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
        if tools:
            tools.runtime.zero()
            tools.runtime.timeline.stop()
            tools.runtime.shutdown()
        await omni.usd.get_context().close_stage_async()
        app.post_quit(0 if report['passed'] else 1)


VALIDATION_TASK=asyncio.ensure_future(validate())
