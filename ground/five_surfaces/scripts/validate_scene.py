"""Run inside Isaac Sim: verify geometry/materials and drive over all five tiles."""
import asyncio
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

import omni.kit.app
import omni.usd
from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
from pxr import UsdGeom, UsdPhysics

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'scripts'))


async def capture(path):
    app=omni.kit.app.get_app()
    for _ in range(20):
        await app.next_update_async()
    job=capture_viewport_to_file(get_active_viewport(),str(path))
    await job.wait_for_result()
    deadline=time.monotonic()+20
    while time.monotonic()<deadline:
        if path.is_file() and path.stat().st_size>1000:
            with path.open('rb') as handle:
                handle.seek(-12,2)
                if handle.read()==b'\x00\x00\x00\x00IEND\xaeB`\x82':
                    return
        await app.next_update_async()
    raise RuntimeError('Screenshot did not finish writing')


async def validate():
    app=omni.kit.app.get_app()
    output=PROJECT/'results/validation'
    output.mkdir(parents=True,exist_ok=True)
    report={'passed':False,'tested_project':str(PROJECT)}
    tools=None
    try:
        from open_stage import open_scene
        tools=await open_scene(scm_enabled=False)
        stage=omni.usd.get_context().get_stage()
        config=tools.config
        checks={}
        materials=json.loads((PROJECT/'configs/river_forest.json').read_text())['materials']
        manifest=json.loads((PROJECT/'source_manifest.json').read_text())
        checks['original_robot_and_dynamics_unchanged']=all(
            hashlib.sha256((PROJECT/path).read_bytes()).hexdigest()==digest
            for path,digest in manifest['unchanged_copies_sha256'].items())
        checks['exactly_five_ground_tiles']=len(stage.GetPrimAtPath('/World/Ground').GetChildren())==5
        extents=[]
        for tile in config['surfaces']:
            prim=stage.GetPrimAtPath('/World/Ground/'+tile['id'])
            points=UsdGeom.Mesh(prim).GetPointsAttr().Get()
            xs,ys,zs=zip(*points)
            extents.append((min(xs),max(xs)))
            material=UsdPhysics.MaterialAPI(stage.GetPrimAtPath('/World/Materials/'+tile['id']))
            source=materials[tile['id']]
            checks[tile['id']+'_geometry_and_collision']=(prim.HasAPI(UsdPhysics.CollisionAPI)
                and abs(max(zs))<1e-6 and abs(max(xs)-min(xs)-4)<1e-6
                and abs(max(ys)-min(ys)-6)<1e-6)
            checks[tile['id']+'_original_friction']=(
                abs(material.GetStaticFrictionAttr().Get()-source['static_friction'])<1e-6
                and abs(material.GetDynamicFrictionAttr().Get()-source['dynamic_friction'])<1e-6
                and material.GetRestitutionAttr().Get()==0)
        checks['no_gaps_or_overlaps']=all(abs(a[1]-b[0])<1e-6 for a,b in zip(extents,extents[1:]))
        checks['robot_mass_48_kg']=UsdPhysics.MassAPI(stage.GetPrimAtPath('/World/Robot')).GetMassAttr().Get()==48.
        checks['physics_hz_120']=stage.GetPrimAtPath('/World/PhysicsScene').GetAttribute('physxScene:timeStepsPerSecond').Get()==120
        report['checks']=checks
        if not all(checks.values()):
            raise RuntimeError('Structural verification failed: '+str(checks))
        tools.runtime.timeline.play()
        while tools.runtime.sim_time<1.:
            if tools.runtime.error:
                raise RuntimeError(tools.runtime.error)
            await app.next_update_async()
        tools.runtime.timeline.pause()
        await capture(output/'overview.png')
        await tools.run_route()
        deadline=time.monotonic()+300
        while tools.runtime.running_route:
            if tools.runtime.error:
                raise RuntimeError(tools.runtime.error)
            if time.monotonic()>deadline:
                raise RuntimeError('Traversal wall-clock timeout')
            await app.next_update_async()
        runtime=tools.runtime
        expected={tile['id'] for tile in config['surfaces']}
        checks['robot_crossed_all_five_with_contact']=runtime.visited==expected
        checks['route_completed_and_stopped']=runtime.route_reason=='completed'
        checks['robot_remained_on_strip']=max(abs(row['y_m']) for row in runtime.history)<2.5
        checks['zero_water_forces']=all(abs(row['buoyancy_n'])+abs(row['left_thrust_n'])+abs(row['right_thrust_n'])<1e-6 for row in runtime.history)
        for tile in config['surfaces']:
            interior=[row for row in runtime.history if tile['x_min_m']+.6<row['x_m']<tile['x_max_m']-.6]
            checks[tile['id']+'_interior_support']=any(row['left_normal_n']+row['right_normal_n']>100 for row in interior)
        runtime.save_csv(output/'physics.csv')
        await capture(output/'finish.png')
        report.update(passed=all(checks.values()),route_reason=runtime.route_reason,
                      visited_surfaces=sorted(runtime.visited),physics_samples=len(runtime.history),
                      simulated_duration_s=runtime.sim_time,
                      final_position_m=[runtime.latest[k] for k in ['x_m','y_m','z_m']],
                      source_scene_sha256=hashlib.sha256((PROJECT/config['scene_file']).read_bytes()).hexdigest(),
                      terrain_physics_modified=False,soil_deformation_enabled=False)
        print('FIVE_SURFACES_VALIDATION: '+json.dumps(report),flush=True)
    except Exception:
        report['error']=traceback.format_exc()
        print(report['error'],flush=True)
    finally:
        (output/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
        if tools:
            tools.runtime.zero()
            tools.runtime.timeline.stop()
            tools.runtime.shutdown()
        await omni.usd.get_context().close_stage_async()
        app.post_quit(0 if report['passed'] else 1)


VALIDATION_TASK=asyncio.ensure_future(validate())
