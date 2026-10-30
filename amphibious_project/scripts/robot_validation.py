"""Repeatable component experiments with measurements from the native PhysX body."""

import asyncio
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import time
import traceback

import numpy as np
import omni.kit.app
import omni.timeline
import omni.usd
from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file
from pxr import Gf

from robot_assets import project_path, create_validation_stage, set_camera
from robot_runtime import attach_robot_runtime, detach_robot_runtime


def phase_samples(history, name):
    return [row for row in history if row.get("phase")==name]


def displacement(rows):
    a,b=rows[0],rows[-1]
    return b["x_m"]-a["x_m"], b["y_m"]-a["y_m"]


def speed(row):
    return math.hypot(row["vx_m_s"],row["vy_m_s"])


class RobotValidation:
    def __init__(self,project,asset,cfg,ui=None):
        self.project=Path(project).resolve(strict=True)
        self.asset=asset;self.cfg=cfg;self.ui=ui
        stamp=datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output=project_path(project,"results/robot_v1/run_"+stamp)
        self.output.mkdir(parents=True,exist_ok=False)
        self.progress_file=project_path(project,".runtime/robot_validation_progress.json")
        self.summary={"started":datetime.now().astimezone().isoformat(),"run_directory":str(self.output),
                      "parameter_status":"initial uncalibrated prototype","cases":{},"passed":False,
                      "pose_integration":"native PhysX; no commanded trajectory or position overwrite during phases"}
        self.active_runtime=None
        self.main_scene=project_path(project,"scenes/river_robot_v1.usda")

    def progress(self,case,state,extra=None):
        value={"case":case,"state":state,"run_directory":str(self.output),"cases_finished":list(self.summary["cases"])}
        if extra:value.update(extra)
        self.progress_file.write_text(json.dumps(value,indent=2)+"\n")
        self.save_summary()

    def save_summary(self):
        (self.output/"summary.json").write_text(json.dumps(self.summary,indent=2)+"\n")

    async def wait(self,runtime,duration,phase,tracks=None,thrusters=None,mode="HYBRID"):
        runtime.phase=phase
        runtime.set_commands(mode,tracks if tracks is not None else [0,0],thrusters if thrusters is not None else [0,0])
        start=runtime.sim_time;deadline=time.monotonic()+max(35,duration*5)
        app=omni.kit.app.get_app()
        while runtime.sim_time-start<duration:
            await app.next_update_async()
            if runtime.error:raise RuntimeError(runtime.error)
            if time.monotonic()>deadline:raise RuntimeError("Physics experiment stalled during "+phase)
        if not runtime.latest:raise RuntimeError("No native physics state was recorded.")

    async def open_case(self,kind,position,roll=0,pitch=0):
        timeline=omni.timeline.get_timeline_interface()
        timeline.stop();detach_robot_runtime()
        for _ in range(3):await omni.kit.app.get_app().next_update_async()
        file=create_validation_stage(self.project,self.cfg,self.asset,kind)
        ok,error=await omni.usd.get_context().open_stage_async(str(file))
        if not ok:raise RuntimeError(error)
        stage=omni.usd.get_context().get_stage();root=stage.GetPrimAtPath("/World/Robot")
        root.GetAttribute("xformOp:translate").Set(Gf.Vec3d(*position))
        rotation=Gf.Rotation(Gf.Vec3d(0,1,0),pitch)*Gf.Rotation(Gf.Vec3d(1,0,0),roll)
        q=rotation.GetQuat();root.GetAttribute("xformOp:orient").Set(Gf.Quatf(q.GetReal(),Gf.Vec3f(q.GetImaginary())))
        get_active_viewport().set_active_camera("/World/Cameras/Overview")
        runtime=attach_robot_runtime(self.project,kind)
        self.active_runtime=runtime
        timeline.set_current_time(0);timeline.play()
        return runtime

    async def screenshot(self,runtime,name):
        timeline=omni.timeline.get_timeline_interface();timeline.pause()
        state=runtime.latest
        if not state:raise RuntimeError("No measured pose for the screenshot.")
        x,y,z=state["x_m"],state["y_m"],state["z_m"]
        stage=omni.usd.get_context().get_stage()
        set_camera(stage,"/World/Cameras/Robot",(x-1.9,y-2.2,z+1.15),(x,y,z+.05),28)
        viewport=get_active_viewport();viewport.set_active_camera("/World/Cameras/Robot")
        for _ in range(35):await omni.kit.app.get_app().next_update_async()
        job=capture_viewport_to_file(viewport,str(self.output/(name+".png")))
        await job.wait_for_result()

    async def land(self):
        self.progress("land","running")
        r=await self.open_case("land",(-4,0,.75))
        await self.wait(r,.18,"airborne_tracks",[.8,.8],mode="LAND")
        await self.wait(r,.9,"settle",mode="LAND")
        await self.wait(r,4.,"straight",[.7,.7],mode="LAND")
        await self.wait(r,1.,"brake",mode="LAND")
        await self.wait(r,3.,"turn",[-.3,.3],mode="LAND")
        await self.wait(r,1.5,"stop",mode="LAND")
        r.zero();r.save_csv(self.output/"land.csv")
        await self.screenshot(r,"land")
        straight=phase_samples(r.history,"straight");turn=phase_samples(r.history,"turn");air=phase_samples(r.history,"airborne_tracks")
        dx,dy=displacement(straight)
        headings=np.unwrap([x["yaw_rad"] for x in turn]);turn_angle=math.degrees(float(headings[-1]-headings[0]))
        airborne_dx=displacement(air)[0]
        last=phase_samples(r.history,"stop")[-1]
        mean_normal=float(np.mean([x["left_normal_n"]+x["right_normal_n"] for x in straight[-120:]]))
        checks={"straight_advance_over_2m":dx>2.,"straight_lateral_drift_under_0_15m":abs(dy)<.15,
                "left_turn_over_40deg":turn_angle>40.,"stopped_under_0_08m_s":speed(last)<.08,
                "no_track_drive_in_air":abs(airborne_dx)<.005 and max(x["left_normal_n"]+x["right_normal_n"] for x in air)<1.,
                "normal_load_consistent_with_weight":abs(mean_normal-self.cfg["mass_kg"]*9.81)<.18*self.cfg["mass_kg"]*9.81}
        checks={key:bool(value) for key,value in checks.items()}
        return {"passed":all(checks.values()),"checks":checks,"straight_displacement_m":dx,"lateral_drift_m":dy,
                "turn_angle_deg":turn_angle,"final_speed_m_s":speed(last),"mean_track_normal_load_n":mean_normal,
                "airborne_displacement_m":airborne_dx,"samples":len(r.history),"csv":"land.csv","image":"land.png"}

    async def floating(self):
        self.progress("floating","running")
        negative=await self.open_case("water",(-4,0,.45))
        negative.buoyancy_enabled=False
        await self.wait(negative,.48,"buoyancy_disabled",mode="WATER")
        negative.save_csv(self.output/"water_without_buoyancy.csv")
        negative_z=negative.latest["z_m"]
        negative_contact=max(x["left_normal_n"]+x["right_normal_n"] for x in negative.history)
        r=await self.open_case("water",(-4,0,.45),roll=6,pitch=-4)
        await self.wait(r,15.,"float_settle",mode="WATER")
        r.zero();r.save_csv(self.output/"floating.csv")
        await self.screenshot(r,"floating")
        tail=[x for x in r.history if x["time_s"]>r.sim_time-3.]
        z=float(np.mean([x["z_m"] for x in tail]));vz=max(abs(x["vz_m_s"]) for x in tail)
        roll=max(abs(x["roll_deg"]) for x in tail);pitch=max(abs(x["pitch_deg"]) for x in tail)
        buoy=float(np.mean([x["buoyancy_n"] for x in tail]));weight=self.cfg["mass_kg"]*9.81
        contacts=max(x["left_normal_n"]+x["right_normal_n"] for x in r.history)
        checks={"height_near_archimedes_prediction":abs(z-r.model.equilibrium_z)<.025,
                "vertical_speed_under_0_04m_s":vz<.04,"roll_pitch_under_3deg":max(roll,pitch)<3,
                "buoyancy_balances_weight":abs(buoy-weight)<.05*weight,"no_riverbed_support":contacts<1,
                "disabled_buoyancy_sinks_through_visual_surface":negative_z<-.25 and negative_contact<1}
        checks={key:bool(value) for key,value in checks.items()}
        return {"passed":all(checks.values()),"checks":checks,"mean_body_origin_z_m":z,
                "predicted_body_origin_z_m":r.model.equilibrium_z,"max_abs_vertical_speed_m_s":vz,
                "max_abs_roll_deg":roll,"max_abs_pitch_deg":pitch,"mean_buoyancy_n":buoy,"weight_n":weight,
                "max_ground_contact_n":contacts,"without_buoyancy_final_z_m":negative_z,
                "samples":len(r.history),"csv":"floating.csv","image":"floating.png"}

    async def propulsion(self):
        self.progress("propulsion","running")
        dry=await self.open_case("water",(-4,0,.75))
        await self.wait(dry,.18,"dry_thrusters",thrusters=[1,1],mode="WATER")
        dry.save_csv(self.output/"dry_thrusters.csv")
        dry_dx=displacement(dry.history)[0]
        dry_thrust=max(abs(x["left_thrust_n"])+abs(x["right_thrust_n"]) for x in dry.history)
        r=await self.open_case("water",(-4,0,.03544))
        await self.wait(r,2.,"float_settle",mode="WATER")
        await self.wait(r,4.,"straight",thrusters=[.65,.65],mode="WATER")
        await self.wait(r,3.,"coast",mode="WATER")
        await self.wait(r,4.,"differential_turn",thrusters=[.10,.80],mode="WATER")
        await self.wait(r,4.,"stop",mode="WATER")
        r.zero();r.save_csv(self.output/"propulsion.csv")
        await self.screenshot(r,"propulsion")
        straight=phase_samples(r.history,"straight");coast=phase_samples(r.history,"coast");turn=phase_samples(r.history,"differential_turn")
        dx,dy=displacement(straight);heading=np.unwrap([x["yaw_rad"] for x in turn])
        angle=math.degrees(float(heading[-1]-heading[0]));end_speed=speed(straight[-1]);coast_speed=speed(coast[-1])
        contact=max(x["left_normal_n"]+x["right_normal_n"] for x in r.history)
        checks={"water_advance_over_2m":dx>2.,"water_lateral_drift_under_0_20m":abs(dy)<.20,
                "forward_speed_over_0_6m_s":end_speed>.6,"differential_left_turn_over_35deg":angle>35.,
                "coast_speed_reduced_by_over_50percent":coast_speed<end_speed*.5,
                "no_ground_contact":contact<1.,"dry_thrusters_generate_no_drive":dry_thrust<.01 and abs(dry_dx)<.005}
        checks={key:bool(value) for key,value in checks.items()}
        return {"passed":all(checks.values()),"checks":checks,"straight_displacement_m":dx,"lateral_drift_m":dy,
                "end_straight_speed_m_s":end_speed,"end_coast_speed_m_s":coast_speed,"turn_angle_deg":angle,
                "max_ground_contact_n":contact,"dry_thruster_displacement_m":dry_dx,"dry_thruster_effective_force_n":dry_thrust,
                "samples":len(r.history),"csv":"propulsion.csv","image":"propulsion.png"}

    async def run(self):
        self.progress("suite","starting")
        timeline=omni.timeline.get_timeline_interface()
        try:
            for name,method in [("land",self.land),("floating",self.floating),("propulsion",self.propulsion)]:
                result=await method();self.summary["cases"][name]=result
                self.progress(name,"passed" if result["passed"] else "failed")
            self.summary["passed"]=all(x["passed"] for x in self.summary["cases"].values())
            self.summary["finished"]=datetime.now().astimezone().isoformat()
            self.progress("suite","passed" if self.summary["passed"] else "failed")
            print("L17_ROBOT_SUITE_RESULT "+json.dumps(self.summary),flush=True)
        except Exception:
            self.summary["error"]=traceback.format_exc()
            self.progress("suite","error")
            print("L17_ROBOT_VALIDATION_ERROR\n"+self.summary["error"],flush=True)
        finally:
            timeline.stop();detach_robot_runtime()
            for _ in range(3):await omni.kit.app.get_app().next_update_async()
            ok,error=await omni.usd.get_context().open_stage_async(str(self.main_scene))
            if ok:
                attach_robot_runtime(self.project,"river")
                if self.ui:self.ui.restore_river()
            self.save_summary()
        return self.summary
