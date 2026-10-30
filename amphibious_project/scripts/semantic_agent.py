"""Qwen/MSGNav target evidence and observed-map route planning, outside Kit.

This module has no stage, river geometry, target configuration or robot pose
setter. Navigation input is the instruction, RGB-D and measured self state.
"""
import json
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw

from depth_localization import DepthFrame, localize_surface
from ground_target_api import ground
from msgnav_bridge import MSGNavDecision
from navigation_io import own_path, read_json, write_json
from online_terrain_map import OnlineTerrainMap
from resolve_navigation_depth import selected_frame


def parse_json_response(raw):
    text=raw.strip()
    if text.startswith('```') and text.endswith('```'):text='\n'.join(text.splitlines()[1:-1])
    value=json.loads(text)
    if not isinstance(value,dict):raise ValueError('Expected a JSON object from semantic verifier')
    return value


def refine_structural_pixel(frame,image_path,grounding,max_shift_px=24.):
    """Refine the VLM pixel only to nearby visible orange/brown steel evidence."""
    if not grounding.get('visible') or not grounding.get('format_valid'):return grounding,None
    with Image.open(image_path) as source:rgb=np.asarray(source.convert('RGB')).astype(float)
    h,w=rgb.shape[:2];box=grounding['bbox_xyxy'];u,v=grounding['point_uv']
    yy,xx=np.mgrid[:h,:w]
    appearance=(rgb[:,:,0]>rgb[:,:,1]*1.14)&(rgb[:,:,1]>rgb[:,:,2]*1.12)&(rgb[:,:,0]>28)
    appearance&=(xx>=box[0])&(xx<=box[2])&(yy>=box[1])&(yy<=box[3])
    valid=appearance&np.isfinite(frame.depth_m)&(frame.depth_m>.08)&(frame.depth_m<70.)
    distance=(xx-u)**2+(yy-v)**2
    candidates=valid&(distance<=max_shift_px**2)
    if not candidates.any():
        return {**grounding,'visible':False,'refinement_error':'no_nearby_opaque_structure_color_support'},None
    score=np.where(candidates,distance,np.inf);index=np.unravel_index(np.argmin(score),score.shape)
    refined={**grounding,'original_point_uv':list(grounding['point_uv']),'point_uv':[int(index[1]),int(index[0])],
             'pixel_refinement':'nearest_visible_orange_brown_structure_pixel_within_24px',
             'pixel_shift_px':float(math.sqrt(distance[index]))}
    return refined,appearance


def verify_flat_top_identity(client,image_path,grounding,output):
    with Image.open(image_path) as original:
        image=original.convert('RGB');w,h=image.size
    left,top,right,bottom=grounding['bbox_xyxy'];bw=right-left;bh=bottom-top
    bounds=(max(0,int(left-.18*bw)),max(0,int(top-.25*bh)),min(w,int(right+.18*bw)+1),min(h,int(bottom+.12*bh)+1))
    crop=image.crop(bounds);scale=max(1.,360/max(crop.size))
    crop=crop.resize((max(1,round(crop.width*scale)),max(1,round(crop.height*scale))),Image.Resampling.LANCZOS)
    path=Path(output);crop.save(path)
    prompt=('Classify the HIGHEST structural level of the steel tower in this image. '
            'Return only JSON: {"highest_top_shape":"flat_platform" or "pointed_apex" or "unclear",'
            '"upper_structure_fully_visible":true or false,"evidence":"brief visual evidence"}. '
            'A triangular or pyramidal steel lattice rising above a lower deck means pointed_apex. '
            'Use flat_platform only when a horizontal rectangular deck with railings is the highest level. '
            'If the upper structure is cut off or ambiguous, use unclear. Judge only visible geometry.')
    raw=client.describe([path],prompt,max_tokens=256)
    try:
        value=parse_json_response(raw)
        accepted=value.get('highest_top_shape')=='flat_platform' and value.get('upper_structure_fully_visible') is True
        return {'accepted':accepted,'raw_response':raw,'parsed':value,'crop_bounds_xyxy':list(bounds)}
    except (ValueError,TypeError) as error:return {'accepted':False,'raw_response':raw,'error':str(error)}


class SemanticPlanner:
    def __init__(self,project,client):
        self.project=Path(project).resolve();self.client=client;self.missions={}

    def update(self,request):
        observation=read_json(own_path(self.project,request['observation_file']))
        state=request['state'];step=int(request['step']);run_id=request['run_id']
        if observation.get('source')!='perception':raise ValueError('Live semantics requires real robot perception')
        epoch=observation['controller_epoch']
        if epoch!=request['controller_epoch']:raise ValueError('Semantic runtime epoch mismatch')
        instruction=request['instruction']
        if not isinstance(instruction,str) or len(instruction)<8:raise ValueError('Use a target appearance description')
        root=own_path(self.project,request['output_relative'])
        output=root/'planner'/f'cycle_{step:04d}'
        output.mkdir(parents=True,exist_ok=False)
        mission=self.missions.get(run_id)
        if mission is None:
            mission={'epoch':epoch,'instruction':instruction,'map':OnlineTerrainMap(observation['anchor_pose_xyyaw'][:2]),
                     'targets':[],'last_semantic_step':-100,'visited':[],'semantic_calls':0,'last_evidence':None}
            self.missions[run_id]=mission
        if mission['epoch']!=epoch or mission['instruction']!=instruction:raise ValueError('Do not change epoch or instruction within one mission')
        mapper=mission['map']
        for relative in observation['egocentric_rgbd_frames']+observation.get('mapping_rgbd_frames',[]):
            mapper.integrate(own_path(self.project,relative))
        mapper.update_state(state)
        xy=[float(state['x_m']),float(state['y_m'])]
        old_target=np.median(mission['targets'][-5:],axis=0).tolist() if mission['targets'] else None
        near=old_target is not None and math.dist(xy,old_target)<4.5
        refresh=not mission['targets'] or step-mission['last_semantic_step']>=8 or near
        semantics=None
        if refresh:
            semantics=self._update_semantics(observation,instruction,mission,output)
            mission['last_semantic_step']=step;mission['semantic_calls']+=1
        target=np.median(mission['targets'][-5:],axis=0).tolist() if mission['targets'] else None
        candidate=False
        if near and semantics and semantics.get('surface',{}).get('accepted'):
            distance=math.dist(xy,semantics['surface']['world_point_m'][:2])
            supported=float(state['left_normal_n'])+float(state['right_normal_n'])>.2*48*9.81
            if 1.6<=distance<=3.2 and supported:
                prompt=('Task: '+instruction+'\nInspect this current robot camera image. Return only JSON with boolean '
                        'target_visible, matches_target, flat_rectangular_upper_platform, and string evidence. '
                        'A successful match needs the flat upper platform and steel lattice legs to be visible. '
                        'Make the judgement only from the image.')
                raw=self.client.describe([own_path(self.project,semantics['selected_rgb'])],prompt,max_tokens=256)
                confirmation=parse_json_response(raw)
                candidate=all(confirmation.get(k) is True for k in ['target_visible','matches_target','flat_rectangular_upper_platform'])
                write_json(output/'final_visual_confirmation.json',{'raw':raw,'parsed':confirmation,'candidate':candidate})
        route=None;scan_reason=None
        if not candidate:
            try:route=mapper.plan(xy,target,mission['visited'],min_standoff=1.8,max_standoff=3.0)
            except ValueError as error:scan_reason=str(error)
        mapper.save(output/'map',route,target_xy=target,robot_xy=xy)
        mission['visited'].append(xy)
        result={'source':'MSGNav_Qwen_RGBD_observed_planner','step':step,'target_instruction':instruction,
                'target_surface_xy_m':target,'target_distance_m':math.dist(xy,target) if target is not None else None,
                'target_coordinate_source':'observed_Qwen_grounding_plus_calibrated_RGBD',
                'route':route,'need_scan':route is None and not candidate,'scan_reason':scan_reason,
                'semantic_arrival_candidate':candidate,'grid_file':(output/'map/grid.json').relative_to(self.project).as_posix(),
                'semantic_update':semantics,'semantic_refresh_count':mission['semantic_calls'],
                'hidden_scene_geometry_used':False,'manual_target_coordinate_used':False,
                'perception_mapping_frames':len(mapper.seen_frames),'footprint_radius_m':mapper.footprint_radius}
        write_json(output/'plan.json',result)
        return result

    def _update_semantics(self,observation,instruction,mission,output):
        decision=None;decision_error=None
        try:
            decision=MSGNavDecision(self.client).decide(observation,self.project)
            write_json(output/'msgnav_decision.json',decision)
        except Exception as error:
            # This explicit fallback still requires a fresh visual detection;
            # it never substitutes a waypoint or a hidden scene coordinate.
            decision_error=type(error).__name__+': '+str(error)
            write_json(output/'msgnav_decision_error.json',{'error':decision_error})
        selected=[]
        if decision and decision.get('choice_type')=='image':
            try:
                rgb,frame=selected_frame(observation,decision,self.project)
                selected.append((rgb,frame))
            except (KeyError,ValueError) as error:decision_error=str(error)
        for rgb_relative,frame_relative in zip(observation['egocentric_images'],observation['egocentric_rgbd_frames']):
            rgb=own_path(self.project,rgb_relative)
            if all(str(rgb)!=str(old[0]) for old in selected):selected.append((rgb,DepthFrame.load(own_path(self.project,frame_relative))))
        attempts=[]
        for index,(rgb,frame) in enumerate(selected):
            raw_grounding=ground(self.client,rgb,instruction)
            identity=None
            if raw_grounding.get('visible') and raw_grounding.get('format_valid'):
                identity=verify_flat_top_identity(self.client,rgb,raw_grounding,output/f'candidate_{index:02d}_crop.png')
                write_json(output/f'candidate_{index:02d}_identity.json',identity)
                if not identity['accepted']:
                    attempts.append({'selected_rgb':rgb.relative_to(self.project).as_posix(),
                                     'raw_grounding':raw_grounding,'identity':identity,'rejected_before_depth_goal':True})
                    continue
            refined,mask=refine_structural_pixel(frame,rgb,raw_grounding)
            surface=localize_surface(frame,refined,min_samples=3,mask=mask)
            evidence={'selected_rgb':rgb.relative_to(self.project).as_posix(),'selected_frame_id':frame.metadata['frame_id'],
                      'raw_grounding':raw_grounding,'grounding':refined,'surface':surface,'identity':identity}
            attempts.append(evidence);write_json(output/f'grounding_{index:02d}.json',evidence)
            if surface['accepted']:
                xy=surface['world_point_m'][:2]
                if mission['targets'] and math.dist(xy,np.median(mission['targets'][-5:],axis=0))>4.:
                    evidence['surface']['accepted']=False;evidence['surface']['reason']='inconsistent_with_observed_target_memory'
                    continue
                mission['targets'].append(xy);mission['last_evidence']=evidence
                with Image.open(rgb) as original:picture=original.convert('RGB')
                draw=ImageDraw.Draw(picture);box=refined['bbox_xyxy'];u,v=refined['point_uv']
                draw.rectangle(tuple(box),outline=(255,80,20),width=3);draw.ellipse((u-4,v-4,u+4,v+4),fill=(255,0,230))
                picture.save(output/'target_grounding.png')
                return {**evidence,'msgnav_decision':decision,'msgnav_error':decision_error,'grounding_attempt_count':len(attempts)}
        return {'surface':{'accepted':False},'msgnav_decision':decision,'msgnav_error':decision_error,
                'grounding_attempts':attempts,'target_not_currently_visible':True}
