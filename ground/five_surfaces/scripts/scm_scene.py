"""Session-layer SCM visualization and explicit collision ownership."""
import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt


def configure_ground(stage, ground, enabled, settings):
    # Never save these overrides to the original baseline USD.
    with Usd.EditContext(stage,stage.GetSessionLayer()):
        for tile in ground['surfaces']:
            prim = stage.GetPrimAtPath('/World/Ground/'+tile['id'])
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(not enabled)
            UsdGeom.Imageable(prim).GetVisibilityAttr().Set('invisible' if enabled else 'inherited')
        if stage.GetPrimAtPath('/World/SCM'):
            stage.RemovePrim('/World/SCM')
        if enabled:
            UsdGeom.Xform.Define(stage,'/World/SCM')
            tiles = ground['surfaces']
            xmin,xmax = min(t['x_min_m'] for t in tiles),max(t['x_max_m'] for t in tiles)
            ymin,ymax = min(t['y_min_m'] for t in tiles),max(t['y_max_m'] for t in tiles)
            bottom = ground['top_z_m']-settings['bedrock_depth_m']
            bed = UsdGeom.Cube.Define(stage,'/World/SCM/Bedrock')
            bed.CreateSizeAttr(1)
            bed.AddTranslateOp().Set(Gf.Vec3d((xmin+xmax)/2,(ymin+ymax)/2,bottom-.05))
            bed.AddScaleOp().Set(Gf.Vec3f(xmax-xmin,ymax-ymin,.1))
            UsdPhysics.CollisionAPI.Apply(bed.GetPrim())
            bed.MakeInvisible()


class SCMTerrainView:
    def __init__(self,stage,terrain):
        self.stage,self.terrain = stage,terrain
        self.meshes = []
        self.next_update_s = 0.
        self.last_update_ms = 0.
        spacing = terrain.settings['visual_spacing_m']
        with Usd.EditContext(stage,stage.GetSessionLayer()):
            for tile in terrain.ground['surfaces']:
                x = np.linspace(tile['x_min_m'],tile['x_max_m'],int(round((tile['x_max_m']-tile['x_min_m'])/spacing))+1)
                y = np.linspace(tile['y_min_m'],tile['y_max_m'],int(round((tile['y_max_m']-tile['y_min_m'])/spacing))+1)
                xx,yy = np.meshgrid(x,y,indexing='ij')
                points = np.column_stack((xx.ravel(),yy.ravel(),np.full(xx.size,terrain.h0))).astype(np.float32)
                # Query just inside each tile at its outer border.
                query = points[:,:2].astype(float)
                query[:,0] = np.clip(query[:,0],tile['x_min_m']+1e-5,tile['x_max_m']-1e-5)
                query[:,1] = np.clip(query[:,1],tile['y_min_m']+1e-5,tile['y_max_m']-1e-5)
                a = (np.arange(len(x)-1)[:,None]*len(y)+np.arange(len(y)-1)[None,:]).ravel()
                faces = np.column_stack((a,a+len(y),a+len(y)+1,a+1)).astype(np.int32)
                mesh = UsdGeom.Mesh.Define(stage,'/World/SCM/'+tile['id'])
                mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points))
                mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(a),4,dtype=np.int32)))
                mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.ravel()))
                mesh.CreateSubdivisionSchemeAttr('none')
                mesh.CreateDoubleSidedAttr(True)
                # Disturbed soil is darker as well as geometrically depressed.
                # A primvar reader lets the same soil state drive both effects.
                material = UsdShade.Material.Define(stage,'/World/SCM/Looks/'+tile['id'])
                shader = UsdShade.Shader.Define(stage,material.GetPath().AppendChild('Surface'))
                shader.CreateIdAttr('UsdPreviewSurface')
                from pxr import Sdf
                shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.95)
                reader = UsdShade.Shader.Define(stage,material.GetPath().AppendChild('Color'))
                reader.CreateIdAttr('UsdPrimvarReader_float3')
                reader.CreateInput('varname',Sdf.ValueTypeNames.Token).Set('displayColor')
                reader.CreateOutput('result',Sdf.ValueTypeNames.Float3)
                shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).ConnectToSource(reader.ConnectableAPI(),'result')
                shader.CreateOutput('surface',Sdf.ValueTypeNames.Token)
                material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
                UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(
                    material)
                color = mesh.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex)
                color.Set(Vt.Vec3fArray.FromNumpy(np.tile(np.asarray(tile['color'],dtype=np.float32),(len(points),1))))
                self.meshes.append((mesh,points,query,color,np.asarray(tile['color'])))
        # A newly built view must reconstruct every material, including old ruts
        # outside the currently active contact patch after a vehicle-only reset.
        terrain.dirty_materials.update(range(len(self.meshes)))
        self.update(0.,force=True)

    def update(self,sim_time,force=False):
        import time
        if not force and sim_time < self.next_update_s:
            return
        started = time.perf_counter()
        self.next_update_s = sim_time+1/self.terrain.settings['visual_update_hz']
        with Usd.EditContext(self.stage,self.stage.GetSessionLayer()):
            for i in self.terrain.dirty_materials:
                mesh,points,query,color,base = self.meshes[i]
                points[:,2] = self.terrain.heights(query)
                mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points))
                depth = self.terrain.h0-self.terrain.heights(query,unloaded=True)
                colors = (base*(1-.3*np.clip(depth/.025,0,1))[:,None]).astype(np.float32)
                color.Set(Vt.Vec3fArray.FromNumpy(colors))
        self.terrain.dirty_materials.clear()
        self.last_update_ms = (time.perf_counter()-started)*1000
