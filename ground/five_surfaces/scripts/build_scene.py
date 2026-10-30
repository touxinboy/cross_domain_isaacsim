"""Build five contiguous rigid tiles using unchanged L17 material values."""
import json
import math
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def vec(values):
    return '(' + ', '.join(f'{v:.9g}' for v in values) + ')'


def camera(name, eye, target):
    def unit(v):
        length = math.sqrt(sum(x*x for x in v))
        return [x/length for x in v]
    def cross(a, b):
        return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
    back = unit([a-b for a, b in zip(eye, target)])
    right = unit(cross([0, 0, 1], back))
    up = cross(back, right)
    matrix = ', '.join(vec(row) for row in [right+[0], up+[0], back+[0], list(eye)+[1]])
    return f'''
        def Camera "{name}"
        {{
            float focalLength = 28
            float horizontalAperture = 36
            float verticalAperture = 20.25
            float2 clippingRange = (0.03, 250)
            matrix4d xformOp:transform = ({matrix})
            uniform token[] xformOpOrder = ["xformOp:transform"]
        }}
'''


def marker(name, size, position, color=(0.94, 0.95, 0.93)):
    return f'''
        def Cube "{name}"
        {{
            double size = 1
            color3f[] primvars:displayColor = [{vec(color)}]
            double3 xformOp:translate = {vec(position)}
            double3 xformOp:scale = {vec(size)}
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        }}
'''


def build():
    config = json.loads((PROJECT/'configs/five_surfaces.json').read_text())
    materials, tiles, markers = [], [], []
    for tile in config['surfaces']:
        name = tile['id']
        materials.append(f'''
        def Material "{name}" (prepend apiSchemas = ["PhysicsMaterialAPI"])
        {{
            float physics:staticFriction = {tile['static_friction']}
            float physics:dynamicFriction = {tile['dynamic_friction']}
            float physics:restitution = {tile['restitution']}
            token outputs:surface.connect = </World/Materials/{name}/Surface.outputs:surface>
            def Shader "Surface"
            {{
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor = {vec(tile['color'])}
                float inputs:roughness = 0.9
                float inputs:metallic = 0
                token outputs:surface
            }}
        }}
''')
        a, b = tile['x_min_m'], tile['x_max_m']
        c, d = tile['y_min_m'], tile['y_max_m']
        bottom = config['top_z_m']-config['tile_thickness_m']
        top = config['top_z_m']
        points = [(a,c,bottom),(b,c,bottom),(b,d,bottom),(a,d,bottom),
                  (a,c,top),(b,c,top),(b,d,top),(a,d,top)]
        tiles.append(f'''
        def Mesh "{name}" (prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI", "MaterialBindingAPI"])
        {{
            bool physics:collisionEnabled = true
            token physics:approximation = "none"
            rel material:binding = </World/Materials/{name}>
            rel material:binding:physics = </World/Materials/{name}>
            point3f[] points = [{', '.join(vec(p) for p in points)}]
            int[] faceVertexCounts = [4, 4, 4, 4, 4, 4]
            int[] faceVertexIndices = [0,3,2,1, 4,5,6,7, 0,1,5,4, 1,2,6,5, 2,3,7,6, 3,0,4,7]
            uniform token subdivisionScheme = "none"
        }}
''')
        # Seven-segment numerals are visual geometry only, kept outside the driving line.
        x, y = (a+b)/2, 2.35
        segments = {'a':(0,.3,.38,.055),'b':(.19,.15,.055,.25),
                    'c':(.19,-.15,.055,.25),'d':(0,-.3,.38,.055),
                    'e':(-.19,-.15,.055,.25),'f':(-.19,.15,.055,.25),'g':(0,0,.38,.055)}
        digits = {1:'bc',2:'abged',3:'abgcd',4:'fgbc',5:'afgcd'}
        for segment in digits[tile['index']]:
            dx,dy,sx,sy = segments[segment]
            markers.append(marker(f"Number{tile['index']}_{segment}",(sx,sy,.008),(x+dx,y+dy,.007)))
        if tile['index'] > 1:
            markers.append(marker(f"Boundary{tile['index']}",(.035,6,.005),(a,0,.004)))
    markers.append(marker('StartLine',(.06,4.3,.005),(-8.5,0,.005),(.12,.75,.28)))
    markers.append(marker('FinishLine',(.06,4.3,.005),(8.5,0,.005),(.95,.25,.08)))
    cameras = camera('Overview',(10,-18,16),(0,0,0))
    cameras += camera('Robot',(-11.5,-4,3),(-8.3,0,.2))
    cameras += camera('Side',(0,-19,7),(0,0,.1))
    content = f'''#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Z"
    timeCodesPerSecond = 60
    startTimeCode = 0
    endTimeCode = 36000
    customLayerData = {{ string purpose = "L17 five-surface rigid baseline; no soil model changes" }}
)
def Xform "World"
{{
    def PhysicsScene "PhysicsScene" (prepend apiSchemas = ["PhysxSceneAPI"])
    {{
        vector3f physics:gravityDirection = (0, 0, -1)
        float physics:gravityMagnitude = 9.81
        uint physxScene:timeStepsPerSecond = 120
        bool physxScene:enableGPUDynamics = false
        token physxScene:broadphaseType = "MBP"
    }}
    def Scope "Materials" {{ {''.join(materials)} }}
    def Xform "Ground" {{ {''.join(tiles)} }}
    def Xform "Markings" {{ {''.join(markers)} }}
    def Xform "Robot" (prepend references = @../robots/tracked_twin_thruster_v1.usda@)
    {{
        double3 xformOp:translate = {vec(config['start_position_m'])}
        quatf xformOp:orient = (1, 0, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]
    }}
    def Xform "Cameras" {{ {cameras} }}
    def DomeLight "Sky"
    {{
        float inputs:intensity = 700
        color3f inputs:color = (0.82, 0.89, 1)
    }}
    def DistantLight "Sun"
    {{
        float inputs:intensity = 1800
        float inputs:angle = 1
        float3 xformOp:rotateXYZ = (-40, -25, -25)
        uniform token[] xformOpOrder = ["xformOp:rotateXYZ"]
    }}
}}
'''
    output = PROJECT/config['scene_file']
    output.parent.mkdir(exist_ok=True)
    output.write_text(content)
    print(output)


if __name__ == '__main__':
    build()
