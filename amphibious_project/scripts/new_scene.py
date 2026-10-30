"""Create the user's own empty USD stage, without loading anyone else's assets."""

from pathlib import Path

project = Path(__file__).resolve().parents[1]
target = project / "scenes/my_river.usda"
if target.exists():
    print(f"Preserving existing user scene: {target}")
else:
    target.parent.mkdir(parents=True, exist_ok=True)
    content = '''#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "World"
{
    def PhysicsScene "PhysicsScene"
    {
        vector3f physics:gravityDirection = (0, 0, -1)
        float physics:gravityMagnitude = 9.81
    }
}
'''
    with target.open("x", encoding="utf-8") as handle:
        handle.write(content)
    print(f"L17_NEW_EMPTY_SCENE: {target}")
