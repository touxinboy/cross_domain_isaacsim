"""Isaac XY/Z-up to OmniNav right/forward conventions, in metres and radians."""
import math


def finite_vector(value, length, label):
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(label + " has the wrong shape")
    converted = [float(x) for x in value]
    if not all(math.isfinite(x) for x in converted):
        raise ValueError(label + " must be finite")
    return converted


def isaac_world_to_habitat(point):
    x, y, z = finite_vector(point, 3, "world point")
    return [-y, z, -x]


def omni_local_point(world_point, anchor_pose):
    """Planar anchor [world x, world y, yaw]; positive yaw is left in Isaac."""
    x, y = finite_vector(world_point, 2, "world point")
    ax, ay, yaw = finite_vector(anchor_pose, 3, "anchor pose")
    dx, dy = x - ax, y - ay
    forward = math.cos(yaw) * dx + math.sin(yaw) * dy
    left = -math.sin(yaw) * dx + math.cos(yaw) * dy
    # Upstream transform_to_local_frame rounds local XYZ to 2 decimals.
    return [round(-left, 2), round(forward, 2)]


def pointgoal_positions(history_xy, current_xy, target_xy, anchor_pose):
    if len(history_xy) != 4:
        raise ValueError("OmniNav PointGoal requires four captured history frames and poses")
    points = list(history_xy) + [current_xy, target_xy]
    return [[x / .3 for x in omni_local_point(point, anchor_pose)] for point in points]


def native_head_first_step(waypoints, angles, anchor_pose):
    """Follow upstream first-step radius/angle execution, without pose writes.

    The head's Cartesian waypoint norm sets distance; the separately predicted
    angle sets direction. Preserve both raw outputs for diagnosis: they are not
    automatically the same as a five-point Cartesian trajectory.
    """
    if not isinstance(waypoints, list) or len(waypoints) != 5:
        raise ValueError("Expected five waypoints from the dedicated OmniNav head")
    points = [finite_vector(point, 2, "head waypoint") for point in waypoints]
    turns = finite_vector(angles, 5, "head angles")
    ax, ay, yaw = finite_vector(anchor_pose, 3, "capture anchor")
    radius = math.hypot(*points[0]) * .3
    return [ax + radius * math.cos(yaw + turns[0]), ay + radius * math.sin(yaw + turns[0])]
