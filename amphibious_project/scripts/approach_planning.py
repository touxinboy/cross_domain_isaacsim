"""Conservative arrival-pose selection on an externally observed XY grid.

This is not a mapper. Unknown cells are blocked. The input's land/water cells
must already have passed the robot's terrain/medium capability checks.
"""
import argparse
import heapq
import math
from pathlib import Path

import numpy as np

from navigation_io import file_sha, own_path, read_json, write_json


class ObservedGrid:
    def __init__(self, state, origin_xy_m, resolution_m, footprint_radius_m=.75):
        self.state = np.asarray(state)
        self.origin = np.asarray(origin_xy_m, dtype=float)
        self.resolution = float(resolution_m)
        if (self.state.ndim != 2 or min(self.state.shape) < 3 or not np.isin(self.state, [-1, 0, 1, 2]).all()
                or self.origin.shape != (2,) or not np.isfinite(self.origin).all()
                or not math.isfinite(self.resolution) or self.resolution <= 0
                or not math.isfinite(footprint_radius_m) or footprint_radius_m <= 0):
            raise ValueError("Invalid observed traversability grid")
        # -1 unknown; 0 blocked; 1 observed traversable land; 2 traversable water.
        free = np.isin(self.state, [1, 2])
        radius = footprint_radius_m + self.resolution * math.sqrt(2) / 2
        cells = math.ceil(radius / self.resolution)
        padded = np.pad(free, cells, constant_values=False)
        self.safe = free.copy()
        h, w = free.shape
        for dy in range(-cells, cells+1):
            for dx in range(-cells, cells+1):
                if math.hypot(dx, dy)*self.resolution <= radius:
                    self.safe &= padded[cells+dy:cells+dy+h, cells+dx:cells+dx+w]

    @classmethod
    def load(cls, metadata_path, footprint_radius_m=.75):
        path = Path(metadata_path).resolve()
        meta = read_json(path)
        if meta.get("schema") != "L17.traversability.v1" or meta.get("source") != "perception":
            raise ValueError("Approach planning requires an observed grid, not a hidden scene map")
        raster = own_path(path.parent, meta["state_file"])
        if file_sha(raster) != meta["state_sha256"]: raise ValueError("Observed grid identity mismatch")
        return cls(np.load(raster, allow_pickle=False), meta["origin_xy_m"], meta["resolution_m"], footprint_radius_m)

    def cell(self, xy):
        xy = np.asarray(xy, dtype=float)
        if xy.shape != (2,) or not np.isfinite(xy).all(): raise ValueError("Expected finite XY")
        x, y = np.floor((xy-self.origin)/self.resolution).astype(int)
        return int(y), int(x)

    def xy(self, cell): return (self.origin + (np.array(cell[::-1])+ .5)*self.resolution).tolist()

    def is_safe(self, cell):
        y, x = cell
        return 0 <= y < self.safe.shape[0] and 0 <= x < self.safe.shape[1] and bool(self.safe[y, x])

    def segment_is_safe(self, start, end):
        a, b = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
        count = max(2, math.ceil(np.linalg.norm(b-a)/(self.resolution/4))+1)
        return all(self.is_safe(self.cell(p)) for p in np.linspace(a, b, count))

    def approach(self, surface, robot_xy, medium="either", min_standoff=1.5, max_standoff=3.):
        if not surface.get("accepted") or surface.get("camera_role") != "robot_mounted":
            raise ValueError("Use an accepted robot-mounted depth surface observation")
        if medium not in ["land", "water", "either"] or not 0 < min_standoff < max_standoff:
            raise ValueError("Invalid arrival medium or standoff")
        target = np.asarray(surface["world_point_m"][:2], dtype=float)
        if not np.isfinite(target).all(): raise ValueError("Invalid observed target position")
        start = self.cell(robot_xy)
        if not self.is_safe(start): raise ValueError("Robot's current footprint is outside observed safe space")
        costs, parents = {start: 0.}, {}
        queue = [(0., start)]
        allowed = {"land": {1}, "water": {2}, "either": {1, 2}}[medium]
        chosen = None
        while queue:
            cost, cell = heapq.heappop(queue)
            if cost > costs[cell]+1e-9: continue
            distance = np.linalg.norm(np.asarray(self.xy(cell))-target)
            if min_standoff <= distance <= max_standoff and self.state[cell] in allowed:
                chosen = cell; break
            for dy, dx in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                neighbour = cell[0]+dy, cell[1]+dx
                if not self.is_safe(neighbour): continue
                if dy and dx and (not self.is_safe((cell[0]+dy,cell[1])) or not self.is_safe((cell[0],cell[1]+dx))): continue
                penalty = 1.25 if self.state[neighbour] == 2 else 1.
                candidate = cost + math.hypot(dx,dy)*self.resolution*penalty
                if candidate < costs.get(neighbour, float("inf")):
                    costs[neighbour] = candidate; parents[neighbour] = cell
                    heapq.heappush(queue,(candidate,neighbour))
        if chosen is None: raise ValueError("No observed reachable arrival pose at the requested standoff")
        path, cell = [chosen], chosen
        while cell != start: cell = parents[cell]; path.append(cell)
        xy = self.xy(chosen)
        return {"kind": "robot_arrival_pose", "source": "observed_traversability", "approach_validated": True,
                "goal_xy_m": xy, "goal_yaw_rad": math.atan2(target[1]-xy[1],target[0]-xy[0]),
                "arrival_medium": medium, "standoff_m": float(np.linalg.norm(np.asarray(xy)-target)),
                "observed_path_xy_m": [self.xy(c) for c in reversed(path)],
                "target_surface": surface, "task_success_verified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface", type=Path, required=True)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--robot-xy", type=float, nargs=2, required=True)
    parser.add_argument("--medium", choices=["land","water","either"], default="either")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project",type=Path,default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    value = read_json(args.surface)
    goal = ObservedGrid.load(args.grid).approach(value.get("surface",value), args.robot_xy, args.medium)
    goal["grid_file"] = own_path(args.project,args.grid.resolve()).relative_to(args.project.resolve()).as_posix()
    write_json(args.output, goal)
    print("Observed arrival pose:", goal["goal_xy_m"])


if __name__ == "__main__": main()
