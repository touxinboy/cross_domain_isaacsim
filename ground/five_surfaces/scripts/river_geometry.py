"""Deterministic river geometry and object placement, independent of the renderer."""

import math
import random


def bell(y, center, width):
    return math.exp(-0.5 * ((y - center) / width) ** 2)


class RiverGeometry:
    def __init__(self, config):
        self.config = config
        self.half_x = config["width_m"] / 2
        self.half_y = config["length_m"] / 2
        self.resolution = config["terrain_resolution_m"]
        self.water_z = config["water_level_m"]
        self.nx = round(config["width_m"] / self.resolution)
        self.ny = round(config["length_m"] / self.resolution)

    def cross_section(self, y):
        center = 2.1 * math.sin(0.075 * y) + 0.65 * math.sin(0.22 * y + 0.6)
        shore = 3.5 + 0.65 * math.sin(0.11 * y + 1) + 0.30 * math.sin(0.39 * y - 0.3)
        shore += 0.75 * bell(y, -23, 5)
        depth = 1.85 + 0.4 * math.sin(0.12 * y - 0.5)
        depth -= 1.15 * bell(y, -23, 4) + 0.45 * bell(y, 21, 4.5)
        depth = max(0.65, min(2.30, depth))
        left_run = 2.1 + 2.9 * bell(y, -23, 4.5) + 1.6 * bell(y, 21, 5)
        right_run = 1.9 + 3.1 * bell(y, -23, 4.5) + 2.0 * bell(y, 21, 5)
        right_run -= 1.25 * bell(y, 1, 5)
        return {"center_x": center, "shore_half_width": shore, "depth": depth,
                "bed_half_width": 0.48 * shore, "left_run": left_run,
                "right_run": max(0.65, right_run)}

    def height(self, x, y):
        s = self.cross_section(y)
        dx = x - s["center_x"]
        r = abs(dx)
        bed = -s["depth"] + 0.045 * math.sin(0.55 * x + 0.21 * y) * math.cos(0.44 * y)
        if r <= s["bed_half_width"]:
            return self.water_z + bed
        if r <= s["shore_half_width"]:
            t = (r - s["bed_half_width"]) / (s["shore_half_width"] - s["bed_half_width"])
            return self.water_z + bed * (1 - t)
        run = s["left_run"] if dx < 0 else s["right_run"]
        plateau = 1.0 + 0.12 * math.sin(0.16 * y + 0.3) if dx < 0 else 1.05 + 0.15 * math.cos(0.14 * y - 0.3)
        plateau += 0.095 * math.sin(0.4 * x + 0.23 * y) * math.cos(0.19 * y - 0.11 * x)
        plateau += 0.025 * math.sin(1.1 * x + 0.7 * y)
        t = min(1, (r - s["shore_half_width"]) / run)
        return self.water_z + plateau * t

    def soil(self, x, y):
        z = self.height(x, y) - self.water_z
        s = self.cross_section(y)
        dx = x - s["center_x"]
        dry_distance = abs(dx) - s["shore_half_width"]
        if z < -0.12:
            return "Riverbed"
        muddy = ((y - 20) / 8) ** 2 + ((dry_distance - 1.5) / 1.8) ** 2 < 1 and dx < 0
        muddy |= ((y + 9) / 4) ** 2 + ((dry_distance - 1.0) / 1.1) ** 2 < 1 and dx > 0
        if muddy and z > 0.02:
            return "Mud"
        if dry_distance < 0.65 or z < 0.15:
            return "WetSand"
        run = s["left_run"] if dx < 0 else s["right_run"]
        return "DrySand" if dry_distance < run + 0.25 else "Grass"

    def grid(self):
        points = []
        for iy in range(self.ny + 1):
            y = -self.half_y + iy * self.resolution
            for ix in range(self.nx + 1):
                x = -self.half_x + ix * self.resolution
                points.append((x, y, self.height(x, y)))
        groups = {name: [] for name in self.config["materials"]}
        for iy in range(self.ny):
            for ix in range(self.nx):
                a = iy * (self.nx + 1) + ix
                b, c, d = a + 1, a + self.nx + 2, a + self.nx + 1
                x = -self.half_x + (ix + 0.5) * self.resolution
                y = -self.half_y + (iy + 0.5) * self.resolution
                groups[self.soil(x, y)].extend([(a, b, c), (a, c, d)])
        return points, groups

    def mesh_height(self, x, y):
        """Height on the actual piecewise triangular mesh, not a smooth approximation."""
        fx = min(self.nx - 1e-8, max(0, (x + self.half_x) / self.resolution))
        fy = min(self.ny - 1e-8, max(0, (y + self.half_y) / self.resolution))
        ix, iy = int(fx), int(fy)
        u, v = fx - ix, fy - iy
        x0, y0 = -self.half_x + ix * self.resolution, -self.half_y + iy * self.resolution
        a = self.height(x0, y0)
        b = self.height(x0 + self.resolution, y0)
        c = self.height(x0 + self.resolution, y0 + self.resolution)
        d = self.height(x0, y0 + self.resolution)
        return a + u * (b - a) + v * (c - b) if v <= u else a + u * (c - d) + v * (d - a)

    def probes(self):
        out = []
        for y in [-28.0, -23.0, -12.0, 0.0, 12.0, 21.0, 28.0]:
            s = self.cross_section(y)
            offsets = [-(s["shore_half_width"] + s["left_run"] + 1.1),
                       -(s["shore_half_width"] + 0.6), 0.0,
                       s["shore_half_width"] + 0.4,
                       s["shore_half_width"] + s["right_run"] + 1.1]
            for i, off in enumerate(offsets):
                x = s["center_x"] + off
                out.append({"x": x, "y": y, "z": self.mesh_height(x, y), "zone": i})
        return out

    def reserved(self, x, y, radius=0.0):
        for mission in self.config["missions"]:
            for marker in ["start_xy_m", "goal_xy_m"]:
                a, b = mission[marker]
                if math.hypot(x - a, y - b) < radius + 1.6:
                    return True
        for p in self.probes():
            if math.hypot(x - p["x"], y - p["y"]) < radius + 0.5:
                return True
        if min(abs(y + 23), abs(y - 21)) < radius + 1.2:
            return True
        return False

    def objects(self):
        rng = random.Random(self.config["seed"])
        trees, rocks = [], []
        for _ in range(16000):
            if len(trees) >= self.config["tree_count"]:
                break
            x, y = rng.uniform(-self.half_x + 0.7, self.half_x - 0.7), rng.uniform(-self.half_y + 0.9, self.half_y - 0.9)
            s = self.cross_section(y)
            dx = x - s["center_x"]
            run = s["left_run"] if dx < 0 else s["right_run"]
            if abs(dx) < s["shore_half_width"] + run + 0.65 or self.reserved(x, y, 0.6):
                continue
            if any(math.hypot(x - t["x"], y - t["y"]) < 1.45 for t in trees):
                continue
            height = rng.uniform(2.7, 5.3)
            trees.append({"x": x, "y": y, "z": self.mesh_height(x, y), "height": height,
                          "radius": rng.uniform(0.13, 0.21), "crown_radius": height * rng.uniform(0.19, 0.24),
                          "variant": rng.randrange(8), "yaw": rng.uniform(0, 360)})
        for _ in range(12000):
            if len(rocks) >= self.config["rock_count"]:
                break
            n = len(rocks)
            y = rng.uniform(-self.half_y + 1.4, self.half_y - 1.4)
            s = self.cross_section(y)
            sign = rng.choice([-1, 1])
            if n < 12:
                x = s["center_x"] + rng.uniform(-0.85, 0.85) * s["shore_half_width"]
            elif n < 50:
                x = s["center_x"] + sign * (s["shore_half_width"] + rng.uniform(-0.45, 1.5))
            else:
                x = rng.uniform(-self.half_x + 1.2, self.half_x - 1.2)
            size = rng.uniform(0.35, 0.85)
            if abs(x) > self.half_x - 1.1 or self.reserved(x, y, size):
                continue
            if any(math.hypot(x - t["x"], y - t["y"]) < size + t["radius"] + 0.45 for t in trees):
                continue
            if any(math.hypot(x - r["x"], y - r["y"]) < size + r["radii"][0] + 0.3 for r in rocks):
                continue
            rocks.append({"x": x, "y": y, "z": self.mesh_height(x, y),
                          "radii": [size, size * rng.uniform(0.65, 1.15), size * rng.uniform(0.7, 1.35)],
                          "variant": rng.randrange(8), "yaw": rng.uniform(0, 360)})
        if len(trees) != self.config["tree_count"] or len(rocks) != self.config["rock_count"]:
            raise RuntimeError("Object placement could not satisfy the requested counts and clearances.")
        return trees, rocks

    def water_mesh(self):
        points, triangles = [], []
        for i in range(self.ny + 1):
            y = -self.half_y + i * self.resolution
            s = self.cross_section(y)
            for sign in [-1, 1]:
                points.append((s["center_x"] + sign * s["shore_half_width"], y, self.water_z))
            if i:
                a = (i - 1) * 2
                triangles.extend([(a, a + 1, a + 3), (a, a + 3, a + 2)])
        return points, triangles
