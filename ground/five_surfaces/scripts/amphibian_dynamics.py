"""Force-based prototype model. PhysX alone integrates the resulting rigid-body motion."""

import math
import numpy as np


def quaternion_matrix_xyzw(q):
    x, y, z, w = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def euler_from_matrix(r):
    return (math.atan2(r[2, 1], r[2, 2]), math.asin(float(np.clip(-r[2, 0], -1, 1))),
            math.atan2(r[1, 0], r[0, 0]))


class AmphibianForces:
    def __init__(self, config):
        self.cfg = config
        b = config["buoyancy"]
        size = np.asarray(b["sealed_box_size_m"], dtype=float)
        count = np.asarray(b["quadrature_cells"], dtype=int)
        self.cell_size = size / count
        axes = [(np.arange(n) + 0.5) * d - s / 2 for n, d, s in zip(count, self.cell_size, size)]
        grid = np.meshgrid(*axes, indexing="ij")
        self.cells = np.stack(grid, axis=-1).reshape(-1, 3) + b["sealed_box_center_body_m"]
        self.cell_volume = float(np.prod(self.cell_size))
        self.max_volume = float(np.prod(size))
        self.rho = b["water_density_kg_m3"]
        self.gravity = config["gravity_m_s2"]
        self.equilibrium_z = float(-(b["sealed_box_center_body_m"][2] - size[2]/2) - config["mass_kg"] / (self.rho * size[0] * size[1]))
        self.reset()

    def reset(self):
        self.belt_speed = np.zeros(2)
        self.thrust_state = np.zeros(2)
        self.load = np.zeros(2)

    def buoyancy(self, position, rotation, environment, enabled=True):
        if not enabled:
            return np.zeros(3), np.zeros(3), 0.0
        centers = self.cells @ rotation.T + position
        height = max(1e-6, float(np.abs(rotation[2]) @ self.cell_size))
        levels, beds, wet = environment.water_columns(centers[:, 0], centers[:, 1])
        lower = np.maximum(centers[:, 2] - height / 2, beds)
        upper = np.minimum(centers[:, 2] + height / 2, levels)
        fraction = np.clip((upper - lower) / height, 0, 1) * wet
        magnitudes = self.rho * self.gravity * self.cell_volume * fraction
        force = np.array([0., 0., float(magnitudes.sum())])
        relative = centers - position
        torque = np.array([float(np.sum(relative[:, 1] * magnitudes)),
                           float(-np.sum(relative[:, 0] * magnitudes)), 0.])
        return force, torque, float(fraction.sum() * self.cell_volume)

    def compute(self, position, quaternion_xyzw, velocity, commands, contacts, environment, dt,
                buoyancy_enabled=True):
        position = np.asarray(position, dtype=float)
        velocity = np.asarray(velocity, dtype=float)
        rotation = quaternion_matrix_xyzw(quaternion_xyzw)
        mode = commands.get("mode", "HYBRID")
        track_ref = np.asarray(commands.get("tracks", [0., 0.]), dtype=float)
        throttle = np.asarray(commands.get("thrusters", [0., 0.]), dtype=float)
        if mode == "LAND":
            throttle *= 0
        elif mode == "WATER":
            track_ref *= 0
        tr = self.cfg["tracks"]
        th = self.cfg["thrusters"]
        track_ref = np.clip(track_ref, -tr["max_speed_m_s"], tr["max_speed_m_s"])
        throttle = np.clip(throttle, -1, 1)
        self.belt_speed += (1-math.exp(-dt/tr["motor_time_constant_s"])) * (track_ref-self.belt_speed)
        desired_thrust = th["max_thrust_each_n"] * throttle
        self.thrust_state += (1-math.exp(-dt/th["time_constant_s"])) * (desired_thrust-self.thrust_state)
        force, torque, volume = self.buoyancy(position, rotation, environment, buoyancy_enabled)
        buoyancy_n = float(force[2])
        buoyancy_torque = torque.copy()
        reference_volume = self.cfg["mass_kg"] / self.rho
        wet_scale = min(1.5, volume/reference_volume)
        hydro = self.cfg["hydrodynamics"]
        rel_velocity = velocity[:3] - hydro["current_world_m_s"]
        body_v = rotation.T @ rel_velocity
        body_w = rotation.T @ velocity[3:]
        drag_body = -(np.asarray(hydro["linear_drag_body"])*body_v +
                      np.asarray(hydro["quadratic_drag_body"])*np.abs(body_v)*body_v) * wet_scale
        angular_drag = -(np.asarray(hydro["linear_angular_drag_body"])*body_w +
                         np.asarray(hydro["quadratic_angular_drag_body"])*np.abs(body_w)*body_w) * wet_scale
        force += rotation @ drag_body
        torque += rotation @ angular_drag
        effective_thrust = []
        immersion = []
        for i, point in enumerate(th["positions_body_m"]):
            lever = rotation @ np.asarray(point)
            world = position + lever
            levels, beds, wet = environment.water_columns(np.array([world[0]]), np.array([world[1]]))
            radius = th["propeller_radius_m"]
            fraction = float(np.clip((levels[0]-world[2]+radius)/(2*radius), 0, 1)) * float(wet[0])
            fraction *= float(world[2] + radius > beds[0])
            thrust = self.thrust_state[i] * fraction
            f = rotation @ (np.asarray(th["force_axis_body"]) * thrust)
            force += f
            torque += np.cross(lever, f)
            effective_thrust.append(float(thrust))
            immersion.append(fraction)
        raw_load = np.array([max(0, c["normal_n"]) for c in contacts], dtype=float)
        self.load += (1-math.exp(-dt/tr["contact_load_filter_s"])) * (raw_load-self.load)
        drive_forces = np.zeros(2)
        track_power = 0.0
        for side in range(2):
            normal = np.asarray(contacts[side].get("normal", [0., 0., 1.]), dtype=float)
            normal /= max(1e-9, np.linalg.norm(normal))
            longitudinal = rotation[:, 0] - normal * float(rotation[:, 0] @ normal)
            longitudinal /= max(1e-9, np.linalg.norm(longitudinal))
            lateral = np.cross(normal, longitudinal)
            # Actual contact reports gate traction. Smoothed load alone cannot drive the robot in air.
            load = self.load[side] if raw_load[side] > 0.5 else 0.0
            mu = contacts[side].get("mu", environment.default_ground_mu)
            y = tr["center_separation_m"] / 2 * (1 if side == 0 else -1)
            for x in tr["patch_x_m"]:
                lever = rotation @ np.array([x, y, tr["center_z_m"]-tr["height_m"]/2])
                patch_v = velocity[:3] + np.cross(velocity[3:], lever)
                vx, vy = float(patch_v @ longitudinal), float(patch_v @ lateral)
                limit = mu * load / len(tr["patch_x_m"])
                fx = tr["longitudinal_stiffness_per_track_ns_m"] / 2 * (self.belt_speed[side]-vx)
                fx -= tr["rolling_resistance_coefficient"] * load / 2 * math.tanh(vx/0.05)
                fx = float(np.clip(fx, -limit, limit))
                lateral_limit = math.sqrt(max(0, limit*limit-fx*fx))
                fy = float(np.clip(-tr["lateral_stiffness_per_patch_ns_m"]*vy, -lateral_limit, lateral_limit))
                f = longitudinal*fx + lateral*fy
                force += f
                torque += np.cross(lever, f)
                drive_forces[side] += fx
                track_power += abs(fx*self.belt_speed[side])
        ideal_prop_power = sum(abs(t)**1.5 for t in effective_thrust) / math.sqrt(2*self.rho*math.pi*th["propeller_radius_m"]**2)
        roll, pitch, yaw = euler_from_matrix(rotation)
        telemetry = {"roll_deg": math.degrees(roll), "pitch_deg": math.degrees(pitch), "yaw_rad": yaw,
                     "buoyancy_n": buoyancy_n, "displaced_volume_m3": volume,
                     "submerged_fraction": volume/self.max_volume,
                     "buoyancy_roll_torque_nm": float(buoyancy_torque[0]),
                     "buoyancy_pitch_torque_nm": float(buoyancy_torque[1]),
                     "left_normal_n": float(raw_load[0]), "right_normal_n": float(raw_load[1]),
                     "left_belt_m_s": float(self.belt_speed[0]), "right_belt_m_s": float(self.belt_speed[1]),
                     "left_traction_n": float(drive_forces[0]), "right_traction_n": float(drive_forces[1]),
                     "left_thrust_n": effective_thrust[0], "right_thrust_n": effective_thrust[1],
                     "left_thruster_immersion": immersion[0], "right_thruster_immersion": immersion[1],
                     "track_mechanical_power_w": track_power, "thruster_ideal_power_proxy_w": ideal_prop_power,
                     "force_x_n": float(force[0]), "force_y_n": float(force[1]), "force_z_n": float(force[2]),
                     "torque_x_nm": float(torque[0]), "torque_y_nm": float(torque[1]), "torque_z_nm": float(torque[2])}
        return force, torque, telemetry
