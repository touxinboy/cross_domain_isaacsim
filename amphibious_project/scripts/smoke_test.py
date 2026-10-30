"""Verify real rigid-body stepping, ground contact and CSV output."""

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--portable-root", required=True)
    args, _ = parser.parse_known_args()
    project = Path(__file__).resolve().parents[1]
    config = json.loads((project / "configs/task.json").read_text())

    from isaacsim import SimulationApp
    application = SimulationApp({
        "headless": True, "active_gpu": config["gpu_index"],
        "physics_gpu": config["gpu_index"], "multi_gpu": False,
        "width": 640, "height": 480, "fast_shutdown": False,
        "extra_args": ["--portable-root", args.portable_root,
                       "--/app/extensions/registryEnabled=false"],
    })
    try:
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleRigidPrim

        scene = project / "scenes/river_stage_01.usda"
        if not omni.usd.get_context().open_stage(str(scene)):
            raise RuntimeError(f"Could not open {scene}")
        application.update()
        world = World(physics_dt=config["physics_dt_s"], rendering_dt=1.0 / 30.0,
                      stage_units_in_meters=1.0, physics_prim_path="/World/PhysicsScene")
        robot = world.scene.add(SingleRigidPrim(prim_path="/World/Robot", name="robot"))
        world.reset()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output = project / "results" / f"smoke_{stamp}"
        output.mkdir(parents=True, exist_ok=False)
        columns = ["step", "sim_time_s", "x_m", "y_m", "z_m", "qw", "qx", "qy", "qz",
                   "vx_m_s", "vy_m_s", "vz_m_s"]
        samples = []
        with (output / "states.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(columns)
            for step in range(config["smoke_steps"]):
                world.step(render=False)
                position, orientation = robot.get_world_pose()
                velocity = robot.get_linear_velocity()
                row = [step + 1, float(world.current_time), *map(float, position),
                       *map(float, orientation), *map(float, velocity)]
                writer.writerow(row)
                samples.append(row)
        final_z = samples[-1][4]
        final_vz = samples[-1][-1]
        passed = (all(math.isfinite(value) for row in samples for value in row)
                  and 0.43 <= final_z <= 0.65 and abs(final_vz) < 0.2
                  and samples[0][4] - final_z > 0.2)
        summary = {"test": "rigid_body_ground_contact", "passed": passed,
                   "samples": len(samples), "duration_s": samples[-1][1],
                   "final_z_m": final_z, "final_vz_m_s": final_vz,
                   "output_dir": str(output), "navigation_test": False,
                   "water_dynamics_test": False}
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print("L17_SMOKE_RESULT " + json.dumps(summary), flush=True)
        if not passed:
            raise RuntimeError("Ground-contact verification failed; inspect the recorded states.")
    finally:
        application.close()


if __name__ == "__main__":
    main()
