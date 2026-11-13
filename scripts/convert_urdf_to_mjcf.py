"""Convert the team's URDF to MJCF with MuJoCo's built-in importer.

    python scripts/convert_urdf_to_mjcf.py assets/robots/custom/urdf/robot.urdf \
        --out assets/robots/custom/mjcf/robot.xml

Before converting, add this inside the URDF's <robot> tag so meshes resolve and
visual geometry is kept:

    <mujoco><compiler meshdir="../meshes" balanceinertia="true" discardvisual="false"/></mujoco>

Replace any package:// mesh paths with plain file names.
"""

from __future__ import annotations

import argparse
from pathlib import Path

CHECKLIST = """
Converted. MuJoCo's URDF import is only a starting point. Edit the MJCF by hand:
  [ ] add <freejoint/> to the base body (URDF imports as fixed-base)
  [ ] add one <position> or <motor> actuator per joint, with ctrlrange / forcerange
  [ ] set joint armature, damping, frictionloss (from swing tests; start from datasheet)
  [ ] give each foot a named sphere collision geom (names go in the robot YAML)
  [ ] simplify collision geometry (primitives instead of meshes where possible)
  [ ] check total mass and base COM against your measurements
  [ ] write a scene.xml with a floor and light that includes the robot
"""


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("urdf", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    import mujoco

    model = mujoco.MjModel.from_xml_path(str(args.urdf))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    mujoco.mj_saveLastXML(str(args.out), model)
    print(f"Wrote {args.out}  (bodies: {model.nbody}, joints: {model.njnt}, mass: "
          f"{model.body_mass.sum():.3f} kg)")
    print(CHECKLIST)


if __name__ == "__main__":
    main()
