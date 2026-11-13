"""Run a policy trained on the GPU backend in plain CPU MuJoCo. If it fails here,
it has overfit to simulator quirks and will not survive hardware.

    python scripts/sim2sim.py outputs/ppo/rough_velocity/<run>/
"""


def main() -> None:
    raise SystemExit("scripts/sim2sim.py: not implemented yet (phase 2).")


if __name__ == "__main__":
    main()
