# Measurement protocol for the team's robot

Record the results in `data/sysid/processed/<name>.yaml` and pass that file to
`scripts/import_robot.py --measurements`, which applies them to the model and the config
([robot_import.md](robot_import.md), section 2, lists the keys). Formulas are implemented in
`src/quadruped_rl/sysid/mass_properties.py`.

## Accuracy targets

| Parameter | Target | Notes |
|---|---|---|
| Total mass | 1-2% | sets the torque needed to stand and walk |
| Base COM (x, y) | 5-10 mm | errors make the policy lean or drift |
| Joint zero offsets | about 1 deg | errors compound down the leg |
| Leg link masses | 5-10% | legs are light relative to the body |
| Armature, friction, damping | from swing tests | dominate leg dynamics on geared motors |
| Base COM height | 1-2 cm | tipping and pitch |
| Inertias | 20-30% | partly absorbed by domain randomization |
| Foot friction | rough estimate | then randomize around it |

Whatever uncertainty remains sets the domain randomization range.

## Steps

1. **Correct the CAD.** Real material densities; add battery, computer, wiring, and
   fasteners as boxes or point masses where they actually sit.
2. **Weigh** the whole robot and individual links (one leg is enough if all four match).
3. **Horizontal COM.** Each foot on its own scale in the nominal stance:
   $x_{\text{com}} = \dfrac{\sum_i F_i x_i}{\sum_i F_i}$, and likewise for $y$.
4. **COM height (tilt test).** Body on two supports a distance $L$ apart. Read the support
   at the end you will raise ($F$), raise it by $\theta$ of 20-30 degrees, read again ($F'$):
   $h = \dfrac{(F - F')\,L}{W \tan\theta}$.
5. **Inertia (pendulum tests).** Compound pendulum:
   $I_{\text{com}} = \dfrac{m g d\, T^2}{4\pi^2} - m d^2$.
   Bifilar pendulum (yaw of the body): $I = \dfrac{m g d^2 T^2}{16 \pi^2 L}$.
   Time 10-20 swings and divide.
6. **Swing tests** on the gantry, motors off: release each leg and log $q(t)$ from the
   encoders. Fit armature, damping, and friction with `scripts/fit_sysid.py`.
   Armature starts from $N^2 I_{\text{rotor}}$.
7. **Zero offsets** with an alignment jig; **foot friction** from the slip angle on a tilted
   board, $\mu = \tan\theta_{\text{slip}}$, for each floor surface.

Record raw readings in `data/sysid/raw/` and the resulting values in `data/sysid/processed/`.
