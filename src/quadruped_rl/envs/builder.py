"""Builds an mjlab environment from a robot config and a task config.

This is where YAML becomes simulation: the robot YAML becomes an mjlab entity (model
file, PD actuators, collisions, spawn pose), and the task YAML picks observation,
reward, and termination terms by name from observations.py, rewards.py, and
terminations.py. Nothing here is specific to the Go1, so the team's robot only needs
its own YAML.

    env = make_env(load_robot("go1"), load_group("tasks", "flat_velocity"),
                   num_envs=4096, device="cuda:0")
"""

from __future__ import annotations

import re

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg, DcMotorActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, GridPatternCfg, ObjRef, RayCastSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity import mdp as velocity_mdp
from mjlab.utils.noise import UniformNoiseCfg
from mjlab.utils.spec_config import CollisionCfg
from mjlab.viewer import ViewerConfig

from quadruped_rl.envs import observations as obs_terms
from quadruped_rl.envs import rewards as reward_terms
from quadruped_rl.envs import terminations as term_terms
from quadruped_rl.envs.base import QuadrupedEnv
from quadruped_rl.envs.commands import COMMAND_NAME, make_velocity_command
from quadruped_rl.envs.domain_randomization import action_delay_lag, make_events
from quadruped_rl.envs.terrain import make_terrain, terrain_levels_vel, uses_generator

HEIGHT_SCAN_RANGE = 5.0  # m, max ray distance; scan values are divided by this


def _exact(names) -> str:
    """Regex matching exactly the given names."""
    return "^(" + "|".join(re.escape(n) for n in names) + ")$"


# ---- robot ------------------------------------------------------------------------

def _actuator_cfg(act, j: int, name: str, lag_min: int, lag_max: int):
    """One joint's actuator. ideal_pd (and anything else): MuJoCo's built-in position
    actuator, clipped at torque_limit. dc_motor: the same PD computed in torch, clipped to
    the torque-speed line saturation_effort * (1 - w / no_load_speed), capped at
    torque_limit (mjlab DcMotorActuator)."""
    common = dict(
        target_names_expr=(re.escape(name),),
        stiffness=act.kp[j],
        damping=act.kd[j],
        effort_limit=act.torque_limit[j],
        armature=act.armature[j],
        frictionloss=None if act.frictionloss is None else act.frictionloss[j],
        viscous_damping=None if act.viscous_damping is None else act.viscous_damping[j],
        delay_min_lag=lag_min,
        delay_max_lag=lag_max,
    )
    if act.model == "dc_motor":
        return DcMotorActuatorCfg(**common, saturation_effort=act.params["saturation_effort"][j],
                                  velocity_limit=act.params["no_load_speed"][j])
    return BuiltinPositionActuatorCfg(**common)


def make_robot_entity(robot, decimation: int) -> EntityCfg:
    act = robot.actuator
    lag_min, lag_max = action_delay_lag(robot, decimation)
    actuators = tuple(_actuator_cfg(act, j, name, lag_min, lag_max)
                      for j, name in enumerate(robot.joint_names))
    for geom in robot.foot_geoms:
        if not re.fullmatch(robot.collision_geoms, geom):
            raise ValueError(f"{robot.name}: foot geom '{geom}' does not match collision_geoms "
                             f"'{robot.collision_geoms}'")
    feet = _exact(robot.foot_geoms)
    collisions = CollisionCfg(
        geom_names_expr=(robot.collision_geoms,),
        contype=1,
        conaffinity=1,
        solref=(0.01, 1),
        condim={feet: 6, robot.collision_geoms: 1},   # only feet carry friction
        priority={feet: 1, ".*": 0},
        friction={feet: (1, 5e-3, 5e-4)},
    )
    model_file = robot.model_file
    if not model_file.exists():
        raise FileNotFoundError(f"{robot.name}: model not found at {model_file}")
    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, robot.base_height_nominal),
            joint_pos={re.escape(n): q for n, q in zip(robot.joint_names, robot.default_joint_pos)},
            joint_vel={".*": 0.0},
        ),
        collisions=(collisions,),
        spec_fn=lambda: mujoco.MjSpec.from_file(str(model_file)),
        articulation=EntityArticulationInfoCfg(actuators=actuators, soft_joint_pos_limit_factor=0.9),
    )


def make_action(robot, action_cfg: dict) -> JointPositionActionCfg:
    scale = float(action_cfg.get("scale", 0.25))
    mode = action_cfg.get("scale_mode", "absolute")
    act = robot.actuator
    if mode == "effort_relative":
        scales = [scale * tau / kp for tau, kp in zip(act.torque_limit, act.kp)]
    elif mode == "absolute":
        scales = [scale] * robot.num_joints
    else:
        raise ValueError(f"Unknown action scale_mode '{mode}'")
    return JointPositionActionCfg(
        entity_name="robot",
        actuator_names=tuple(re.escape(n) for n in robot.joint_names),
        scale={re.escape(n): s for n, s in zip(robot.joint_names, scales)},
        use_default_offset=True,
        preserve_order=True,
    )


# ---- sensors ----------------------------------------------------------------------

def _contact_sensor(name: str, patterns, air_time: bool = False) -> ContactSensorCfg:
    return ContactSensorCfg(
        name=name,
        primary=ContactMatch(mode="geom", pattern=tuple(patterns), entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce" if air_time else "none",
        num_slots=1,
        track_air_time=air_time,
    )


def make_sensors(robot, needed: set[str]) -> tuple:
    sensors = []
    if "feet_contact" in needed:
        sensors.append(_contact_sensor("feet_contact", robot.foot_geoms, air_time=True))
    if "undesired_contact" in needed:
        if not robot.undesired_contact_geoms:
            raise ValueError(f"{robot.name}: undesired_contacts needs undesired_contact_geoms")
        sensors.append(_contact_sensor("undesired_contact", robot.undesired_contact_geoms))
    if "base_contact" in needed:
        if not robot.base_contact_geoms:
            raise ValueError(f"{robot.name}: base_contact needs base_contact_geoms")
        sensors.append(_contact_sensor("base_contact", robot.base_contact_geoms))
    if "terrain_scan" in needed:
        sensors.append(RayCastSensorCfg(
            name="terrain_scan",
            frame=ObjRef(type="body", name=robot.base_body, entity="robot"),
            ray_alignment="yaw",
            pattern=GridPatternCfg(size=(1.6, 1.0), resolution=0.1),
            max_distance=HEIGHT_SCAN_RANGE,
            exclude_parent_body=True,
            include_geom_groups=(0,),
            debug_vis=True,
        ))
    return tuple(sensors)


# ---- observations, rewards, terminations ------------------------------------------

def _obs_term(name: str, noise: float | None, biased_joints: bool) -> ObservationTermCfg:
    func = getattr(obs_terms, name, None)
    if func is None or name.startswith("_") or not callable(func):
        raise KeyError(f"Unknown observation term '{name}' (see envs/observations.py)")
    params, scale = {}, None
    if name == "joint_pos_rel" and biased_joints:
        params["biased"] = True
    if name == "height_scan":
        scale = 1.0 / HEIGHT_SCAN_RANGE
    return ObservationTermCfg(
        func=func,
        params=params,
        noise=UniformNoiseCfg(n_min=-noise, n_max=noise) if noise else None,
        scale=scale,
    )


def make_observations(obs_cfg: dict, robot) -> dict[str, ObservationGroupCfg]:
    noise = obs_cfg.get("noise") or {}
    biased = "encoder_bias" in robot.domain_randomization
    groups = {}
    for group, corrupt in (("actor", True), ("critic", False)):
        names = obs_cfg[group]
        terms = {n: _obs_term(n, noise.get(n) if corrupt else None, biased and corrupt)
                 for n in names}
        groups[group] = ObservationGroupCfg(terms=terms, concatenate_terms=True,
                                            enable_corruption=corrupt)
    return groups


def _num(value):
    """float(value), keeping None (a null reward_params entry is reported below)."""
    return None if value is None else float(value)


def _band(value) -> dict:
    low, high = (None, None) if value is None else value
    return {"low": _num(low), "high": _num(high)}


def make_rewards(task: dict, robot=None) -> dict[str, RewardTermCfg]:
    rp = task.get("reward_params") or {}
    params_for = {
        "track_lin_vel_xy": lambda: {"sigma": rp["tracking_sigma"]},
        "track_ang_vel_z": lambda: {"sigma": rp["tracking_sigma"]},
        "feet_air_time": lambda: {"target": rp["air_time_target"]},
        "base_height": lambda: {"target": rp["base_height_target"]},
        "foot_clearance": lambda: {"target": _num(rp.get("foot_clearance_target")),
                                   "std": _num(rp.get("foot_clearance_std")),
                                   "tanh_mult": _num(rp.get("foot_clearance_tanh_mult"))},
        "feet_stumble": lambda: {"ratio": _num(rp.get("stumble_force_ratio"))},
        "base_height_band": lambda: _band(rp.get("base_height_band")),
        "knee_bend": lambda: {"joint_ids": reward_terms.calf_joint_ids(robot.joint_names),
                              "min_bend": _num(rp.get("knee_min_bend"))},
        "torques_rated": lambda: {"tau_ref": reward_terms.rated_torque(robot)},
        "torque_rate": lambda: {"tau_ref": reward_terms.rated_torque(robot)},
        "feet_impact": lambda: {"k": _num(rp.get("impact_force_k"))},
    }
    rewards = {}
    for name, weight in task["rewards"].items():
        if isinstance(weight, dict):
            raise NotImplementedError(f"Reward '{name}' is an imitation term (phase 4).")
        if float(weight) == 0.0:
            continue
        func = getattr(reward_terms, name, None)
        if func is None or name.startswith("_") or not callable(func):
            raise KeyError(f"Unknown reward term '{name}' (see envs/rewards.py)")
        params = params_for[name]() if name in params_for else {}
        if any(v is None for v in params.values()):
            raise ValueError(f"Reward '{name}' is enabled but its reward_params entry is null")
        rewards[name] = RewardTermCfg(func=func, weight=float(weight), params=params)
    return rewards


def make_reward_ramps(task: dict, rewards: dict) -> dict[str, CurriculumTermCfg]:
    """Curriculum terms for ``reward_ramps: {term: {start_steps, end_steps, start_scale}}``:
    the term's weight goes linearly from start_scale x weight to weight between those env
    steps (env.common_step_counter). Ramps on disabled (weight 0) terms are ignored."""
    ramps = {}
    for name, ramp in (task.get("reward_ramps") or {}).items():
        if name not in task["rewards"]:
            raise KeyError(f"reward_ramps.{name}: no such entry under rewards")
        if name not in rewards:
            continue
        ramps[f"ramp_{name}"] = CurriculumTermCfg(func=reward_terms.reward_weight_ramp, params={
            "reward_name": name,
            "start_steps": int(ramp["start_steps"]),
            "end_steps": int(ramp["end_steps"]),
            "start_scale": float(ramp.get("start_scale", 0.0)),
        })
    return ramps


def make_terminations(term_cfg: dict, generator_terrain: bool) -> dict[str, TerminationTermCfg]:
    terms = {"time_out": TerminationTermCfg(func=term_terms.time_out, time_out=True)}
    if term_cfg.get("base_contact"):
        terms["base_contact"] = TerminationTermCfg(func=term_terms.base_contact)
    if term_cfg.get("bad_orientation_rad") is not None:
        terms["bad_orientation"] = TerminationTermCfg(
            func=term_terms.bad_orientation, params={"limit_rad": float(term_cfg["bad_orientation_rad"])})
    if generator_terrain:  # walking off the generated grid is a time-out, not a failure
        terms["out_of_terrain_bounds"] = TerminationTermCfg(
            func=velocity_mdp.out_of_terrain_bounds, time_out=True)
    return terms


def _sensors_needed(task: dict) -> set[str]:
    needed = set()
    for group in ("actor", "critic"):
        for name in task["observations"][group]:
            if name in obs_terms.NEEDS_SENSOR:
                needed.add(obs_terms.NEEDS_SENSOR[name])
    for name, weight in task["rewards"].items():
        if not isinstance(weight, dict) and float(weight) != 0.0 and name in reward_terms.NEEDS_SENSOR:
            needed.add(reward_terms.NEEDS_SENSOR[name])
    if task.get("terminations", {}).get("base_contact"):
        needed.add("base_contact")
    return needed


# ---- assembly ---------------------------------------------------------------------

def make_env_cfg(robot, task: dict, num_envs: int, play: bool = False,
                 seed: int | None = None) -> ManagerBasedRlEnvCfg:
    sim = task["sim"]
    decimation = int(sim["decimation"])
    generator = uses_generator(task["terrain"])
    observations = make_observations(task["observations"], robot)
    if play:
        observations["actor"].enable_corruption = False
    curriculum = {}
    if generator and not play:
        promote = bool((task.get("curriculum") or {}).get("promote_requires_timeout", False))
        curriculum["terrain_levels"] = CurriculumTermCfg(
            func=terrain_levels_vel,
            params={"command_name": COMMAND_NAME, "promote_requires_timeout": promote})
    rewards = make_rewards(task, robot)
    if not play:                                # play/evaluation uses the full weights
        curriculum.update(make_reward_ramps(task, rewards))

    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=make_terrain(task["terrain"], play=play),
            entities={"robot": make_robot_entity(robot, decimation)},
            sensors=make_sensors(robot, _sensors_needed(task)),
            num_envs=num_envs,
            extent=2.0,
        ),
        observations=observations,
        actions={"joint_pos": make_action(robot, task["action"])},
        commands={COMMAND_NAME: make_velocity_command(task["commands"])},
        events=make_events(robot, play=play),
        rewards=rewards,
        terminations=make_terminations(task.get("terminations") or {}, generator),
        curriculum=curriculum,
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name=robot.base_body,
            distance=1.8,
            elevation=-10.0,
            azimuth=90.0,
        ),
        sim=SimulationCfg(
            nconmax=35 if generator else None,
            njmax=1500 if generator else 300,
            contact_sensor_maxmatch=500 if generator else 64,
            mujoco=MujocoCfg(
                timestep=float(sim["physics_dt"]),
                iterations=10,
                ls_iterations=20,
                ccd_iterations=500 if generator else 50,
                impratio=10,
                cone="elliptic",
            ),
        ),
        decimation=decimation,
        episode_length_s=1e9 if play else float(sim["episode_length_s"]),
        seed=seed,
    )


def make_env(robot, task: dict, num_envs: int, device: str, play: bool = False,
             seed: int | None = None, render_mode: str | None = None) -> QuadrupedEnv:
    cfg = make_env_cfg(robot, task, num_envs, play=play, seed=seed)
    env = QuadrupedEnv(cfg, device=device, render_mode=render_mode,
                       clip_reward_at_zero=bool(task.get("clip_reward_at_zero", False)))
    model_joints = list(env.scene["robot"].joint_names)
    if model_joints != list(robot.joint_names):
        raise ValueError(f"{robot.name}: joint_names in the YAML must match the model's order.\n"
                         f"  YAML : {robot.joint_names}\n  model: {model_joints}")
    return env
