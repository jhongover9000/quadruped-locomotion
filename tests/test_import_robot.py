"""scripts/import_robot.py on a URDF written to be awkward: its own link and joint names,
right-side axes negated, hind-leg frames turned half a turn, a bent zero pose, joints that
are not in one plane, feet on fixed links, a mesh behind a package:// path."""

import importlib.util
import math
import re
from pathlib import Path

import mujoco
import numpy as np
import pytest
import yaml

from quadruped_rl.robots import urdf_import as ui
from quadruped_rl.robots.config import robot_from_dict

REPO = Path(__file__).resolve().parents[1]
THIGH, CALF, FOOT_R = 0.2, 0.21, 0.02
MASS = 6.0 + 0.05 + 4 * (0.5 + 0.8 + 0.15 + 0.03)


def _stl_box(path: Path, centre, half) -> None:
    """ASCII STL of a box that is not centred on the mesh origin."""
    c, h = np.array(centre), np.array(half)
    v = [c + h * np.array(s) for s in [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
                                       (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]]
    faces = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4),
             (2, 3, 7), (2, 7, 6), (1, 2, 6), (1, 6, 5), (0, 4, 7), (0, 7, 3)]
    out = ["solid box"]
    for f in faces:
        out += ["facet normal 0 0 0", "outer loop"] + [f"vertex {v[i][0]} {v[i][1]} {v[i][2]}" for i in f] + ["endloop", "endfacet"]
    path.write_text("\n".join(out + ["endsolid box"]))


def write_urdf(folder: Path, joints_per_leg: int = 3, feet: bool = True) -> Path:
    (folder / "meshes").mkdir(parents=True)
    _stl_box(folder / "meshes" / "trunk.stl", (0.1, 0.0, 0.02), (0.15, 0.07, 0.04))

    def inertial(m, xyz="0 0 0"):
        i = 0.4 * m * 0.05**2
        return f'<inertial><origin xyz="{xyz}"/><mass value="{m}"/><inertia ixx="{i}" iyy="{2 * i}" izz="{1.5 * i}" ixy="0" ixz="0" iyz="0"/></inertial>'

    links = [f"""<link name="base_link">{inertial(6.0, "0.01 0.002 0.01")}
      <visual><origin xyz="0.02 0 0" rpy="0 0 0.3"/><geometry><mesh filename="package://awkward_description/meshes/trunk.stl"/></geometry></visual>
      <collision><geometry><box size="0.4 0.16 0.1"/></geometry></collision></link>
    <link name="imu_link">{inertial(0.05)}</link>
    <joint name="imu_fixed" type="fixed"><parent link="base_link"/><child link="imu_link"/><origin xyz="0.05 0 0.03"/></joint>"""]
    for leg, front, side in (("lf", 1, 1), ("rf", 1, -1), ("lh", -1, 1), ("rh", -1, -1)):
        flip = front                      # hind-leg frames are turned half a turn about z
        yaw = 0 if front > 0 else math.pi
        ax = "1 0 0" if side > 0 else "-1 0 0"
        ay = "0 1 0" if side > 0 else "0 -1 0"
        lim = '<limit lower="-3" upper="3" effort="25" velocity="20"/><dynamics damping="0.02" friction="0.1"/>'
        links.append(f"""
    <link name="{leg}_hip">{inertial(0.5)}<collision><origin rpy="0 1.5708 0"/><geometry><cylinder radius="0.03" length="0.05"/></geometry></collision></link>
    <joint name="{leg}_haa" type="revolute"><parent link="base_link"/><child link="{leg}_hip"/>
      <origin xyz="{0.2 * front} {0.06 * side} 0" rpy="0 0 {yaw}"/><axis xyz="{ax}"/>
      <limit lower="-0.8" upper="0.8" effort="25" velocity="20"/><dynamics damping="0.02" friction="0.1"/></joint>
    <link name="{leg}_thigh">{inertial(0.8, "0 0 -0.08")}
      <visual><geometry><mesh filename="package://awkward_description/meshes/thigh.dae"/></geometry></visual>
      <collision><origin xyz="0 0 -0.1"/><geometry><box size="0.03 0.03 0.18"/></geometry></collision></link>
    <joint name="{leg}_hfe" type="revolute"><parent link="{leg}_hip"/><child link="{leg}_thigh"/>
      <origin xyz="{0.03 * flip} {0.05 * side * flip} 0" rpy="0 0.4 0"/><axis xyz="{ay}"/>{lim}</joint>""")
        if joints_per_leg < 3:
            continue
        links.append(f"""
    <link name="{leg}_calf">{inertial(0.15, "0 0 -0.1")}<collision><origin xyz="0 0 -0.1"/><geometry><cylinder radius="0.012" length="0.16"/></geometry></collision></link>
    <joint name="{leg}_kfe" type="revolute"><parent link="{leg}_thigh"/><child link="{leg}_calf"/>
      <origin xyz="0 0 {-THIGH}" rpy="0 -0.9 0"/><axis xyz="{ay}"/>{lim}</joint>""")
        if feet:
            links.append(f"""
    <link name="{leg}_foot">{inertial(0.03)}<collision><geometry><sphere radius="{FOOT_R}"/></geometry></collision></link>
    <joint name="{leg}_foot_fixed" type="fixed"><parent link="{leg}_calf"/><child link="{leg}_foot"/><origin xyz="0 {0.01 * side * flip} {-CALF}"/></joint>""")
    urdf = folder / "awkward.urdf"
    urdf.write_text('<?xml version="1.0"?>\n<robot name="awkward">' + "".join(links) + "\n</robot>\n")
    return urdf


@pytest.fixture(scope="module")
def imported(tmp_path_factory):
    folder = tmp_path_factory.mktemp("awkward")
    robot = ui.import_urdf(write_urdf(folder / "src"), "awkward", folder / "out" / "meshes")
    return robot, folder / "out" / "meshes"


def test_same_robot_as_the_urdf(imported):
    robot, meshes = imported
    assert ui.verify_against_urdf(robot, meshes, samples=30) < 1e-9
    assert robot.total_mass == pytest.approx(MASS, abs=1e-9)


def test_legs_are_found_and_measured(imported):
    robot, _ = imported
    urdf_leg = {"FR": "rf", "FL": "lf", "RR": "rh", "RL": "lh"}
    for leg in ui.LEGS:
        assert [robot.joints[f"{leg}_{p}_joint"].urdf_name for p in ui.PARTS] == \
            [f"{urdf_leg[leg]}_{j}" for j in ("haa", "hfe", "kfe")]
        d, l1, l2 = robot.leg_lengths(leg)
        assert d == pytest.approx(0.06 if leg[1] == "L" else -0.06, abs=1e-9)
        assert (l1, l2) == pytest.approx((THIGH, CALF), abs=1e-9)
    assert robot.foot_radius == pytest.approx(FOOT_R)
    assert not robot.retarget_problems


def test_hardware_map(imported):
    robot, _ = imported
    j = robot.joints
    # Left fore leg: same direction as the URDF; its zero is 0.4 rad (thigh) and -0.9 rad
    # (calf) away from the straight leg.
    assert (j["FL_hip_joint"].sign, j["FL_thigh_joint"].sign, j["FL_calf_joint"].sign) == (1, 1, 1)
    assert j["FL_thigh_joint"].offset == pytest.approx(-0.4)
    assert j["FL_calf_joint"].offset == pytest.approx(0.9)
    # Right side: axes negated in the URDF. Hind legs: frame turned half a turn, which
    # negates both again for the left side.
    assert (j["FR_hip_joint"].sign, j["FR_thigh_joint"].sign) == (-1, -1)
    assert (j["RL_hip_joint"].sign, j["RL_thigh_joint"].sign) == (-1, -1)
    assert (j["RR_hip_joint"].sign, j["RR_thigh_joint"].sign) == (1, 1)
    # Limits follow: URDF [-3, 3] around a shifted zero.
    assert j["FL_thigh_joint"].range == pytest.approx((-2.6, 3.4))
    assert j["FR_hip_joint"].range == pytest.approx((-0.8, 0.8))
    assert j["FL_hip_joint"].effort == 25 and j["FL_hip_joint"].velocity == 20
    assert j["RR_calf_joint"].frictionloss == pytest.approx(0.1)


def test_model_follows_the_project_layout(imported):
    """What the tasks and SuperMimic's retargeting (quadmotion.robot.load_robot) rely on."""
    robot, meshes = imported
    m = ui.compile_mjcf(robot, meshes)
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)]
    assert names == ["floating_base_joint", *ui.JOINT_NAMES]
    assert m.nbody == 14 and m.nu == 0
    d = mujoco.MjData(m)                          # initial state: feet clear of a floor at z = 0
    mujoco.mj_kinematics(m, d)
    assert 0 < d.geom_xpos[:, 2].min() - FOOT_R < 0.03
    for leg in ui.LEGS:
        ja, jt, jc = (mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{leg}_{p}_joint") for p in ui.PARTS)
        for jid, axis in ((ja, (1, 0, 0)), (jt, (0, 1, 0)), (jc, (0, 1, 0))):
            assert np.allclose(m.jnt_axis[jid], axis) and np.allclose(m.jnt_pos[jid], 0)
        thigh_b, calf_b = int(m.jnt_bodyid[jt]), int(m.jnt_bodyid[jc])
        site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, leg)
        assert m.site_bodyid[site] == calf_b
        assert np.allclose(m.body_pos[thigh_b][[0, 2]], 0)
        assert np.allclose(m.body_pos[calf_b][:2], 0)
        assert np.allclose(m.site_pos[site][:2], 0)
        foot = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{leg}_foot_collision")
        assert m.geom_type[foot] == mujoco.mjtGeom.mjGEOM_SPHERE and m.geom_bodyid[foot] == calf_b
    trunk = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "trunk_collision")
    assert m.geom_type[trunk] == mujoco.mjtGeom.mjGEOM_BOX
    assert np.allclose(m.geom_size[trunk], (0.2, 0.08, 0.05))
    assert np.all(np.linalg.eigvalsh(np.diag(m.body_inertia[1])) > 0)


def test_mesh_keeps_its_place(imported):
    """The visual mesh (off-centre, behind a package:// path) sits where the URDF put it."""
    robot, meshes = imported
    assert (meshes / "trunk.stl").exists()
    assert any("thigh.dae" in w for w in robot.warnings)       # unreadable format: dropped, said so
    assert any("drawn with their collision shapes" in w for w in robot.warnings)
    src, new = robot.source["model"], ui.compile_mjcf(robot, meshes)
    ds, dn = mujoco.MjData(src), mujoco.MjData(new)
    dn.qpos[:7] = (0, 0, 0, 1, 0, 0, 0)        # the model itself starts above the floor
    mujoco.mj_kinematics(src, ds)
    mujoco.mj_kinematics(new, dn)
    gs = next(g for g in range(src.ngeom) if src.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH)
    gn = next(g for g in range(new.ngeom) if new.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH)
    # Legs have no visual mesh left: each collision shape and foot is also drawn.
    visual = [g for g in range(new.ngeom) if new.geom_group[g] == 2]
    assert len(visual) == 1 + 4 * 4 and all(new.geom_contype[g] == 0 for g in visual)
    assert np.allclose(ds.geom_xpos[gs], dn.geom_xpos[gn], atol=1e-9)
    assert np.allclose(ds.geom_xmat[gs], dn.geom_xmat[gn], atol=1e-9)
    assert np.allclose(ds.geom_xpos[gs], [0.02 + 0.1 * math.cos(0.3), 0.1 * math.sin(0.3), 0.02], atol=1e-6)


def test_measurements_replace_cad_values(tmp_path):
    robot = ui.import_urdf(write_urdf(tmp_path / "src"), "awkward", tmp_path / "meshes")
    notes = ui.apply_measurements(robot, {
        "total_mass": 13.0, "link_mass": {"calf": 0.25},
        "joint": {"rotor_inertia": [1e-4, 1e-4, 1e-4], "gear_ratio": [6, 6, 9], "frictionloss": [0.2, 0.2, 0.3]}})
    q0 = ui.stance_vector(robot, ui.default_stance(robot))
    notes.append(ui.match_com(robot, tmp_path / "meshes", q0, [0.012, -0.004, None]))
    assert len(notes) == 5
    m = ui.compile_mjcf(robot, tmp_path / "meshes")
    d = mujoco.MjData(m)
    d.qpos[:7], d.qpos[7:] = (0, 0, 0, 1, 0, 0, 0), q0
    mujoco.mj_forward(m, d)
    assert m.body_mass.sum() == pytest.approx(13.0)
    assert d.subtree_com[1][:2] == pytest.approx([0.012, -0.004], abs=1e-9)
    calf = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "RL_calf")
    assert m.body_mass[calf] == pytest.approx(0.25)
    assert robot.joints["FR_calf_joint"].armature == pytest.approx(81e-4)
    assert robot.joints["FR_calf_joint"].frictionloss == pytest.approx(0.3)
    with pytest.raises(ui.ImportError_, match="unknown keys"):
        ui.apply_measurements(robot, {"mass": 1})


def test_default_stance_and_config(imported):
    robot, meshes = imported
    spec = importlib.util.spec_from_file_location("import_robot", REPO / "scripts" / "import_robot.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    thigh, calf = ui.default_stance(robot)
    q0 = ui.stance_vector(robot, (thigh, calf))
    height = ui.standing_height(ui.compile_mjcf(robot, meshes), robot, q0)
    assert height == pytest.approx(0.62 * (THIGH + CALF) + FOOT_R, abs=1e-6)
    assert thigh > 0 > calf
    args = cli.argparse.Namespace(kp=[30, 30, 30], kd=[1, 1, 1], armature=None, torque_limit=None,
                                  velocity_limit=None, natural_frequency=10.0, damping_ratio=2.0)
    act, _ = cli.resolve_actuator(robot, args)
    text = cli.robot_yaml(robot, act, q0, height, "assets/robots/awkward/mjcf/awkward.xml", {}, "awkward.urdf")
    raw = yaml.safe_load(text)
    cfg = robot_from_dict(raw)
    assert cfg.joint_names == list(ui.JOINT_NAMES) and cfg.actuator.torque_limit == [25.0] * 12
    assert cfg.base_contact_geoms == ["trunk_collision"]
    assert raw["hardware"]["urdf_joint_names"][0] == "rf_haa" and raw["hardware"]["sign"][0] == -1
    drop = ui.drop_test(robot, meshes, q0, height, act["kp"], act["kd"], act["torque_limit"], act["armature"])
    assert drop["stands"] and drop["self_collisions"] == []
    with pytest.raises(ui.ImportError_, match="no PD gains"):
        cli.resolve_actuator(robot, cli.argparse.Namespace(**{**vars(args), "kp": None, "kd": None}))


def test_unsupported_urdfs_say_why(tmp_path):
    with pytest.raises(ui.ImportError_, match="expected 12 revolute joints"):
        ui.import_urdf(write_urdf(tmp_path / "a", joints_per_leg=2), "x", tmp_path / "m")
    with pytest.raises(ui.ImportError_, match="no foot found"):
        ui.import_urdf(write_urdf(tmp_path / "b", feet=False), "x", tmp_path / "m")
    with pytest.raises(ui.ImportError_, match="no such link"):
        ui.import_urdf(write_urdf(tmp_path / "c"), "x", tmp_path / "m", foot_links=["a", "b", "c", "d"])


def test_joint_range_option(tmp_path):
    """--joint-range gives a joint kind a range before compiling: here continuous knees."""
    urdf = write_urdf(tmp_path / "src")
    def continuous(knee: re.Match) -> str:              # no range, as a CAD export of a knee
        return knee.group(0).replace('type="revolute"', 'type="continuous"').replace('lower="-3" upper="3" ', "")

    urdf.write_text(re.sub(r'<joint name="\w+_kfe".*?</joint>', continuous, urdf.read_text(), flags=re.DOTALL))
    plain = ui.import_urdf(urdf, "awkward", tmp_path / "m1")
    assert not plain.joints["FL_calf_joint"].limited and plain.range_overrides == []
    assert any("'lf_kfe' has no limits" in w for w in plain.warnings)

    robot = ui.import_urdf(urdf, "awkward", tmp_path / "m2", joint_ranges={"calf": (-2.5, 0.5)})
    assert ui.verify_against_urdf(robot, tmp_path / "m2") < 1e-9
    for leg in ui.LEGS:
        j = robot.joints[f"{leg}_calf_joint"]
        assert j.limited and j.range == pytest.approx(sorted(j.sign * (r - j.offset) for r in (-2.5, 0.5)))
        for part in ("hip", "thigh"):
            assert robot.joints[f"{leg}_{part}_joint"].range == plain.joints[f"{leg}_{part}_joint"].range
    assert robot.joints["FL_calf_joint"].range == pytest.approx((-3.4, -0.4))      # offset 0.9, sign +1
    assert not any("no limits" in w for w in robot.warnings)
    assert len(robot.range_overrides) == 1
    note = robot.range_overrides[0]
    assert note.startswith("calf joints") and "lf_kfe" in note and "rh_kfe" in note and "continuous" in note

    spec = importlib.util.spec_from_file_location("import_robot", REPO / "scripts" / "import_robot.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    assert cli.parse_joint_ranges([["calf", "-2.5", "0.5"]]) == {"calf": (-2.5, 0.5)}
    assert cli.parse_joint_ranges([]) is None
    with pytest.raises(ui.ImportError_, match="given twice"):
        cli.parse_joint_ranges([["calf", "0", "1"], ["calf", "0", "2"]])
    q0 = [0.0] * 12          # only written out here (the URDF's legs differ, so the calf ranges do too)
    args = cli.argparse.Namespace(kp=[30, 30, 30], kd=[1, 1, 1], armature=None, torque_limit=None,
                                  velocity_limit=None, natural_frequency=10.0, damping_ratio=2.0)
    act, notes = cli.resolve_actuator(robot, args)
    text = cli.robot_yaml(robot, act, q0, 0.3, "assets/robots/awkward/mjcf/awkward.xml", {}, "awkward.urdf")
    assert f"#   {note}" in text                                   # a comment in the config ...
    robot_from_dict(yaml.safe_load(text))                          # ... that still loads
    drop = {"self_collisions": [], "stands": True, "height_start": 0.3, "height_end": 0.3, "tilt_deg": 0.0,
            "joint_sag_rad": 0.0, "hold_torque_fraction": 0.1}
    report = cli.report_md(robot, act, q0, 0.3, {"actuator": notes, "measurements": []}, drop, 0.0)
    assert f"- {note}" in report

    with pytest.raises(ui.ImportError_, match="not one of"):
        ui.import_urdf(urdf, "awkward", tmp_path / "m3", joint_ranges={"knee": (0.0, 1.0)})
    with pytest.raises(ui.ImportError_, match="not below"):
        ui.import_urdf(urdf, "awkward", tmp_path / "m3", joint_ranges={"calf": (1.0, 0.0)})


def test_collision_mesh_at_the_foot_becomes_the_foot_sphere(tmp_path):
    """A collision mesh at a foot is dropped for the foot sphere; one elsewhere still becomes a box."""
    urdf = write_urdf(tmp_path / "src")
    meshes = tmp_path / "src" / "meshes"
    _stl_box(meshes / "foot_col.stl", (0.0, 0.0, 0.0), (0.015, 0.015, 0.015))   # corners 26 mm < 2 x 20 mm
    _stl_box(meshes / "thigh_col.stl", (0.0, 0.0, -0.1), (0.015, 0.015, 0.08))
    mesh = '<collision><origin xyz="{}"/><geometry><mesh filename="package://awkward_description/meshes/{}"/></geometry></collision></link>'
    text = re.sub(r'(<link name="\w+_calf">.*?)</link>', lambda m: m.group(1) + mesh.format(f"0 0 {-CALF}", "foot_col.stl"),
                  urdf.read_text(), flags=re.DOTALL)
    text = re.sub(r'(<link name="\w+_thigh">.*?)</link>', lambda m: m.group(1) + mesh.format("0 0 0", "thigh_col.stl"),
                  text, flags=re.DOTALL)
    urdf.write_text(text)
    robot = ui.import_urdf(urdf, "awkward", tmp_path / "out")
    assert ui.verify_against_urdf(robot, tmp_path / "out") < 1e-9
    foot_lines = [w for w in robot.warnings if "collision mesh at the foot replaced by the foot sphere (r 0.020 m)" in w]
    box_lines = [w for w in robot.warnings if "mesh collision replaced by its bounding box" in w]
    assert len(foot_lines) == 4 and all("_calf'" in w for w in foot_lines)
    assert len(box_lines) == 4 and all("_thigh'" in w for w in box_lines)
    m = ui.compile_mjcf(robot, tmp_path / "out")
    for leg in ui.LEGS:
        assert [g.kind for g in robot.legs[leg]["calf"].geoms if g.collision] == ["cylinder"]
        assert sorted(g.kind for g in robot.legs[leg]["thigh"].geoms if g.collision) == ["box", "box"]
        calf = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_calf")
        cols = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in range(m.ngeom)
                if m.geom_bodyid[g] == calf and m.geom_contype[g]]
        assert sorted(cols) == sorted([f"{leg}_calf_collision", f"{leg}_foot_collision"])


def test_supermimic_files_follow_the_config(imported, tmp_path):
    """The generated SuperMimic spec and retarget config carry the robot YAML's numbers."""
    from quadruped_rl.robots import supermimic_port as port

    robot, meshes = imported
    spec = importlib.util.spec_from_file_location("import_robot", REPO / "scripts" / "import_robot.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    q0 = ui.stance_vector(robot, ui.default_stance(robot))
    height = ui.standing_height(ui.compile_mjcf(robot, meshes), robot, q0)
    xml = tmp_path / "awkward.xml"
    xml.write_text(ui.to_mjcf(robot, meshdir=str(meshes)))
    args = cli.argparse.Namespace(kp=[30, 30, 40], kd=[1, 1, 1.5], armature=[0.004, 0.004, 0.009],
                                  torque_limit=None, velocity_limit=None, natural_frequency=10.0, damping_ratio=2.0)
    act, _ = cli.resolve_actuator(robot, args)
    raw = yaml.safe_load(cli.robot_yaml(robot, act, q0, height, str(xml), {}, "awkward.urdf"))
    cfg = robot_from_dict(raw)
    model = port.check_layout(cfg)
    scales = port.action_scales(cfg, {"scale": 0.25, "scale_mode": "effort_relative"})
    assert scales[2] == pytest.approx(0.25 * 25 / 40)
    source = port.spec_source(cfg, model, scales, {})
    compile(source, "awkward.py", "exec")                       # valid Python
    assert '"FR_calf_joint": 40' in source and f"STAND_Z = {height:.4f}" in source
    assert "SCALED FROM THE GO1" in source
    measured = port.spec_source(cfg, model, scales, {"err_scale": [1, 2, 3, 4, 5, 6], "out_scale": [1] * 6, "in_scale": [2] * 6})
    assert "err_scale=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)" in measured and "SCALED FROM THE GO1" not in measured
    with pytest.raises(port.PortError, match="needs 6 values"):
        port.spec_source(cfg, model, scales, {"err_scale": [1, 2]})
    retarget = yaml.safe_load(port.retarget_yaml(cfg, "x.xml"))
    assert retarget["stance"] == pytest.approx([q0[1], q0[2]], abs=1e-4) and retarget["knee_sign"] == -1
    assert retarget["legs"]["RL"] == ["RL_hip_joint", "RL_thigh_joint", "RL_calf_joint", "RL"]
