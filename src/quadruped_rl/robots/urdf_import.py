"""URDF -> the robot model this project trains on (CLI: scripts/import_robot.py).

A URDF from CAD describes the same robot in whatever joint frames, zero pose, axis signs
and names the exporter chose. Training needs one fixed convention, the one the Go1
stand-in uses, so that tasks, the motion retargeting and SuperMimic work unchanged:

    bodies   trunk, <LEG>_hip, <LEG>_thigh, <LEG>_calf      LEG in FR, FL, RR, RL
    joints   <LEG>_hip_joint    about +x of the trunk        (abduction)
             <LEG>_thigh_joint  about +y                     (0 = thigh straight down)
             <LEG>_calf_joint   about +y                     (0 = calf in line with thigh)
    frames   every body frame is parallel to the trunk frame when all joints are 0
             (x forward, y left, z up), with the leg hanging straight down
    feet     a sphere geom <LEG>_foot_collision and a site <LEG> on the calf body

``import_urdf`` compiles the URDF with MuJoCo, finds the four legs, and rebuilds the model
in that convention. Masses, inertias, geometry and joint limits are carried over exactly;
the only things that change are names, frames and where each joint's zero is. That change
is recorded per joint as ``q_urdf = offset + sign * q_sim`` (the "hardware map"), which is
what the deployment code needs to talk to motors that still use the URDF convention.

The rebuilt model is checked against the URDF at random poses before anything is written.
"""

from __future__ import annotations

import hashlib
import math
import re
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

LEGS = ("FR", "FL", "RR", "RL")
PARTS = ("hip", "thigh", "calf")
JOINT_NAMES = tuple(f"{leg}_{part}_joint" for leg in LEGS for part in PARTS)
MESH_FORMATS = (".stl", ".obj", ".msh")       # what MuJoCo reads; .dae visuals are dropped
SNAP = 5e-4                                    # m: smaller frame offsets are rounded away
AXIS_TOL = 2e-3                                # rad-ish: how parallel a joint axis must be


class ImportError_(ValueError):
    """The URDF cannot be brought into the project's convention; the message says why."""


@dataclass
class Inertia:
    mass: float
    com: np.ndarray        # (3,) in the trunk frame at the straight-leg pose
    tensor: np.ndarray     # (3, 3) about the COM, trunk axes


@dataclass
class Geom:
    kind: str              # sphere | capsule | cylinder | box | mesh
    size: np.ndarray
    pos: np.ndarray        # trunk frame at the straight-leg pose
    mat: np.ndarray        # (3, 3)
    collision: bool
    mesh: str | None = None
    rgba: np.ndarray | None = None
    name: str | None = None


@dataclass
class Body:
    name: str
    origin: np.ndarray                         # trunk frame at the straight-leg pose
    inertia: Inertia
    geoms: list[Geom] = field(default_factory=list)
    urdf_links: list[str] = field(default_factory=list)
    draw_collisions: bool = False              # no visual geometry: show the collision shapes


@dataclass
class Joint:
    name: str
    urdf_name: str
    axis: tuple[int, int, int]
    sign: float            # q_urdf = offset + sign * q_sim
    offset: float
    range: tuple[float, float]
    limited: bool
    effort: float | None
    velocity: float | None
    damping: float
    frictionloss: float
    armature: float


@dataclass
class ImportedRobot:
    name: str
    trunk: Body
    legs: dict[str, dict[str, Body]]           # leg -> part -> Body
    joints: dict[str, Joint]                   # canonical joint name -> Joint
    foot_radius: float
    foot_pos: dict[str, np.ndarray]            # leg -> foot centre, trunk frame, straight pose
    meshes: list[dict]                         # {name, file, scale}
    imu_pos: np.ndarray
    warnings: list[str] = field(default_factory=list)
    retarget_problems: list[str] = field(default_factory=list)
    source: dict = field(default_factory=dict)  # compiled URDF model + bookkeeping for checks
    range_overrides: list[str] = field(default_factory=list)  # joint ranges set by the caller, not the URDF

    @property
    def bodies(self) -> list[Body]:
        return [self.trunk] + [self.legs[leg][p] for leg in LEGS for p in PARTS]

    def leg_lengths(self, leg: str) -> tuple[float, float, float]:
        """(sideways thigh offset, thigh length, calf length to the foot centre)."""
        b = self.legs[leg]
        return (float(b["thigh"].origin[1] - b["hip"].origin[1]),
                float(b["thigh"].origin[2] - b["calf"].origin[2]),
                float(b["calf"].origin[2] - self.foot_pos[leg][2]))

    @property
    def total_mass(self) -> float:
        return float(sum(b.inertia.mass for b in self.bodies))


# ---- reading the URDF -------------------------------------------------------------

def _find_mesh(filename: str, urdf: Path, mesh_dirs: list[Path]) -> Path | None:
    """Locate a mesh named in the URDF: package:// and file:// prefixes are dropped, then the
    path is tried against the URDF's folder and its parents, then searched for by file name."""
    rel = re.sub(r"^(package|file|model)://", "", filename).lstrip("/")
    parts = Path(rel).parts
    roots = [urdf.parent, *list(urdf.parent.parents)[:3], *mesh_dirs]
    if Path(filename).is_absolute() and Path(filename).exists():
        return Path(filename)
    for root in roots:
        for skip in range(len(parts)):            # also without the leading package name
            p = root.joinpath(*parts[skip:])
            if p.is_file():
                return p
    for root in [*mesh_dirs, urdf.parent, urdf.parent.parent]:
        hits = sorted(root.rglob(Path(rel).name)) if root.is_dir() else []
        if hits:
            return hits[0]
    return None


def _usable_mesh(path: Path | None) -> Path | None:
    """The mesh itself if MuJoCo reads its format, else a same-named file in a format it does."""
    if path is None:
        return None
    if path.suffix.lower() in MESH_FORMATS:
        return path
    for ext in MESH_FORMATS:
        for cand in (path.with_suffix(ext), path.with_suffix(ext.upper())):
            if cand.is_file():
                return cand
    return None


def _copy_mesh(src: Path, dst: Path) -> bool:
    """Copy a mesh; an ASCII STL is rewritten as binary. Returns True if it was converted."""
    raw = src.read_bytes()
    ascii_stl = src.suffix.lower() == ".stl" and raw[:5].lower() == b"solid" and b"facet" in raw[:2000]
    if not ascii_stl:
        if not dst.exists() or dst.read_bytes() != raw:
            shutil.copyfile(src, dst)
        return False
    verts = [[float(x) for x in line.split()[1:4]] for line in raw.decode("ascii", "replace").splitlines()
             if line.strip().lower().startswith("vertex")]
    tris = np.asarray(verts, dtype="<f4").reshape(-1, 3, 3)
    normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, lengths, out=np.zeros_like(normals), where=lengths > 0)
    record = np.zeros(len(tris), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    record["n"], record["v"] = normals, tris
    dst.write_bytes(b"binary STL written by quadruped_rl.robots.urdf_import".ljust(80, b" ")
                    + np.uint32(len(tris)).tobytes() + record.tobytes())
    return True


def _set_joint_ranges(root: ET.Element, joint_ranges: dict[str, tuple[float, float]]) -> list[str]:
    """Give every joint of a kind (hip, thigh, calf: the 1st, 2nd, 3rd moving joint below the
    root link) the range ``(lower, upper)`` in URDF radians, before MuJoCo compiles it. A
    continuous joint becomes revolute. Returns one line per kind, saying what was changed."""
    for kind, (lo, hi) in joint_ranges.items():
        if kind not in PARTS:
            raise ImportError_(f"--joint-range: kind '{kind}' is not one of {PARTS}")
        if not lo < hi:
            raise ImportError_(f"--joint-range {kind}: lower {lo} is not below upper {hi}")
    moving = {"revolute", "continuous", "prismatic"}
    parent_joint = {j.find("child").get("link"): j for j in root.iter("joint") if j.find("child") is not None}

    def depth(j: ET.Element) -> int:
        n, link, seen = 1, j.find("parent").get("link"), set()
        while link in parent_joint and link not in seen:      # (a cycle is MuJoCo's to reject)
            seen.add(link)
            up = parent_joint[link]
            n += up.get("type") in moving
            link = up.find("parent").get("link")
        return n

    by_kind: dict[str, list[ET.Element]] = {}
    for j in root.iter("joint"):
        if j.get("type") in moving and j.find("parent") is not None:
            d = depth(j)
            if 1 <= d <= len(PARTS):
                by_kind.setdefault(PARTS[d - 1], []).append(j)
    notes = []
    for kind, (lo, hi) in joint_ranges.items():
        joints = by_kind.get(kind)
        if not joints:
            raise ImportError_(f"--joint-range {kind}: the URDF has no {kind} joints")
        was = set()
        for j in joints:
            lim = j.find("limit")
            if lim is None:
                lim = ET.SubElement(j, "limit")
            was.add("continuous" if j.get("type") == "continuous"
                    else f"[{lim.get('lower', '?')}, {lim.get('upper', '?')}]")
            j.set("type", "revolute")
            lim.set("lower", repr(float(lo)))
            lim.set("upper", repr(float(hi)))
        notes.append(f"{kind} joints ({', '.join(j.get('name') for j in joints)}): range set to "
                     f"[{lo:.5g}, {hi:.5g}] rad in URDF angles by --joint-range (URDF: {', '.join(sorted(was))})")
    return notes


def _prepare_urdf(urdf: Path, mesh_dirs: list[Path], mesh_out: Path, warnings: list[str],
                  joint_ranges: dict[str, tuple[float, float]] | None = None):
    """Copy the meshes next to the model and write a URDF MuJoCo can compile as given:
    plain mesh file names, every link kept as its own body, visual geometry kept."""
    tree = ET.parse(urdf)
    root = tree.getroot()
    if root.tag != "robot":
        raise ImportError_(f"{urdf}: not a URDF (root element is <{root.tag}>, expected <robot>). "
                           "Convert xacro first:  xacro robot.urdf.xacro > robot.urdf")
    range_notes = _set_joint_ranges(root, joint_ranges) if joint_ranges else []
    for old in root.findall("mujoco"):
        root.remove(old)
    mesh_out.mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {}
    for link in root.iter("link"):
        for kind in ("visual", "collision"):
            for el in list(link.findall(kind)):
                mesh = el.find("geometry/mesh")
                if mesh is None:
                    continue
                found = _find_mesh(mesh.get("filename", ""), urdf, mesh_dirs)
                src = _usable_mesh(found)
                if src is None:
                    why = ("not found (pass --mesh-dir)" if found is None
                           else f"format {found.suffix} is not readable by MuJoCo; export .stl or .obj")
                    warnings.append(f"link '{link.get('name')}': {kind} mesh "
                                    f"'{mesh.get('filename')}' dropped: {why}")
                    link.remove(el)
                    continue
                digest = hashlib.sha1(src.read_bytes()).hexdigest()
                if copied.setdefault(src.name, digest) != digest:
                    raise ImportError_(f"two different mesh files are both named {src.name}; rename one")
                dst = mesh_out / src.name
                if _copy_mesh(src, dst):
                    warnings.append(f"mesh {src.name}: ASCII STL converted to binary (MuJoCo reads only binary)")
                mesh.set("filename", src.name)
    ET.SubElement(ET.SubElement(root, "mujoco"), "compiler", fusestatic="false",
                  discardvisual="false", strippath="true", balanceinertia="true",
                  meshdir=str(mesh_out.resolve()))
    limits = {}
    for j in root.iter("joint"):
        lim = j.find("limit")
        if lim is not None:
            limits[j.get("name")] = (float(lim.get("effort", 0) or 0), float(lim.get("velocity", 0) or 0))
    return ET.tostring(root, encoding="unicode"), limits, range_notes


def _name(model, kind, i) -> str:
    return mujoco.mj_id2name(model, kind, i) or ""


def _angle_about(axis: int, v: np.ndarray) -> float:
    """Angle of v in the plane normal to the given trunk axis (x: from +y towards +z;
    y: the leg angle, 0 = pointing down, positive = swung backwards, as in the Go1)."""
    if axis == 0:
        return math.atan2(v[2], v[1])
    return math.atan2(-v[0], -v[2])


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


# ---- the import -------------------------------------------------------------------

def import_urdf(urdf: Path, name: str, mesh_out: Path, *, mesh_dirs: list[Path] | None = None,
                foot_links: list[str] | None = None, foot_radius: float | None = None,
                joint_ranges: dict[str, tuple[float, float]] | None = None) -> ImportedRobot:
    """Compile the URDF and rebuild it in the project's convention (see module docstring).

    ``foot_links``: the four URDF links at the feet, any order; found by name (foot / toe)
    when omitted. ``foot_radius``: needed when the URDF's feet have no sphere collision.
    ``joint_ranges``: {hip|thigh|calf: (lower, upper)} in URDF radians, replacing the URDF's
    range for every joint of that kind (e.g. a continuous knee) before it is compiled.
    """
    urdf = Path(urdf).resolve()
    warnings: list[str] = []
    text, urdf_limits, range_notes = _prepare_urdf(urdf, [Path(d) for d in (mesh_dirs or [])], Path(mesh_out),
                                                   warnings, joint_ranges)
    try:
        spec = mujoco.MjSpec.from_string(text)
        model = spec.compile()
    except Exception as e:  # MuJoCo's own message names the element
        raise ImportError_(f"MuJoCo could not compile {urdf.name}: {e}") from e
    data = mujoco.MjData(model)
    B, J = mujoco.mjtObj.mjOBJ_BODY, mujoco.mjtObj.mjOBJ_JOINT

    # -- structure: one root link, 12 hinges in four chains of three
    roots = [b for b in range(1, model.nbody) if model.body_parentid[b] == 0]
    if len(roots) != 1:
        raise ImportError_(f"expected one root link, found {[_name(model, B, b) for b in roots]}")
    root = roots[0]
    kinds = {int(t) for t in model.jnt_type}
    if kinds - {int(mujoco.mjtJoint.mjJNT_HINGE)}:
        bad = [_name(model, J, j) for j in range(model.njnt) if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE]
        raise ImportError_(f"only revolute joints are supported; these are not: {bad}. "
                           "Make them 'fixed' in the URDF (and remove a floating base joint).")
    if model.njnt != 12:
        names = [_name(model, J, j) for j in range(model.njnt)]
        raise ImportError_(f"expected 12 revolute joints (4 legs x hip, thigh, calf), found {model.njnt}: "
                           f"{names}. Make any other joint 'fixed' in the URDF.")
    joint_of = {int(model.jnt_bodyid[j]): j for j in range(model.njnt)}
    if len(joint_of) != 12:
        raise ImportError_("a link has more than one joint; each leg link needs exactly one")

    mover = {}                                   # body -> the moving body it is welded to
    for b in range(1, model.nbody):
        mover[b] = b if (b in joint_of or b == root) else mover[int(model.body_parentid[b])]

    def children(parent_body: int) -> list[int]:
        return [b for b in joint_of if mover[int(model.body_parentid[b])] == parent_body]

    chains = []
    for hip in children(root):
        chain = [hip]
        while True:
            nxt = children(chain[-1])
            if not nxt:
                break
            if len(nxt) > 1:
                raise ImportError_(f"link '{_name(model, B, chain[-1])}' has {len(nxt)} child joints; "
                                   "a leg must be a single chain")
            chain.append(nxt[0])
        chains.append(chain)
    if len(chains) != 4 or any(len(c) != 3 for c in chains):
        shape = [[_name(model, J, joint_of[b]) for b in c] for c in chains]
        raise ImportError_(f"expected 4 legs of 3 joints from the root link; found {shape}")

    # -- which leg is which: hip position relative to the centre of the four hips
    mujoco.mj_kinematics(model, data)
    hips = np.array([data.xanchor[joint_of[c[0]]] for c in chains])
    centre = hips.mean(axis=0)
    labelled = {}
    for c, p in zip(chains, hips):
        label = ("F" if p[0] > centre[0] else "R") + ("L" if p[1] > centre[1] else "R")
        if label in labelled:
            raise ImportError_(
                "could not tell the legs apart from the hip positions "
                f"{np.round(hips, 3).tolist()}: the root link must have x forward, y left, z up")
        labelled[label] = c
    if abs(np.ptp(hips[:, 0])) < 2 * abs(np.ptp(hips[:, 2])) or abs(np.ptp(hips[:, 1])) < 1e-3:
        warnings.append("hip positions are unusual for x-forward / y-left / z-up; check the "
                        "leg labels in the report")

    # -- feet
    def welded(body: int) -> list[int]:
        return [b for b in range(1, model.nbody) if mover[b] == body]

    def is_collision(g: int) -> bool:
        return bool(model.geom_contype[g] or model.geom_conaffinity[g])

    wanted = set(foot_links or [])
    unknown = wanted - {_name(model, B, b) for b in range(1, model.nbody)}
    if unknown:
        raise ImportError_(f"--foot-links: no such link(s) in the URDF: {sorted(unknown)}")
    foot_body, foot_geom = {}, {}
    for leg, chain in labelled.items():
        calf = chain[2]
        extra = [b for b in welded(calf) if b != calf]
        if wanted:
            cands = [b for b in extra if _name(model, B, b) in wanted]
        else:
            cands = [b for b in extra if re.search(r"foot|toe|paw", _name(model, B, b), re.IGNORECASE)] or \
                    (extra if len(extra) == 1 else [])
        spheres = [g for g in range(model.ngeom) if is_collision(g)
                   and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE
                   and int(model.geom_bodyid[g]) in (cands or welded(calf))]
        if len(cands) > 1:
            raise ImportError_(f"{leg}: several candidate foot links {[_name(model, B, b) for b in cands]}; "
                               "name the four feet with --foot-links")
        if not cands and not spheres:
            raise ImportError_(
                f"{leg}: no foot found below '{_name(model, B, calf)}'. Add a link at the centre of the "
                "foot, attached to the calf link with a fixed joint, and name it with --foot-links.")
        foot_body[leg] = cands[0] if cands else None
        if spheres:                               # the sphere farthest from the knee is the foot
            knee = data.xanchor[joint_of[calf]]
            foot_geom[leg] = max(spheres, key=lambda g: np.linalg.norm(data.geom_xpos[g] - knee))
    if len(foot_geom) == 4:
        radii = [float(model.geom_size[g][0]) for g in foot_geom.values()]
        radius = float(np.mean(radii))
        if np.ptp(radii) > 1e-4:
            warnings.append(f"foot sphere radii differ {np.round(radii, 4).tolist()}; using the mean")
        if foot_radius is not None and abs(foot_radius - radius) > 1e-4:
            warnings.append(f"--foot-radius {foot_radius} overrides the URDF's foot spheres ({radius:.4f} m)")
            radius = float(foot_radius)
    elif foot_radius is not None:
        radius, foot_geom = float(foot_radius), {}
    else:
        raise ImportError_("the feet have no sphere collision in the URDF, so the foot radius is "
                           "unknown: pass --foot-radius (metres)")

    def foot_point(leg: str) -> np.ndarray:
        if leg in foot_geom:
            return data.geom_xpos[foot_geom[leg]].copy()
        return data.xpos[foot_body[leg]].copy()

    # -- the straight-leg pose: where every joint's new zero is, and each axis sign
    q = np.zeros(model.nq)
    sign = {}

    def kin():
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)

    def axis_sign(j: int, axis: int, what: str) -> float:
        a = data.xaxis[j]
        if abs(abs(a[axis]) - 1.0) > AXIS_TOL:
            raise ImportError_(
                f"joint '{_name(model, J, j)}' ({what}) turns about {np.round(a, 3).tolist()} in the root "
                f"frame; expected the {'xyz'[axis]} axis. Supported legs: hip about x (forward), thigh "
                "and calf about y (sideways), root link x forward / z up.")
        return float(np.sign(a[axis]))

    for _ in range(3):
        for leg, (hip, thigh, calf) in labelled.items():
            ja, jt, jc = joint_of[hip], joint_of[thigh], joint_of[calf]
            kin()
            s = axis_sign(ja, 0, f"{leg} hip")
            at = data.xaxis[jt]
            if abs(at[0]) > AXIS_TOL:
                raise ImportError_(f"joint '{_name(model, J, jt)}' is not perpendicular to the hip axis")
            tilt = _angle_about(0, at)            # thigh axis must end up along +-y
            phi = min((_wrap(-tilt), _wrap(math.pi - tilt)), key=abs)
            q[model.jnt_qposadr[ja]] += s * phi
            kin()
            sign[ja] = s
            sign[jt] = axis_sign(jt, 1, f"{leg} thigh")
            v = data.xanchor[jc] - data.xanchor[jt]
            if math.hypot(v[0], v[2]) < 1e-4:
                raise ImportError_(f"{leg}: thigh and calf joints coincide (thigh length 0)")
            q[model.jnt_qposadr[jt]] += sign[jt] * -_angle_about(1, v)
            kin()
            sign[jc] = axis_sign(jc, 1, f"{leg} calf")
            w = foot_point(leg) - data.xanchor[jc]
            if math.hypot(w[0], w[2]) < 1e-4:
                raise ImportError_(f"{leg}: the foot is at the calf joint (calf length 0)")
            q[model.jnt_qposadr[jc]] += sign[jc] * -_angle_about(1, w)
    kin()
    q_straight = q.copy()

    # -- body frames. A hinge may be anchored anywhere on its axis, so the anchors slide along
    #    their axes until the leg lies in one plane: thigh, calf and foot share x and y.
    retarget_problems: list[str] = []
    origins, foot_pos = {root: np.zeros(3)}, {}
    root_pos, root_mat = data.xpos[root].copy(), data.xmat[root].reshape(3, 3).copy()
    if not np.allclose(root_mat, np.eye(3), atol=1e-9) or np.abs(root_pos).max() > 1e-9:
        raise ImportError_("the URDF root link is not at the world origin")   # MuJoCo always places it there
    for leg, (hip, thigh, calf) in labelled.items():
        pa, pt, pc = (data.xanchor[joint_of[b]] for b in (hip, thigh, calf))
        pf = foot_point(leg)
        x, y = float(pt[0]), float(pf[1])
        dz = float(pt[2] - pa[2])
        if abs(dz) > SNAP:
            retarget_problems.append(
                f"{leg}: the thigh axis passes {dz * 1000:+.1f} mm above the hip axis; the motion "
                "retargeting assumes they intersect")
        elif abs(dz) > 1e-7:
            warnings.append(f"{leg}: thigh axis moved {dz * 1e3:+.3f} mm in z to meet the hip axis")
        zt = float(pa[2]) if abs(dz) <= SNAP else float(pt[2])
        origins[hip] = np.array([x, pa[1], pa[2]])
        origins[thigh] = np.array([x, y, zt])
        origins[calf] = np.array([x, y, pc[2] - (pt[2] - zt)])
        foot_pos[leg] = np.array([x, y, pf[2] - (pt[2] - zt)])
        for label, p in (("calf joint", pc), ("foot", pf)):
            if abs(p[0] - x) > SNAP:
                raise ImportError_(f"{leg}: {label} is {abs(p[0] - x) * 1e3:.1f} mm out of the leg plane")

    # -- mass properties and geometry of everything welded to each moving body
    def inertia_of(body: int) -> Inertia:
        parts = [b for b in welded(body) if model.body_mass[b] > 0]
        m = float(sum(model.body_mass[b] for b in parts))
        if m < 1e-9:
            raise ImportError_(f"link '{_name(model, B, body)}' has no mass; every moving link needs an <inertial>")
        com = sum(model.body_mass[b] * data.xipos[b] for b in parts) / m
        tensor = np.zeros((3, 3))
        for b in parts:
            R = data.ximat[b].reshape(3, 3)
            d = data.xipos[b] - com
            tensor += R @ np.diag(model.body_inertia[b]) @ R.T + model.body_mass[b] * (d @ d * np.eye(3) - np.outer(d, d))
        return Inertia(m, np.array(com), tensor)

    # A collision mesh at a foot (on the calf or the foot link welded to it, every vertex within
    # two foot radii of the foot centre) is the foot: the writer's sphere replaces it.
    foot_meshes = set()
    for leg, chain in labelled.items():
        for g in range(model.ngeom):
            if not (is_collision(g) and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
                    and mover[int(model.geom_bodyid[g])] == chain[2]):
                continue
            mid = int(model.geom_dataid[g])
            v = model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid] + model.mesh_vertnum[mid]]
            world_v = data.geom_xpos[g] + v @ data.geom_xmat[g].reshape(3, 3).T
            if np.linalg.norm(world_v - foot_point(leg), axis=1).max() <= 2 * radius:
                foot_meshes.add(g)
                warnings.append(f"link '{_name(model, B, int(model.geom_bodyid[g]))}': collision mesh at the "
                                f"foot replaced by the foot sphere (r {radius:.3f} m)")

    kind_name = {int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere", int(mujoco.mjtGeom.mjGEOM_CAPSULE): "capsule",
                 int(mujoco.mjtGeom.mjGEOM_CYLINDER): "cylinder", int(mujoco.mjtGeom.mjGEOM_BOX): "box",
                 int(mujoco.mjtGeom.mjGEOM_MESH): "mesh"}

    def geoms_of(body: int, skip_bodies: set[int]) -> list[Geom]:
        out = []
        for g in range(model.ngeom):
            b = int(model.geom_bodyid[g])
            if mover[b] != body if b else True:
                continue
            col = is_collision(g)
            if col and (b in skip_bodies or g in foot_geom.values() or g in foot_meshes):
                continue                          # the foot gets one sphere, added by the writer
            kind = kind_name.get(int(model.geom_type[g]))
            if kind is None:
                warnings.append(f"geom {g} on '{_name(model, B, b)}': unsupported shape, dropped")
                continue
            pos, mat = data.geom_xpos[g].copy(), data.geom_xmat[g].reshape(3, 3).copy()
            size, mesh = model.geom_size[g].copy(), None
            if kind == "mesh":
                mid = int(model.geom_dataid[g])
                if col:                            # mesh collisions become their bounding box
                    centre, half = model.geom_aabb[g][:3], model.geom_aabb[g][3:]
                    kind, size, pos = "box", half.copy(), pos + mat @ centre
                    warnings.append(f"link '{_name(model, B, b)}': mesh collision replaced by its "
                                    f"bounding box {np.round(2 * half, 3).tolist()} m")
                else:                              # undo the re-centring MuJoCo applied to the mesh
                    mq = np.zeros(9)
                    mujoco.mju_quat2Mat(mq, model.mesh_quat[mid])
                    mat = mat @ mq.reshape(3, 3).T
                    pos = pos - mat @ model.mesh_pos[mid]
                    mesh = _name(model, mujoco.mjtObj.mjOBJ_MESH, mid)
            out.append(Geom(kind, size, pos, mat, col, mesh, model.geom_rgba[g].copy()))
        return out

    def body(new_name: str, b: int, skip: set[int] = frozenset()) -> Body:
        return Body(new_name, origins[b], inertia_of(b), geoms_of(b, set(skip)),
                    [_name(model, B, x) for x in welded(b)])

    trunk = body("trunk", root)
    legs, joints = {}, {}
    for leg, chain in labelled.items():
        legs[leg] = {}
        for part, b in zip(PARTS, chain):
            skip = {foot_body[leg]} if part == "calf" and foot_body[leg] is not None else set()
            legs[leg][part] = body(f"{leg}_{part}", b, skip)
            j = joint_of[b]
            s, off = sign[j], float(q_straight[model.jnt_qposadr[j]])
            limited = bool(model.jnt_limited[j])
            lo, hi = sorted(s * (r - off) for r in model.jnt_range[j]) if limited else (-math.pi, math.pi)
            if not limited:
                warnings.append(f"joint '{_name(model, J, j)}' has no limits in the URDF; using +-pi")
            effort, velocity = urdf_limits.get(_name(model, J, j), (0.0, 0.0))
            d = int(model.jnt_dofadr[j])
            joints[f"{leg}_{part}_joint"] = Joint(
                name=f"{leg}_{part}_joint", urdf_name=_name(model, J, j),
                axis=(1, 0, 0) if part == "hip" else (0, 1, 0), sign=s, offset=off,
                range=(float(lo), float(hi)), limited=limited,
                effort=effort or None, velocity=velocity or None,
                damping=float(model.dof_damping[d]), frictionloss=float(model.dof_frictionloss[d]),
                armature=float(model.dof_armature[d]))

    imu = [b for b in welded(root) if re.search(r"imu", _name(model, B, b), re.IGNORECASE)]
    meshes = [{"name": m.name, "file": Path(m.file).name, "scale": [float(x) for x in m.scale]}
              for m in spec.meshes]
    robot = ImportedRobot(
        name=name, trunk=trunk, legs=legs, joints=joints, foot_radius=radius, foot_pos=foot_pos,
        meshes=meshes, imu_pos=data.xpos[imu[0]].copy() if imu else np.zeros(3),
        warnings=warnings, retarget_problems=retarget_problems,
        source={"model": model, "q_straight": q_straight, "joint_of": joint_of, "labelled": labelled,
                "urdf": str(urdf), "foot_point": {leg: foot_point(leg) for leg in LEGS}},
        range_overrides=range_notes)
    _name_collisions(robot)
    undrawn = [b for b in robot.bodies if not any(not g.collision for g in b.geoms)]
    for b in undrawn:
        b.draw_collisions = True
    if undrawn:
        warnings.append("no readable visual geometry for " + ", ".join(b.name for b in undrawn)
                        + ": drawn with their collision shapes")
    return robot


def _name_collisions(robot: ImportedRobot) -> None:
    """Collision geoms get the names the configs match on: <body>_collision[N]; the trunk's
    first is a box called trunk_collision (the retargeting keeps that box off the floor)."""
    cols = [g for g in robot.trunk.geoms if g.collision]
    boxes = sorted((g for g in cols if g.kind == "box"), key=lambda g: -float(np.prod(g.size)))
    if not boxes:
        src = cols or robot.trunk.geoms
        if not src:
            raise ImportError_("the root link has no geometry at all; cannot size the trunk")
        pts = np.concatenate([_corners(g) for g in src])
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        box = Geom("box", (hi - lo) / 2, (hi + lo) / 2, np.eye(3), True)
        robot.trunk.geoms.append(box)
        robot.warnings.append("the root link has no box collision; added one around its "
                              f"{'collision' if cols else 'visual'} geometry "
                              f"({np.round(hi - lo, 3).tolist()} m)")
        boxes, cols = [box], cols + [box]
    boxes[0].name = "trunk_collision"
    rest = [g for g in cols if g is not boxes[0]]
    for i, g in enumerate(rest, start=2):
        g.name = f"trunk_collision{i}"
    for leg in LEGS:
        for part in PARTS:
            b = robot.legs[leg][part]
            cols = [g for g in b.geoms if g.collision]
            for i, g in enumerate(cols, start=1):
                g.name = f"{b.name}_collision" + ("" if len(cols) == 1 else str(i))


def _corners(g: Geom) -> np.ndarray:
    """Eight corners of a geom's bounding box in the trunk frame."""
    if g.kind == "sphere":
        half = np.full(3, g.size[0])
    elif g.kind in ("capsule", "cylinder"):
        half = np.array([g.size[0], g.size[0], g.size[1] + (g.size[0] if g.kind == "capsule" else 0)])
    else:
        half = np.asarray(g.size[:3])
    s = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], float)
    return g.pos + (s * half) @ g.mat.T


# ---- measurements -----------------------------------------------------------------

def _per_joint(values, what: str) -> list[float]:
    v = [float(x) for x in values]
    if len(v) == 3:
        return v * 4
    if len(v) != 12:
        raise ImportError_(f"{what}: need 3 values [hip, thigh, calf] or 12 (FR, FL, RR, RL x 3), got {len(v)}")
    return v


def apply_measurements(robot: ImportedRobot, meas: dict) -> list[str]:
    """Replace CAD numbers with bench measurements (docs/robot_import.md lists the keys).
    Returns one line per change. The centre of mass is applied later, by ``match_com``,
    because it is measured with the legs in the standing pose."""
    notes = []
    known = {"total_mass", "link_mass", "com", "trunk_inertia", "joint", "foot_friction", "notes"}
    if set(meas) - known:
        raise ImportError_(f"measurements: unknown keys {sorted(set(meas) - known)}; known: {sorted(known)}")
    for part, mass in (meas.get("link_mass") or {}).items():
        if part not in PARTS:
            raise ImportError_(f"measurements link_mass: '{part}' is not one of {PARTS}")
        for leg in LEGS:
            inr = robot.legs[leg][part].inertia
            k = float(mass) / inr.mass
            inr.mass, inr.tensor = float(mass), inr.tensor * k
        notes.append(f"{part} links: mass set to {float(mass):.4f} kg each (inertia scaled with it)")
    if meas.get("total_mass") is not None:
        legs = sum(robot.legs[leg][p].inertia.mass for leg in LEGS for p in PARTS)
        new = float(meas["total_mass"]) - legs
        inr = robot.trunk.inertia
        if new <= 0.2 * inr.mass:
            raise ImportError_(f"measurements: total_mass {meas['total_mass']} kg leaves {new:.3f} kg for the "
                               f"trunk (CAD: {inr.mass:.3f} kg, legs: {legs:.3f} kg); check the link masses")
        notes.append(f"trunk mass {inr.mass:.4f} -> {new:.4f} kg so the robot weighs {float(meas['total_mass']):.4f} kg")
        inr.tensor, inr.mass = inr.tensor * (new / inr.mass), new
    if meas.get("trunk_inertia") is not None:
        diag = [float(x) for x in meas["trunk_inertia"]]
        robot.trunk.inertia.tensor = np.diag(diag)
        notes.append(f"trunk inertia set to diag{diag} kg m^2")
    joint = meas.get("joint") or {}
    if set(joint) - {"armature", "frictionloss", "damping", "rotor_inertia", "gear_ratio"}:
        raise ImportError_(f"measurements joint: unknown keys {sorted(joint)}")
    if "rotor_inertia" in joint or "gear_ratio" in joint:
        rotor = _per_joint(joint["rotor_inertia"], "joint.rotor_inertia")
        gear = _per_joint(joint["gear_ratio"], "joint.gear_ratio")
        joint = {**joint, "armature": [n * n * i for n, i in zip(gear, rotor)]}
    for key in ("armature", "frictionloss", "damping"):
        if joint.get(key) is not None:
            for jn, v in zip(JOINT_NAMES, _per_joint(joint[key], f"joint.{key}")):
                setattr(robot.joints[jn], key, v)
            notes.append(f"joint {key} set from measurements")
    return notes


# ---- writing the MJCF -------------------------------------------------------------

def _f(x) -> str:
    return " ".join("0" if abs(v) < 1e-12 else f"{float(v):.9g}" for v in np.atleast_1d(x))


def _quat(mat: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(mat).reshape(9))
    return q


def to_mjcf(robot: ImportedRobot, meshdir: str = "../meshes") -> str:
    """The model as MJCF text: free-floating trunk, no actuators (training adds them)."""
    root = ET.Element("mujoco", model=robot.name)
    ET.SubElement(root, "compiler", angle="radian", meshdir=meshdir, autolimits="true")
    default = ET.SubElement(ET.SubElement(root, "default"), "default", {"class": robot.name})
    ET.SubElement(ET.SubElement(default, "default", {"class": "visual"}), "geom",
                  contype="0", conaffinity="0", group="2")
    ET.SubElement(ET.SubElement(default, "default", {"class": "collision"}), "geom",
                  contype="1", conaffinity="1", group="3")
    if robot.meshes:
        asset = ET.SubElement(root, "asset")
        for m in robot.meshes:
            attrs = {"name": m["name"], "file": m["file"]}
            if not np.allclose(m["scale"], 1.0):
                attrs["scale"] = _f(m["scale"])
            ET.SubElement(asset, "mesh", attrs)

    def add_body(parent_el, body: Body, parent_origin: np.ndarray):
        el = ET.SubElement(parent_el, "body", name=body.name, pos=_f(body.origin - parent_origin))
        inr = body.inertia
        t = inr.tensor
        ET.SubElement(el, "inertial", pos=_f(inr.com - body.origin), mass=_f(inr.mass),
                      fullinertia=_f([t[0, 0], t[1, 1], t[2, 2], t[0, 1], t[0, 2], t[1, 2]]))
        return el

    def add_geoms(el, body: Body):
        for g in body.geoms:
            attrs = {"class": "collision" if g.collision else "visual", "type": g.kind}
            if g.name:
                attrs["name"] = g.name
            if g.kind == "mesh":
                attrs["mesh"] = g.mesh
            else:
                n = {"sphere": 1, "capsule": 2, "cylinder": 2, "box": 3}[g.kind]
                attrs["size"] = _f(g.size[:n])
            attrs["pos"] = _f(g.pos - body.origin)
            if not np.allclose(g.mat, np.eye(3), atol=1e-12):
                attrs["quat"] = _f(_quat(g.mat))
            if not g.collision and g.rgba is not None:
                attrs["rgba"] = _f(g.rgba)
            ET.SubElement(el, "geom", attrs)
            if g.collision and body.draw_collisions:
                shown = {k: v for k, v in attrs.items() if k != "name"}
                ET.SubElement(el, "geom", {**shown, "class": "visual", "rgba": "0.55 0.55 0.6 1"})

    world = ET.SubElement(root, "worldbody")
    span = max(float(np.ptp([robot.foot_pos[leg][0] for leg in LEGS])), 0.2)
    trunk = add_body(world, robot.trunk, np.zeros(3))
    # The model's own initial state: clear of a floor at z = 0 with the legs straight down
    # (a model that starts inside the floor overflows the simulator's contact buffers).
    clearance = max(-robot.foot_pos[leg][2] for leg in LEGS) + robot.foot_radius + 0.01
    trunk.set("pos", _f([0, 0, math.ceil(clearance * 100) / 100]))
    trunk.set("childclass", robot.name)
    ET.SubElement(trunk, "freejoint", name="floating_base_joint")
    ET.SubElement(trunk, "camera", name="tracking", mode="trackcom",
                  pos=_f([2.2 * span, -3.4 * span, 0.8 * span]), xyaxes="0.866 0.5 0 -0.171 0.296 0.94")
    add_geoms(trunk, robot.trunk)
    ET.SubElement(trunk, "site", name="imu", pos=_f(robot.imu_pos), group="5")
    for leg in LEGS:
        parent_el, parent_origin = trunk, np.zeros(3)
        for part in PARTS:
            body = robot.legs[leg][part]
            el = add_body(parent_el, body, parent_origin)
            j = robot.joints[f"{leg}_{part}_joint"]
            attrs = {"name": j.name, "axis": _f(j.axis), "range": _f(j.range)}
            for key in ("damping", "frictionloss", "armature"):
                if getattr(j, key) > 0:
                    attrs[key] = _f(getattr(j, key))
            ET.SubElement(el, "joint", attrs)
            add_geoms(el, body)
            parent_el, parent_origin = el, body.origin
        foot = robot.foot_pos[leg] - robot.legs[leg]["calf"].origin
        ET.SubElement(parent_el, "geom", {"class": "collision", "name": f"{leg}_foot_collision",
                                          "type": "sphere", "size": _f(robot.foot_radius), "pos": _f(foot)})
        if robot.legs[leg]["calf"].draw_collisions:
            ET.SubElement(parent_el, "geom", {"class": "visual", "type": "sphere", "size": _f(robot.foot_radius),
                                              "pos": _f(foot), "rgba": "0.2 0.2 0.2 1"})
        ET.SubElement(parent_el, "site", name=leg, pos=_f(foot), group="5")
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode") + "\n"


# ---- stance, checks ---------------------------------------------------------------

def default_stance(robot: ImportedRobot, height_ratio: float = 0.62, knee_sign: float = -1.0) -> tuple[float, float]:
    """(thigh, calf) that puts each foot under its thigh joint at ``height_ratio`` of the
    straight leg length (0.62 is the Go1's crouch), clipped into the joint limits."""
    _, l1, l2 = robot.leg_lengths("FR")
    h = float(np.clip(height_ratio, 0.05, 0.999)) * (l1 + l2)
    h = max(h, abs(l1 - l2) * 1.001)
    calf = knee_sign * math.acos(np.clip((h * h - l1 * l1 - l2 * l2) / (2 * l1 * l2), -1, 1))
    thigh = -math.atan2(l2 * math.sin(calf), l1 + l2 * math.cos(calf))
    return thigh, calf


def stance_vector(robot: ImportedRobot, stance: tuple[float, float]) -> list[float]:
    q = [0.0, float(stance[0]), float(stance[1])] * 4
    for jn, v in zip(JOINT_NAMES, q):
        lo, hi = robot.joints[jn].range
        if not lo - 1e-6 <= v <= hi + 1e-6:
            raise ImportError_(f"standing pose {jn} = {v:+.3f} rad is outside the joint's range "
                               f"[{lo:+.3f}, {hi:+.3f}] (this project's sign convention); pass --stance")
    return q


def compile_mjcf(robot: ImportedRobot, mesh_dir: Path) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(to_mjcf(robot, meshdir=str(Path(mesh_dir).resolve())))


def standing_height(model: mujoco.MjModel, robot: ImportedRobot, q: list[float]) -> float:
    """Trunk height with the lowest foot touching the floor."""
    data = mujoco.MjData(model)
    data.qpos[:7] = (0, 0, 0, 1, 0, 0, 0)
    data.qpos[7:] = q
    mujoco.mj_kinematics(model, data)
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{leg}_foot_collision") for leg in LEGS]
    return float(-min(data.geom_xpos[g][2] for g in ids) + robot.foot_radius)


def match_com(robot: ImportedRobot, mesh_dir: Path, q: list[float], target) -> str:
    """Move the trunk's centre of mass until the whole robot's, in the standing pose, is at
    ``target`` (trunk frame; None for an axis keeps the CAD value)."""
    model = compile_mjcf(robot, mesh_dir)
    data = mujoco.MjData(model)
    data.qpos[:7] = (0, 0, 0, 1, 0, 0, 0)
    data.qpos[7:] = q
    mujoco.mj_forward(model, data)
    now = data.subtree_com[1].copy()
    want = np.array([now[i] if target[i] is None else float(target[i]) for i in range(3)])
    shift = (want - now) * robot.total_mass / robot.trunk.inertia.mass
    robot.trunk.inertia.com = robot.trunk.inertia.com + shift
    return (f"whole-robot COM in the standing pose {np.round(now, 4).tolist()} -> "
            f"{np.round(want, 4).tolist()} m (trunk COM moved by {np.round(shift, 4).tolist()})")


def verify_against_urdf(robot: ImportedRobot, mesh_dir: Path, samples: int = 20, seed: int = 0) -> float:
    """Largest difference between the URDF and the rebuilt model over random poses, with
    the joints related by the hardware map: foot centres and the whole-robot centre of
    mass (m), and total mass (kg). Run before any measurement is applied."""
    src, new = robot.source["model"], compile_mjcf(robot, mesh_dir)
    ds, dn = mujoco.MjData(src), mujoco.MjData(new)
    J, G = mujoco.mjtObj.mjOBJ_JOINT, mujoco.mjtObj.mjOBJ_GEOM
    # Foot centres in their calf link's frame, from the straight-leg pose.
    ds.qpos[:] = robot.source["q_straight"]
    mujoco.mj_kinematics(src, ds)
    calf = {leg: robot.source["labelled"][leg][2] for leg in LEGS}
    local = {leg: ds.xmat[calf[leg]].reshape(3, 3).T @ (robot.source["foot_point"][leg] - ds.xpos[calf[leg]])
             for leg in LEGS}
    rng = np.random.default_rng(seed)
    worst = abs(float(src.body_mass.sum()) - float(new.body_mass.sum()))
    for _ in range(samples):
        qs = rng.uniform(-0.6, 0.6, 12)
        dn.qpos[:7] = (0, 0, 0, 1, 0, 0, 0)
        dn.qpos[7:] = qs
        for jn, v in zip(JOINT_NAMES, qs):
            j = robot.joints[jn]
            ds.qpos[src.jnt_qposadr[mujoco.mj_name2id(src, J, j.urdf_name)]] = j.offset + j.sign * v
        mujoco.mj_forward(src, ds)
        mujoco.mj_forward(new, dn)
        for leg in LEGS:
            foot_src = ds.xpos[calf[leg]] + ds.xmat[calf[leg]].reshape(3, 3) @ local[leg]
            foot_new = dn.geom_xpos[mujoco.mj_name2id(new, G, f"{leg}_foot_collision")]
            worst = max(worst, float(np.linalg.norm(foot_src - foot_new)))
        # The URDF's root link is fixed to the world, so its whole tree is subtree 0.
        worst = max(worst, float(np.linalg.norm(ds.subtree_com[0] - dn.subtree_com[1])))
    return worst


def drop_test(robot: ImportedRobot, mesh_dir: Path, q0: list[float], height: float,
              kp: list[float], kd: list[float], torque_limit: list[float], armature: list[float],
              seconds: float = 2.0) -> dict:
    """Stand on a floor for a moment with the PD gains from the config (plain CPU MuJoCo).
    Reports whether it stays up, the torque it takes, and any self-collision in the stance."""
    root = ET.fromstring(to_mjcf(robot, meshdir=str(Path(mesh_dir).resolve())))
    ET.SubElement(root.find("worldbody"), "geom", name="floor", type="plane", size="0 0 1",
                  contype="1", conaffinity="1")
    m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    d = mujoco.MjData(m)
    m.opt.timestep = 0.002
    m.dof_armature[6:] = armature
    q0 = np.array(q0)
    d.qpos[:7] = (0, 0, height + 0.005, 1, 0, 0, 0)
    d.qpos[7:] = q0
    mujoco.mj_forward(m, d)
    G = mujoco.mjtObj.mjOBJ_GEOM
    floor = mujoco.mj_name2id(m, G, "floor")
    self_hits = sorted({tuple(sorted((_name(m, G, c.geom1), _name(m, G, c.geom2))))
                        for c in d.contact[: d.ncon] if floor not in (c.geom1, c.geom2) and c.dist < 0})
    kp, kd, lim = np.array(kp), np.array(kd), np.array(torque_limit)
    for _ in range(int(seconds / m.opt.timestep)):
        d.qfrc_applied[6:] = np.clip(kp * (q0 - d.qpos[7:]) - kd * d.qvel[6:], -lim, lim)
        mujoco.mj_step(m, d)
    hold = np.abs(np.clip(kp * (q0 - d.qpos[7:]) - kd * d.qvel[6:], -lim, lim))
    up = float(d.xmat[1].reshape(3, 3)[2, 2])
    return {"height_start": float(height), "height_end": float(d.qpos[2]),
            "tilt_deg": float(np.degrees(np.arccos(np.clip(up, -1, 1)))),
            "hold_torque_fraction": float((hold / lim).max()),
            "joint_sag_rad": float(np.abs(d.qpos[7:] - q0).max()), "self_collisions": self_hits,
            "stands": bool(up > 0.95 and d.qpos[2] > 0.7 * height)}
