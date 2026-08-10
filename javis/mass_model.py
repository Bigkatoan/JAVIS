"""Per-component-group mass/inertia model for the JAVIS chassis.

Why this exists
---------------
`robot.urdf` carries no real mass data (onshape-to-robot exports `mass="1e-9"`
placeholders because the Onshape assembly has no materials assigned). The
previous workaround in `robot_constants.py` applied ONE uniform density across
all 343 chassis parts, scaled so the total hit a guessed `CHASSIS_MASS_KG`.

That is wrong in a way that matters specifically for this robot. The battery is
39% of the chassis mesh volume but ~5x denser than the 3D-printed structure
around it, so a uniform density puts the center of mass in the wrong place --
and for a two-wheel self-balancing robot, the center of mass IS the control
problem.

This module instead groups the chassis parts by what they physically are
(battery / printed PLA / Jetson / camera / ODrive boards / IMU / fasteners /
misc electronics), and gives each group its own mass. Masses come from real
measurements where available, catalog figures otherwise -- see NOMINAL below.

The key property
----------------
Mass, center of mass and inertia are all LINEAR in the per-group masses. So we
precompute, once, each group's unit-mass moments:

    unit_com[g]    = (1/V_g) * integral_g( r )    dV      -- the group's own CoM
    unit_second[g] = (1/V_g) * integral_g( r r^T ) dV     -- about the body origin

and then for any mass vector `m` the fused body properties are just:

    M       = sum_g m_g
    com     = (sum_g m_g * unit_com[g]) / M
    C_org   = sum_g m_g * unit_second[g]
    I_org   = trace(C_org) * E - C_org                    -- covariance -> inertia
    I_com   = I_org - M * (|com|^2 * E - com com^T)       -- parallel axis

That means every environment can get its own physically consistent
(mass, ipos, iquat, inertia) with a couple of batched matmuls on the GPU, with
no MjSpec recompile. `javis/mdp/events.py` uses exactly this to randomize
payload and component masses per environment.

Point-mass groups (`wiring_misc`, `payload`) have no CAD geometry: they are
declared with a position instead of a mesh, which fits the same formulation
with `unit_com = p` and `unit_second = p p^T`.
"""

from __future__ import annotations

import fnmatch
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import trimesh

JAVIS_DIR: Path = Path(__file__).resolve().parent
JAVIS_URDF: Path = JAVIS_DIR / "robot.urdf"
_CACHE_PATH: Path = JAVIS_DIR / "assets" / "_mass_moments.npz"

# Bump when GROUP_PATTERNS or the moment math changes, so a stale cache from an
# earlier definition is rebuilt instead of silently reused.
_CACHE_VERSION = 1

CHASSIS_BODY = "body"
WHEEL_BODIES = ("wheel", "wheel_2")

##
# Group definitions.
##

# Which mesh filenames belong to which physical component group. Matched with
# fnmatch in declaration order -- first match wins. Anything unmatched lands in
# FALLBACK_GROUP and is reported by build_moments() so it can't silently rot.
#
# Mapping notes (verify visually with `view_robot.py --color-by-group`):
#  - "printed" is everything actually 3D-printed in PLA: the structural chassis,
#    covers, holders, the encoder-magnet relocation gear train (confirmed
#    printed, not a power gearbox -- SIM2REAL.md sec 2).
#  - "jetson" is inferred from volume, not from the CAD names: the Onshape
#    library names are generic ("main_board_simplified", "module_board_me"), but
#    their 69.6 cm^3 combined is right for an Orin Nano carrier + module +
#    heatsink. Flagged in SIM2REAL.md as worth confirming by eye.
#  - "odrive" covers 2 real MKS xDrive Mini boards; the CAD repeats each sub-mesh
#    exactly twice, which is what confirms the count is 2 (MKS_XDRIVE_MINI.md).
GROUP_PATTERNS: dict[str, tuple[str, ...]] = {
  "battery": ("battery.stl",),
  "printed": (
    "base_link.stl",
    "headcover.stl",
    "gearbox.stl",
    "part_6.stl",
    "part_6__*.stl",
    "coupler.stl",
    "wheelconnector.stl",
    "jetson_holder.stl",
    "jetsonholder.stl",
    "spur_gear__*.stl",
    "board_frame.stl",
  ),
  "jetson": ("main_board_simplified.stl", "module_board_me*.stl"),
  "camera": ("intelrealsensed435.stl",),
  "imu": ("dfbot_imu_sensor.stl",),
  "odrive": ("mks_odrive_mini_bottom*.stl", "board.stl", "board__*.stl"),
  "hardware": (
    "m3x*.stl",
    "m1_6x*.stl",
    "6802bearing.stl",
    "6808_2rs.stl",
    "693zz.stl",
  ),
}
FALLBACK_GROUP = "electronics_misc"

# Point-mass groups: no CAD geometry at all, declared as (x, y, z) in the
# chassis link frame. Their position is itself randomizable at runtime -- see
# `fuse(point_positions=...)`.
#
# wiring_misc is the "dây và các thứ linh tinh" the user cannot weigh: harness,
# heat-shrink, zip ties, tape, connectors not in CAD. Modeled as one lump so
# it shows up honestly as an unknown instead of being absorbed into a fudged
# chassis total.
POINT_GROUPS: dict[str, tuple[float, float, float]] = {
  "wiring_misc": (0.0, 0.0, 0.15),
  "payload": (0.0, 0.0, 0.30),
}

##
# Nominal masses.
##

# Groups whose mass is known directly (measured or catalog), in kg.
_NOMINAL_MASS_KG: dict[str, float] = {
  # Measured 2026-08-06 (SIM2REAL.md sec 1).
  "battery": 3.423,
  # One 1 kg spool of PLA for the entire robot, printed at ~15% infill (user,
  # 2026-08-10). Cross-check: 1.000 kg / 2438.5 cm^3 = 0.410 g/cm^3, i.e. an
  # effective 33% of solid PLA (1.24 g/cm^3) once perimeters and top/bottom
  # layers are counted on top of 15% sparse infill. Two independent estimates
  # agreeing is why this number is trusted over a density guess.
  "printed": 1.000,
  # Catalog: NVIDIA Jetson Orin Nano Developer Kit, 176 g (module + carrier).
  "jetson": 0.176,
  # Catalog: Intel RealSense D435, 72 g (D400-series datasheet).
  "camera": 0.072,
  # 2 x MKS xDrive Mini. NOT verified -- vendor listings only quote shipping
  # weight. 35 g/board is an estimate for a 65x45 mm populated PCB. At 70 g of
  # ~11 kg it barely moves the CoM, but weigh them when convenient.
  "odrive": 0.070,
  # Bosch BMX160 + BMP388 breakout.
  "imu": 0.005,
  # Measured 2026-08-06, per wheel (hub motor + tire).
  "wheel": 2.936,
}

# Groups whose mass is derived from mesh volume x an assumed material density,
# in kg/m^3.
_NOMINAL_DENSITY: dict[str, float] = {
  # Steel fasteners and bearings. Sanity check: 2 x 6808-2RS at this density
  # gives 95 g, against a ~48 g/bearing catalog figure -- matches.
  "hardware": 7850.0,
  # Connectors, terminals, capacitors, copper busbars, shielding. A mixed bag
  # of copper and plastic; 3000 is a middling guess covered by wide DR.
  "electronics_misc": 3000.0,
}

# Point-mass group nominals, in kg.
_NOMINAL_POINT_MASS_KG: dict[str, float] = {
  "wiring_misc": 0.300,
  "payload": 0.0,
}


##
# Precomputed moments.
##


@dataclass(frozen=True)
class BodyMoments:
  """Unit-mass moments per group, in the body's link frame.

  All arrays are ordered to match `groups`.
  """

  body: str
  groups: tuple[str, ...]
  volume: np.ndarray  # (G,)      m^3; 0.0 for point groups.
  unit_com: np.ndarray  # (G, 3)    m      -- group CoM.
  unit_second: np.ndarray  # (G, 3, 3) m^2    -- (1/m) * integral(r r^T dm), about origin.
  is_point: np.ndarray  # (G,)      bool

  def index(self, group: str) -> int:
    return self.groups.index(group)


def _rpy_to_mat(rpy: tuple[float, float, float]) -> np.ndarray:
  """URDF <origin rpy="..."/> (extrinsic X-Y-Z) -> 3x3 rotation matrix."""
  r, p, y = rpy
  cr, sr = np.cos(r), np.sin(r)
  cp, sp = np.cos(p), np.sin(p)
  cy, sy = np.cos(y), np.sin(y)
  rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
  ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
  rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
  return rz @ ry @ rx


def _classify(mesh_name: str) -> str:
  for group, patterns in GROUP_PATTERNS.items():
    if any(fnmatch.fnmatch(mesh_name, pat) for pat in patterns):
      return group
  return FALLBACK_GROUP


def _mesh_moments(mesh: trimesh.Trimesh) -> tuple[float, np.ndarray, np.ndarray]:
  """Volume, centroid, and second moment integral(r r^T dV) about the mesh origin.

  trimesh reports the inertia tensor `I` about the centroid at unit density.
  The covariance form we want is recovered with C = trace(I)/2 * E - I, which
  inverts I = trace(C) * E - C.
  """
  volume = float(mesh.volume)
  centroid = np.asarray(mesh.center_mass, dtype=np.float64)
  inertia_com = np.asarray(mesh.moment_inertia, dtype=np.float64)
  cov_com = 0.5 * np.trace(inertia_com) * np.eye(3) - inertia_com
  # Parallel axis, covariance form: C_origin = C_com + V * c c^T.
  cov_origin = cov_com + volume * np.outer(centroid, centroid)
  return volume, volume * centroid, cov_origin


def _transform_moments(
  volume: float,
  first: np.ndarray,
  second: np.ndarray,
  rot: np.ndarray,
  pos: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
  """Push (V, integral r dV, integral r r^T dV) through x = R r + t."""
  first_t = rot @ first + volume * pos
  outer_cross = np.outer(rot @ first, pos)
  second_t = (
    rot @ second @ rot.T + outer_cross + outer_cross.T + volume * np.outer(pos, pos)
  )
  return volume, first_t, second_t


def _iter_visual_meshes(link: ET.Element):
  """Yield (mesh_filename, rotation, position) for each <visual> mesh in a link."""
  for visual in link.findall("visual"):
    mesh_el = visual.find("geometry/mesh")
    if mesh_el is None:
      continue
    scale = mesh_el.get("scale")
    if scale is not None and not np.allclose(
      [float(s) for s in scale.split()], 1.0
    ):
      raise NotImplementedError(
        f"mesh {mesh_el.get('filename')} has scale={scale}; moment computation "
        "assumes unit scale"
      )
    origin = visual.find("origin")
    if origin is None:
      rot, pos = np.eye(3), np.zeros(3)
    else:
      pos = np.array([float(v) for v in origin.get("xyz", "0 0 0").split()])
      rpy = tuple(float(v) for v in origin.get("rpy", "0 0 0").split())
      rot = _rpy_to_mat(rpy)  # type: ignore[arg-type]
    yield Path(mesh_el.get("filename", "")).name, rot, pos


def _build_body_moments(
  link: ET.Element, point_groups: dict[str, tuple[float, float, float]]
) -> tuple[BodyMoments, dict[str, list[str]]]:
  acc: dict[str, list[np.ndarray | float]] = {}
  members: dict[str, list[str]] = {}
  mesh_cache: dict[str, trimesh.Trimesh] = {}

  for mesh_name, rot, pos in _iter_visual_meshes(link):
    if mesh_name not in mesh_cache:
      mesh_cache[mesh_name] = trimesh.load(
        JAVIS_DIR / "assets" / mesh_name, force="mesh"
      )
    volume, first, second = _mesh_moments(mesh_cache[mesh_name])
    volume, first, second = _transform_moments(volume, first, second, rot, pos)

    group = _classify(mesh_name)
    members.setdefault(group, [])
    if mesh_name not in members[group]:
      members[group].append(mesh_name)
    if group not in acc:
      acc[group] = [0.0, np.zeros(3), np.zeros((3, 3))]
    acc[group][0] += volume  # type: ignore[operator]
    acc[group][1] = acc[group][1] + first  # type: ignore[operator]
    acc[group][2] = acc[group][2] + second  # type: ignore[operator]

  groups: list[str] = []
  volumes: list[float] = []
  unit_com: list[np.ndarray] = []
  unit_second: list[np.ndarray] = []
  is_point: list[bool] = []

  for group, (volume, first, second) in sorted(acc.items()):  # type: ignore[misc]
    groups.append(group)
    volumes.append(float(volume))
    unit_com.append(np.asarray(first) / volume)
    unit_second.append(np.asarray(second) / volume)
    is_point.append(False)

  for group, position in point_groups.items():
    groups.append(group)
    volumes.append(0.0)
    p = np.asarray(position, dtype=np.float64)
    unit_com.append(p)
    unit_second.append(np.outer(p, p))
    is_point.append(True)

  moments = BodyMoments(
    body=str(link.get("name")),
    groups=tuple(groups),
    volume=np.asarray(volumes),
    unit_com=np.stack(unit_com),
    unit_second=np.stack(unit_second),
    is_point=np.asarray(is_point),
  )
  return moments, members


def build_moments(verbose: bool = False) -> dict[str, BodyMoments]:
  """Compute per-group unit-mass moments for the chassis and both wheels.

  Loads every referenced STL, so it takes a few seconds. Results are cached to
  `assets/_mass_moments.npz`; use `get_moments()` for the cached path.
  """
  root = ET.parse(JAVIS_URDF).getroot()
  links = {link.get("name"): link for link in root.findall("link")}

  out: dict[str, BodyMoments] = {}
  moments, members = _build_body_moments(links[CHASSIS_BODY], POINT_GROUPS)
  out[CHASSIS_BODY] = moments

  if verbose:
    fallback_vol = 0.0
    idx = {g: i for i, g in enumerate(moments.groups)}
    if FALLBACK_GROUP in idx:
      fallback_vol = float(moments.volume[idx[FALLBACK_GROUP]])
    total_vol = float(moments.volume.sum())
    print(f"chassis mesh volume: {total_vol * 1e6:.1f} cm^3")
    for group in moments.groups:
      if moments.is_point[idx[group]]:
        continue
      n = len(members.get(group, []))
      print(
        f"  {group:18s} {moments.volume[idx[group]] * 1e6:9.2f} cm^3  "
        f"({n} unique meshes)"
      )
    if fallback_vol / total_vol > 0.02:
      print(
        f"\nWARNING: {fallback_vol / total_vol:.1%} of chassis volume fell into "
        f"'{FALLBACK_GROUP}'. Meshes: {', '.join(sorted(members[FALLBACK_GROUP]))}"
      )

  for wheel in WHEEL_BODIES:
    wheel_moments, _ = _build_body_moments(links[wheel], {})
    # Both wheel bodies hold a single mesh; force the group name so callers
    # don't have to care which mesh file backs it.
    out[wheel] = BodyMoments(
      body=wheel,
      groups=("wheel",),
      volume=wheel_moments.volume,
      unit_com=wheel_moments.unit_com,
      unit_second=wheel_moments.unit_second,
      is_point=wheel_moments.is_point,
    )
  return out


def get_moments(rebuild: bool = False) -> dict[str, BodyMoments]:
  """Cached `build_moments()`. Rebuilds if the cache is missing or stale."""
  if not rebuild and _CACHE_PATH.exists():
    data = np.load(_CACHE_PATH, allow_pickle=False)
    if int(data["version"]) == _CACHE_VERSION:
      out: dict[str, BodyMoments] = {}
      for body in [CHASSIS_BODY, *WHEEL_BODIES]:
        out[body] = BodyMoments(
          body=body,
          groups=tuple(str(g) for g in data[f"{body}/groups"]),
          volume=data[f"{body}/volume"],
          unit_com=data[f"{body}/unit_com"],
          unit_second=data[f"{body}/unit_second"],
          is_point=data[f"{body}/is_point"],
        )
      return out

  moments = build_moments()
  payload = {"version": np.array(_CACHE_VERSION)}
  for body, m in moments.items():
    payload[f"{body}/groups"] = np.array(m.groups)
    payload[f"{body}/volume"] = m.volume
    payload[f"{body}/unit_com"] = m.unit_com
    payload[f"{body}/unit_second"] = m.unit_second
    payload[f"{body}/is_point"] = m.is_point
  _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
  np.savez(_CACHE_PATH, **payload)
  return moments


##
# Nominal mass vector.
##


def nominal_masses(body: str = CHASSIS_BODY) -> dict[str, float]:
  """Nominal mass in kg for each group of `body`, in `BodyMoments.groups` order.

  Groups listed in `_NOMINAL_MASS_KG` use that mass directly; the rest derive
  theirs from mesh volume x an assumed material density.
  """
  moments = get_moments()[body]
  out: dict[str, float] = {}
  for i, group in enumerate(moments.groups):
    if group in _NOMINAL_MASS_KG:
      out[group] = _NOMINAL_MASS_KG[group]
    elif group in _NOMINAL_POINT_MASS_KG:
      out[group] = _NOMINAL_POINT_MASS_KG[group]
    elif group in _NOMINAL_DENSITY:
      out[group] = _NOMINAL_DENSITY[group] * float(moments.volume[i])
    else:
      raise KeyError(
        f"group '{group}' on body '{body}' has no nominal mass or density. "
        "Add it to _NOMINAL_MASS_KG or _NOMINAL_DENSITY."
      )
  return out


def nominal_mass_vector(
  body: str = CHASSIS_BODY, device: torch.device | str = "cpu"
) -> torch.Tensor:
  """`nominal_masses` as a (G,) tensor ordered to match `BodyMoments.groups`."""
  moments = get_moments()[body]
  masses = nominal_masses(body)
  return torch.tensor(
    [masses[g] for g in moments.groups], dtype=torch.float32, device=device
  )


##
# Fusion: group masses -> MuJoCo body inertial properties.
##


def _mat_to_quat(mat: torch.Tensor) -> torch.Tensor:
  """Batched rotation matrix (..., 3, 3) -> MuJoCo (w, x, y, z) quaternion.

  Uses the largest-diagonal-element branch, which stays numerically stable for
  all rotations (the naive w-first formula degenerates near 180 deg).
  """
  m = mat
  batch = m.shape[:-2]
  m00, m11, m22 = m[..., 0, 0], m[..., 1, 1], m[..., 2, 2]
  trace = m00 + m11 + m22

  # Four candidate reconstructions; pick the one with the largest denominator.
  cands = torch.stack(
    [
      torch.stack(
        [
          1.0 + trace,
          m[..., 2, 1] - m[..., 1, 2],
          m[..., 0, 2] - m[..., 2, 0],
          m[..., 1, 0] - m[..., 0, 1],
        ],
        dim=-1,
      ),
      torch.stack(
        [
          m[..., 2, 1] - m[..., 1, 2],
          1.0 + m00 - m11 - m22,
          m[..., 0, 1] + m[..., 1, 0],
          m[..., 0, 2] + m[..., 2, 0],
        ],
        dim=-1,
      ),
      torch.stack(
        [
          m[..., 0, 2] - m[..., 2, 0],
          m[..., 0, 1] + m[..., 1, 0],
          1.0 - m00 + m11 - m22,
          m[..., 1, 2] + m[..., 2, 1],
        ],
        dim=-1,
      ),
      torch.stack(
        [
          m[..., 1, 0] - m[..., 0, 1],
          m[..., 0, 2] + m[..., 2, 0],
          m[..., 1, 2] + m[..., 2, 1],
          1.0 - m00 - m11 + m22,
        ],
        dim=-1,
      ),
    ],
    dim=-2,
  )  # (..., 4, 4)

  denom = torch.stack([1.0 + trace, 1.0 + m00 - m11 - m22, 1.0 - m00 + m11 - m22, 1.0 - m00 - m11 + m22], dim=-1)
  best = denom.argmax(dim=-1)  # (...,)
  quat = torch.gather(
    cands, -2, best[..., None, None].expand(*batch, 1, 4)
  ).squeeze(-2)
  quat = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
  # Canonical sign: w >= 0.
  return torch.where(quat[..., :1] < 0, -quat, quat)


def fuse(
  masses: torch.Tensor,
  moments: BodyMoments,
  point_positions: dict[str, torch.Tensor] | None = None,
  eigh_fn: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  """Fuse per-group masses into MuJoCo body inertial properties.

  Args:
    masses: (..., G) group masses in kg, ordered like `moments.groups`.
    moments: precomputed unit-mass moments for the body.
    point_positions: optional per-group (..., 3) overrides for point-mass
      groups (e.g. a randomized payload mount position). Groups not listed keep
      the position baked into `moments`.
    eigh_fn: symmetric eigendecomposition, returning (ascending eigenvalues,
      eigenvector columns). Defaults to `torch.linalg.eigh`. On GPU, callers
      should pass mjlab's `_eigh_3x3_jacobi` instead -- `torch.linalg.eigh`
      pulls in cuSOLVER, which permanently reserves several GB of VRAM on first
      use (which is exactly why mjlab ships its own 3x3 Jacobi routine).

  Returns:
    mass (...,), ipos (..., 3), iquat (..., 4) as MuJoCo wxyz,
    inertia (..., 3) as the principal (diagonal) inertia matching `iquat`.
  """
  device, dtype = masses.device, masses.dtype
  unit_com = torch.as_tensor(moments.unit_com, device=device, dtype=dtype)
  unit_second = torch.as_tensor(moments.unit_second, device=device, dtype=dtype)

  batch = masses.shape[:-1]
  unit_com = unit_com.expand(*batch, *unit_com.shape)
  unit_second = unit_second.expand(*batch, *unit_second.shape)

  if point_positions:
    unit_com = unit_com.clone()
    unit_second = unit_second.clone()
    for group, pos in point_positions.items():
      i = moments.index(group)
      if not moments.is_point[i]:
        raise ValueError(f"group '{group}' is mesh-backed; its position is fixed")
      pos = pos.to(device=device, dtype=dtype).expand(*batch, 3)
      unit_com[..., i, :] = pos
      unit_second[..., i, :, :] = pos[..., :, None] * pos[..., None, :]

  total = masses.sum(dim=-1)  # (...,)
  safe_total = total.clamp_min(1e-9)
  com = (masses[..., None] * unit_com).sum(dim=-2) / safe_total[..., None]
  cov_origin = (masses[..., None, None] * unit_second).sum(dim=-3)  # (..., 3, 3)

  eye = torch.eye(3, device=device, dtype=dtype).expand(*batch, 3, 3)
  inertia_origin = cov_origin.diagonal(dim1=-2, dim2=-1).sum(-1)[..., None, None] * eye - cov_origin
  # Parallel axis back to the CoM.
  com_cov = com[..., :, None] * com[..., None, :]
  com_inertia = (com * com).sum(-1)[..., None, None] * eye - com_cov
  inertia_com = inertia_origin - total[..., None, None] * com_inertia
  # Symmetrize: the subtractions above leave ~1e-18 asymmetry that eigh dislikes.
  inertia_com = 0.5 * (inertia_com + inertia_com.transpose(-1, -2))

  if eigh_fn is None:
    eigvals, eigvecs = torch.linalg.eigh(inertia_com.double())
  else:
    eigvals, eigvecs = eigh_fn(inertia_com)
  # eigh may return a reflection; MuJoCo needs a proper rotation. Negating one
  # eigenvector column flips the determinant without breaking the
  # eigenvalue/eigenvector pairing (-v is an eigenvector for the same value),
  # so `inertia` stays aligned with the columns of `eigvecs`.
  det = torch.linalg.det(eigvecs)
  eigvecs = torch.where(
    (det < 0)[..., None, None],
    torch.cat([-eigvecs[..., :1], eigvecs[..., 1:]], dim=-1),
    eigvecs,
  )
  iquat = _mat_to_quat(eigvecs).to(dtype)
  inertia = eigvals.to(dtype).clamp_min(1e-9)

  return total, com, iquat, inertia


def fuse_nominal(
  body: str = CHASSIS_BODY,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
  """`fuse` at nominal masses, as plain numpy. Used by `robot_constants.get_spec`."""
  moments = get_moments()[body]
  masses = nominal_mass_vector(body).double()
  mass, com, iquat, inertia = fuse(masses, moments)
  return float(mass), com.numpy(), iquat.numpy(), inertia.numpy()


##
# Reporting.
##


def report() -> None:
  """Print the full mass budget: per group, per body, and the fused totals."""
  moments = get_moments(rebuild=True) if not _CACHE_PATH.exists() else get_moments()

  grand_total = 0.0
  for body in [CHASSIS_BODY, *WHEEL_BODIES]:
    m = moments[body]
    masses = nominal_masses(body)
    mass, com, iquat, inertia = fuse_nominal(body)
    grand_total += mass

    print(f"\n=== body '{body}' ===")
    print(f"{'group':<20}{'volume':>12}{'mass':>10}{'density':>12}   CoM (m)")
    print(f"{'':<20}{'cm^3':>12}{'kg':>10}{'g/cm^3':>12}")
    for i, group in enumerate(m.groups):
      vol = float(m.volume[i])
      kg = masses[group]
      dens = f"{kg / vol / 1000:12.3f}" if vol > 0 else f"{'(point)':>12}"
      c = m.unit_com[i]
      print(
        f"{group:<20}{vol * 1e6:12.2f}{kg:10.3f}{dens}   "
        f"({c[0]:+.3f}, {c[1]:+.3f}, {c[2]:+.3f})"
      )
    print(f"{'-' * 66}")
    print(f"{'TOTAL':<20}{float(m.volume.sum()) * 1e6:12.2f}{mass:10.3f}")
    print(f"  CoM      = ({com[0]:+.4f}, {com[1]:+.4f}, {com[2]:+.4f}) m")
    print(f"  inertia  = ({inertia[0]:.5f}, {inertia[1]:.5f}, {inertia[2]:.5f}) kg m^2")
    print(f"  iquat    = ({iquat[0]:+.4f}, {iquat[1]:+.4f}, {iquat[2]:+.4f}, {iquat[3]:+.4f})")

  print("\n=== whole robot ===")
  print(f"  total mass = {grand_total:.3f} kg")


if __name__ == "__main__":
  build_moments(verbose=True)
  report()
