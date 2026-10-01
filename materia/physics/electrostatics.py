"""Long-range electrostatics of point charges.

Model
-----
Every atom carries a point charge ``q_i`` in units of the elementary charge.
The charges are never inferred: they come from an explicit
:class:`ChargeModel` that the user chose and that is recorded with every
result.  Formal oxidation states are used as charges only under the named
``formal-point-ion`` model, which is the full ionic limit of a rigid-ion
crystal and is labelled as such.

The electrostatic energy, the potential at every site, the field at every site
and the forces are computed in Gaussian-type internal units and converted with
``k_e = e^2 / (4 pi eps0) = 14.3996 eV A``.  Units of the outputs:

* energy in eV
* site potential in V (eV per e), excluding the atom's own charge
* site field in V/A, excluding the atom's own charge
* force in eV/A, ``F_i = q_i E_i``

Geometries
----------
``bulk-3d`` (all three directions periodic)
    Ewald summation (Ewald 1921; de Leeuw, Perram and Smith 1980):

    ``E = E_real + E_recip + E_self + E_background + E_surface``

    ``E_real  = 1/2 sum_{i,j,n}' q_i q_j erfc(alpha r) / r``
    ``E_recip = (2 pi / V) sum_{k != 0} exp(-k^2 / 4 alpha^2) / k^2 |S(k)|^2``
    ``E_self  = -(alpha / sqrt(pi)) sum_i q_i^2``

    The default surrounding is a conductor at infinity (tin-foil boundary
    conditions), for which ``E_surface = 0`` and the result is independent of
    how atoms are wrapped into the cell.  ``surrounding="vacuum"`` adds the
    spherical-shape term ``(2 pi / 3V) |M|^2`` with ``M = sum q_i r_i`` taken
    from the positions as stored; it is defined only for a neutral cell and
    depends on which periodic image of each atom is stored, which is the
    physics of a finite crystal in vacuum, not a numerical defect.

    A cell with a net charge has no finite periodic Coulomb energy.  It is
    refused unless ``background="uniform"`` is requested, in which case a
    uniform neutralising charge density is included and
    ``E_background = -pi Q^2 / (2 V alpha^2)`` is reported as its own term.
    The energy of a charged cell with a background carries a spurious
    dependence on cell size (Makov and Payne 1995) that is not corrected here.

``slab-2d`` (two directions periodic)
    The Ewald sum in an internally enlarged three-dimensional cell with the
    slab dipole correction of Yeh and Berkowitz (1999),
    ``E_slab = (2 pi / V') M_z^2``, where ``z`` is the slab normal and ``V'``
    the internal cell volume.  For a neutral slab that correction is exact for
    the laterally averaged (G = 0) part of the potential; the remaining error
    comes from interactions of the lateral Fourier components with their
    images, which decay as ``exp(-G_min d)`` with gap ``d``.  The internal gap
    is chosen so that this factor is below the requested accuracy.  The stored
    cell height is not used.  Charged slabs are refused: their energy depends
    on the arbitrary height of the neutralising background.

``isolated`` (no periodic direction)
    Direct pairwise Coulomb sum, exact, charged systems allowed.  A background
    is refused because there is nothing for it to neutralise.

``wire-1d`` (one periodic direction)
    Refused.  No one-dimensional Ewald method is implemented.

Numerical parameters
--------------------
With a requested accuracy ``eps`` and ``s = sqrt(-ln eps)``, the real-space
cutoff defaults to the value that balances the cost of the two sums
(Fincham 1994) for the measured cost ratio of this implementation,
``r_c^6 = c s^6 V^2 / (2 pi^3 N)`` with ``c = 0.11`` the cost of one
reciprocal term relative to one real-space pair, held between 4 and 30 A.
The ratio affects only speed, never accuracy.  Then

``alpha = s / r_c`` and ``k_c = 2 alpha s``

so that the largest neglected real-space factor is ``erfc(alpha r_c)`` and the
largest neglected reciprocal factor is ``exp(-k_c^2 / 4 alpha^2) = eps``.
Every choice is recorded.  A convergence check repeats the calculation with an
independent split (``alpha`` scaled by 0.8 and, for slabs, a 30 % larger gap)
and reports the difference; the result is labelled converged only if the two
agree within the stated energy and force tolerances.

References
----------
P. P. Ewald, Ann. Phys. 369 (1921) 253.
S. W. de Leeuw, J. W. Perram and E. R. Smith, Proc. R. Soc. Lond. A 373 (1980) 27.
D. Fincham, Mol. Simul. 13 (1994) 1.
I.-C. Yeh and M. L. Berkowitz, J. Chem. Phys. 111 (1999) 3155.
G. Makov and M. C. Payne, Phys. Rev. B 51 (1995) 4014.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import erfc

from ..core_model.cell import Cell
from ..core_model.structure import Structure
from ..core_model.units import COULOMB_K_EV_A
from ..elements import periodic_table as pt

CHARGE_MODEL_INFO_KEY = "point_charge_model"
CHARGE_MODEL_KINDS = ("per-element", "per-atom", "formal-point-ion")
SURROUNDINGS = ("tinfoil", "vacuum")
BACKGROUNDS = ("none", "uniform")
GEOMETRIES = ("bulk-3d", "slab-2d", "isolated", "wire-1d")
NEUTRALITY_TOLERANCE_E = 1e-8
COINCIDENCE_TOLERANCE_A = 1e-6
COST_RATIO = 0.11
MIN_AUTO_REAL_CUTOFF_A = 4.0
MAX_AUTO_REAL_CUTOFF_A = 30.0
MIN_SLAB_GAP_A = 10.0
CHECK_ALPHA_SCALE = 0.8
CHECK_GAP_SCALE = 1.3
PAIR_CHUNK = 2_000_000
PHASE_CHUNK = 3_000_000

REFERENCES = [
    "P. P. Ewald, Ann. Phys. 369 (1921) 253",
    "S. W. de Leeuw, J. W. Perram and E. R. Smith, Proc. R. Soc. Lond. A 373 (1980) 27",
    "D. Fincham, Mol. Simul. 13 (1994) 1",
    "I.-C. Yeh and M. L. Berkowitz, J. Chem. Phys. 111 (1999) 3155",
    "G. Makov and M. C. Payne, Phys. Rev. B 51 (1995) 4014",
]


class ElectrostaticsError(Exception):
    """Base class for electrostatics failures."""


class ChargeModelError(ElectrostaticsError):
    """The charge model is missing, incomplete or inconsistent with the structure."""


class UndefinedElectrostatics(ElectrostaticsError):
    """The requested configuration has no defined electrostatic energy."""


class ElectrostaticsCancelled(ElectrostaticsError):
    """The caller cancelled the calculation."""


@dataclass(frozen=True)
class ChargeModel:
    """Where the point charges come from.

    ``per-element``
        One charge per element, supplied by the user with a stated source.
        Every element in the structure must be listed; an element that is not
        listed is refused rather than assumed neutral.
    ``per-atom``
        One charge per atom id.  Every atom must be listed and every listed id
        must exist, so a model assigned before atoms were added or removed is
        refused instead of silently reused.
    ``formal-point-ion``
        The structure's formal charges, taken literally as rigid point
        charges.  This is the full ionic limit.  Real ions in a solid carry
        less charge than their oxidation state, so energies from this model
        are upper bounds on the magnitude of the ionic binding, which is how
        Madelung energies are conventionally quoted.
    """

    kind: str
    by_element: Tuple[Tuple[str, float], ...] = ()
    by_atom_id: Tuple[Tuple[int, float], ...] = ()
    source: str = ""

    def __post_init__(self) -> None:
        if self.kind not in CHARGE_MODEL_KINDS:
            raise ChargeModelError(
                f"Unknown charge model {self.kind!r}. Available: "
                f"{', '.join(CHARGE_MODEL_KINDS)}.")
        elements = tuple(sorted((pt.symbol(k), float(v)) for k, v in self.by_element))
        atoms = tuple(sorted((int(k), float(v)) for k, v in self.by_atom_id))
        object.__setattr__(self, "by_element", elements)
        object.__setattr__(self, "by_atom_id", atoms)
        object.__setattr__(self, "source", str(self.source or "").strip())
        values = [v for _, v in elements] + [v for _, v in atoms]
        if any(not math.isfinite(v) for v in values):
            raise ChargeModelError("Every point charge must be a finite number.")
        if self.kind == "per-element":
            if not elements:
                raise ChargeModelError("A per-element charge model needs at least one element.")
            if len({k for k, _ in elements}) != len(elements):
                raise ChargeModelError("An element is listed twice in the charge model.")
            if atoms:
                raise ChargeModelError("A per-element charge model cannot also list atoms.")
        elif self.kind == "per-atom":
            if not atoms:
                raise ChargeModelError("A per-atom charge model needs at least one atom.")
            if len({k for k, _ in atoms}) != len(atoms):
                raise ChargeModelError("An atom id is listed twice in the charge model.")
            if elements:
                raise ChargeModelError("A per-atom charge model cannot also list elements.")
        elif elements or atoms:
            raise ChargeModelError(
                "The formal-point-ion model takes the structure's formal charges; "
                "it does not accept a charge table.")

    @staticmethod
    def per_element(charges: Mapping[str, float], source: str = "") -> "ChargeModel":
        return ChargeModel("per-element", by_element=tuple(charges.items()), source=source)

    @staticmethod
    def per_atom(charges: Mapping[int, float], source: str = "") -> "ChargeModel":
        return ChargeModel("per-atom", by_atom_id=tuple(charges.items()), source=source)

    @staticmethod
    def formal_point_ion(source: str = "") -> "ChargeModel":
        return ChargeModel("formal-point-ion", source=source)

    def charges(self, structure: Structure) -> np.ndarray:
        """The point charge of every atom, in e, in structure order."""
        n = len(structure)
        if self.kind == "formal-point-ion":
            return np.array(structure.formal_charges, dtype=float)
        if self.kind == "per-element":
            table = dict(self.by_element)
            symbols = [pt.symbol(int(z)) for z in structure.numbers]
            missing = sorted(set(symbols) - set(table))
            if missing:
                raise ChargeModelError(
                    f"The charge model gives no charge for {', '.join(missing)}. "
                    "An unlisted element is not assumed neutral; list it explicitly, "
                    "with 0 if that is what you mean.")
            return np.array([table[s] for s in symbols], dtype=float)
        table = dict(self.by_atom_id)
        ids = [int(i) for i in structure.ids]
        missing = [i for i in ids if i not in table]
        extra = sorted(set(table) - set(ids))
        if missing or extra:
            parts = []
            if missing:
                parts.append(f"{len(missing)} atom(s) have no charge "
                             f"(ids {', '.join(str(i) for i in missing[:8])}"
                             f"{', ...' if len(missing) > 8 else ''})")
            if extra:
                parts.append(f"{len(extra)} listed id(s) no longer exist "
                             f"({', '.join(str(i) for i in extra[:8])}"
                             f"{', ...' if len(extra) > 8 else ''})")
            raise ChargeModelError(
                "The per-atom charge model does not match this structure: "
                + "; ".join(parts) + ". Reassign the charges for the current atoms.")
        return np.array([table[i] for i in ids], dtype=float) if n else np.zeros(0)

    def describe(self) -> str:
        if self.kind == "formal-point-ion":
            text = ("Formal-point-ion model: each atom's formal charge is taken as a "
                    "rigid point charge (the full ionic limit). Real ionic charges in "
                    "a solid are smaller than oxidation states.")
        elif self.kind == "per-element":
            text = ("Per-element point charges supplied by the user: "
                    + ", ".join(f"{k} {v:+g} e" for k, v in self.by_element) + ".")
        else:
            text = (f"Per-atom point charges supplied by the user for "
                    f"{len(self.by_atom_id)} atoms.")
        if self.source:
            text += f" Stated source: {self.source}."
        return text

    def as_dict(self) -> dict:
        return {"kind": self.kind,
                "by_element": {k: v for k, v in self.by_element},
                "by_atom_id": {str(k): v for k, v in self.by_atom_id},
                "source": self.source,
                "unit": "e"}

    @staticmethod
    def from_dict(d: Mapping) -> "ChargeModel":
        if not isinstance(d, Mapping) or "kind" not in d:
            raise ChargeModelError("A charge model needs a 'kind'.")
        return ChargeModel(
            str(d["kind"]),
            by_element=tuple((d.get("by_element") or {}).items()),
            by_atom_id=tuple((int(k), v) for k, v in (d.get("by_atom_id") or {}).items()),
            source=str(d.get("source", "")))


def stored_charge_model(structure: Structure) -> Optional[ChargeModel]:
    """The charge model recorded on a structure, or ``None``."""
    data = structure.info.get(CHARGE_MODEL_INFO_KEY)
    if not data:
        return None
    return ChargeModel.from_dict(data)


def store_charge_model(structure: Structure, model: Optional[ChargeModel]) -> None:
    if model is None:
        structure.info.pop(CHARGE_MODEL_INFO_KEY, None)
    else:
        structure.info[CHARGE_MODEL_INFO_KEY] = model.as_dict()


@dataclass(frozen=True)
class Geometry:
    """How the Coulomb sum has to be performed for a given cell."""

    kind: str
    periodic_axes: Tuple[int, ...]
    description: str
    supported: bool
    reason: str = ""

    def as_dict(self) -> dict:
        return {"kind": self.kind, "periodic_axes": list(self.periodic_axes),
                "description": self.description, "supported": self.supported,
                "reason": self.reason}


def classify(cell: Cell) -> Geometry:
    """Classify a cell by its periodicity."""
    axes = tuple(i for i, p in enumerate(cell.pbc) if p)
    m = cell.matrix
    if len(axes) == 3:
        if cell.volume < 1e-9:
            return Geometry("bulk-3d", axes, "three-dimensional periodic", False,
                            "The cell is periodic in three directions but has zero volume.")
        return Geometry("bulk-3d", axes,
                        "three-dimensional periodic crystal (Ewald summation)", True)
    if len(axes) == 2:
        a, b = m[axes[0]], m[axes[1]]
        if np.linalg.norm(np.cross(a, b)) < 1e-9:
            return Geometry("slab-2d", axes, "two-dimensional periodic slab", False,
                            "The two periodic lattice vectors are collinear.")
        return Geometry("slab-2d", axes,
                        "two-dimensional periodic slab (Ewald with the Yeh-Berkowitz "
                        "slab correction)", True)
    if len(axes) == 1:
        return Geometry("wire-1d", axes, "one-dimensional periodic wire", False,
                        "Materia has no one-dimensional Ewald method, so the Coulomb "
                        "energy of a wire geometry is not computed. Make the cell "
                        "periodic in three directions with enough vacuum, knowing that "
                        "this includes interactions between wire images, or treat a "
                        "finite segment as an isolated cluster.")
    return Geometry("isolated", (), "isolated cluster (direct Coulomb sum)", True)


@dataclass
class EwaldSettings:
    """Numerical and physical settings for one electrostatics calculation.

    ``accuracy`` is the relative size of the largest neglected term in either
    sum.  ``alpha_per_A``, ``real_cutoff_A`` and ``kspace_cutoff_per_A``
    override the automatic choice.  ``surrounding`` and ``background`` are
    physical choices, not numerical ones, and change the answer.
    """

    accuracy: float = 1e-8
    alpha_per_A: Optional[float] = None
    real_cutoff_A: Optional[float] = None
    kspace_cutoff_per_A: Optional[float] = None
    surrounding: str = "tinfoil"
    background: str = "none"
    check_convergence: bool = True
    energy_tolerance_eV_per_atom: float = 1e-6
    force_tolerance_eV_A: float = 1e-4

    def validate(self) -> "EwaldSettings":
        if not (0.0 < float(self.accuracy) < 1e-2):
            raise ValueError(
                f"accuracy must lie between 0 and 1e-2; got {self.accuracy}.")
        for name in ("alpha_per_A", "real_cutoff_A", "kspace_cutoff_per_A"):
            value = getattr(self, name)
            if value is not None and not (math.isfinite(float(value)) and float(value) > 0):
                raise ValueError(f"{name} must be a positive number when given.")
        if self.surrounding not in SURROUNDINGS:
            raise ValueError(f"surrounding must be one of {SURROUNDINGS}; got "
                             f"{self.surrounding!r}.")
        if self.background not in BACKGROUNDS:
            raise ValueError(f"background must be one of {BACKGROUNDS}; got "
                             f"{self.background!r}.")
        if not (self.energy_tolerance_eV_per_atom > 0 and self.force_tolerance_eV_A > 0):
            raise ValueError("Convergence tolerances must be positive.")
        return self

    def as_dict(self) -> dict:
        return {"accuracy": self.accuracy, "alpha_per_A": self.alpha_per_A,
                "real_cutoff_A": self.real_cutoff_A,
                "kspace_cutoff_per_A": self.kspace_cutoff_per_A,
                "surrounding": self.surrounding, "background": self.background,
                "check_convergence": bool(self.check_convergence),
                "energy_tolerance_eV_per_atom": self.energy_tolerance_eV_per_atom,
                "force_tolerance_eV_A": self.force_tolerance_eV_A}

    @staticmethod
    def from_dict(d: Optional[Mapping]) -> "EwaldSettings":
        d = dict(d or {})
        known = EwaldSettings().as_dict().keys()
        unknown = sorted(set(d) - set(known))
        if unknown:
            raise ValueError(f"Unknown electrostatics setting(s): {', '.join(unknown)}. "
                             f"Known: {', '.join(sorted(known))}.")
        out = EwaldSettings()
        for key, value in d.items():
            if value is None or value == "":
                setattr(out, key, None if key in ("alpha_per_A", "real_cutoff_A",
                                                  "kspace_cutoff_per_A") else value)
                continue
            if key in ("surrounding", "background"):
                setattr(out, key, str(value))
            elif key == "check_convergence":
                setattr(out, key, bool(value))
            else:
                setattr(out, key, float(value))
        return out.validate()


@dataclass
class ElectrostaticsOutput:
    """Everything one electrostatics evaluation produced."""

    geometry: Geometry
    charges_e: np.ndarray
    energy_eV: float
    components_eV: Dict[str, float]
    site_potential_V: np.ndarray
    site_field_V_A: np.ndarray
    forces_eV_A: np.ndarray
    parameters: Dict[str, object]
    check: Optional[Dict[str, object]] = None
    converged: bool = True
    convergence_message: str = ""
    wall_time_s: float = 0.0
    log: List[str] = field(default_factory=list)


Progress = Callable[[float, str], Optional[bool]]


class _Reporter:
    def __init__(self, progress: Optional[Progress], cancelled: Optional[Callable[[], bool]],
                 lo: float = 0.0, hi: float = 1.0) -> None:
        self.progress = progress
        self.cancelled = cancelled
        self.lo = lo
        self.hi = hi

    def sub(self, lo: float, hi: float) -> "_Reporter":
        span = self.hi - self.lo
        return _Reporter(self.progress, self.cancelled,
                         self.lo + span * lo, self.lo + span * hi)

    def __call__(self, fraction: float, message: str) -> None:
        if self.cancelled is not None and self.cancelled():
            raise ElectrostaticsCancelled("Electrostatics calculation cancelled.")
        if self.progress is not None:
            answer = self.progress(self.lo + (self.hi - self.lo) * float(fraction), message)
            if answer is False:
                raise ElectrostaticsCancelled("Electrostatics calculation cancelled.")


def charge_accounting(structure: Structure, charges: np.ndarray) -> dict:
    """Totals of the point charges, overall and per element."""
    symbols = [pt.symbol(int(z)) for z in structure.numbers]
    per: Dict[str, dict] = {}
    for sym, q in zip(symbols, charges):
        entry = per.setdefault(sym, {"count": 0, "total_e": 0.0, "min_e": q, "max_e": q})
        entry["count"] += 1
        entry["total_e"] += float(q)
        entry["min_e"] = float(min(entry["min_e"], q))
        entry["max_e"] = float(max(entry["max_e"], q))
    total = float(np.sum(charges)) if len(charges) else 0.0
    return {"total_charge_e": total,
            "sum_abs_charge_e": float(np.sum(np.abs(charges))) if len(charges) else 0.0,
            "sum_sq_charge_e2": float(np.sum(charges ** 2)) if len(charges) else 0.0,
            "neutral": abs(total) <= NEUTRALITY_TOLERANCE_E * max(1, len(charges)),
            "per_element": per,
            "formal_total_e": float(structure.total_charge()),
            "unit": "e"}


def _image_offsets(matrix: np.ndarray, periodic: Sequence[bool], cutoff: float) -> np.ndarray:
    reps = []
    for axis in range(3):
        if not periodic[axis]:
            reps.append(0)
            continue
        others = [matrix[k] for k in range(3) if k != axis]
        normal = np.cross(others[0], others[1])
        width = abs(matrix[axis] @ normal) / np.linalg.norm(normal)
        reps.append(int(math.ceil(cutoff / width)))
    grid = np.array([(a, b, c)
                     for a in range(-reps[0], reps[0] + 1)
                     for b in range(-reps[1], reps[1] + 1)
                     for c in range(-reps[2], reps[2] + 1)], dtype=float)
    return grid @ matrix


def _coincidence_error(i: int, j: int, structure_ids: Optional[np.ndarray]) -> UndefinedElectrostatics:
    label = (f"atoms {int(structure_ids[i])} and {int(structure_ids[j])}"
             if structure_ids is not None else f"atoms at indices {i} and {j}")
    return UndefinedElectrostatics(
        f"The Coulomb energy is infinite: {label} (or a periodic image) coincide. "
        "Point charges at the same position have no defined interaction energy.")


def _check_coincidence(tree: cKDTree, pos: np.ndarray, n: int,
                       ids: Optional[np.ndarray]) -> None:
    counts = tree.query_ball_point(pos, COINCIDENCE_TOLERANCE_A, return_length=True)
    bad = np.nonzero(np.asarray(counts) > 1)[0]
    if bad.size == 0:
        return
    i = int(bad[0])
    hits = [g for g in tree.query_ball_point(pos[i], COINCIDENCE_TOLERANCE_A)]
    other = next((g % n for g in hits if g % n != i), i)
    raise _coincidence_error(i, int(other), ids)


def _real_space(pos: np.ndarray, q: np.ndarray, matrix: np.ndarray, alpha: float,
                rc: float, report: _Reporter, ids: Optional[np.ndarray]
                ) -> Tuple[np.ndarray, np.ndarray, int]:
    n = len(pos)
    offsets = _image_offsets(matrix, (True, True, True), max(rc, COINCIDENCE_TOLERANCE_A))
    ghosts = (pos[None, :, :] + offsets[:, None, :]).reshape(-1, 3)
    tree = cKDTree(ghosts)
    _check_coincidence(tree, pos, n, ids)
    phi = np.zeros(n)
    fld = np.zeros((n, 3))
    pairs = 0
    per_atom = max(1, int(4.0 / 3.0 * math.pi * rc ** 3 * n / abs(np.linalg.det(matrix))) + 1)
    step = max(1, PAIR_CHUNK // per_atom)
    two_over_sqrt_pi = 2.0 / math.sqrt(math.pi)
    for start in range(0, n, step):
        report(start / n, f"real-space sum, atoms {start}-{min(n, start + step)} of {n}")
        stop = min(n, start + step)
        found = cKDTree(pos[start:stop]).sparse_distance_matrix(tree, rc,
                                                                output_type="ndarray")
        if found.size == 0:
            continue
        keep = found["v"] > COINCIDENCE_TOLERANCE_A
        found = found[keep]
        local = found["i"].astype(np.int64)
        flat = found["j"].astype(np.int64)
        i_idx = local + start
        D = ghosts[flat] - pos[i_idx]
        d = found["v"]
        j_idx = flat % n
        pairs += int(d.size)
        qj = q[j_idx]
        ad = alpha * d
        e = erfc(ad) / d
        g = (e + two_over_sqrt_pi * alpha * np.exp(-ad * ad)) / (d * d)
        phi[start:stop] += np.bincount(local, weights=qj * e, minlength=stop - start)
        vec = -(qj * g)[:, None] * D
        for axis in range(3):
            fld[start:stop, axis] += np.bincount(local, weights=vec[:, axis],
                                                 minlength=stop - start)
    report(1.0, "real-space sum done")
    return phi, fld, pairs


def _kvectors(matrix: np.ndarray, kc: float) -> np.ndarray:
    recip = 2.0 * math.pi * np.linalg.inv(matrix).T
    lengths = np.linalg.norm(matrix, axis=1)
    m = [int(math.floor(kc * lengths[a] / (2.0 * math.pi))) for a in range(3)]
    n1 = np.arange(-m[0], m[0] + 1)
    n2 = np.arange(-m[1], m[1] + 1)
    n3 = np.arange(-m[2], m[2] + 1)
    grid = np.stack(np.meshgrid(n1, n2, n3, indexing="ij"), axis=-1).reshape(-1, 3)
    half = ((grid[:, 0] > 0)
            | ((grid[:, 0] == 0) & (grid[:, 1] > 0))
            | ((grid[:, 0] == 0) & (grid[:, 1] == 0) & (grid[:, 2] > 0)))
    grid = grid[half]
    k = grid.astype(float) @ recip
    k2 = np.einsum("ij,ij->i", k, k)
    inside = k2 <= kc * kc
    k = k[inside]
    order = np.lexsort((grid[inside][:, 2], grid[inside][:, 1], grid[inside][:, 0]))
    return k[order]


def _reciprocal(pos: np.ndarray, q: np.ndarray, matrix: np.ndarray, alpha: float,
                kc: float, report: _Reporter) -> Tuple[np.ndarray, np.ndarray, int]:
    n = len(pos)
    volume = abs(float(np.linalg.det(matrix)))
    k = _kvectors(matrix, kc)
    phi = np.zeros(n)
    fld = np.zeros((n, 3))
    if len(k) == 0:
        report(1.0, "no reciprocal vectors inside the cutoff")
        return phi, fld, 0
    k2 = np.einsum("ij,ij->i", k, k)
    weight = (2.0 * 4.0 * math.pi / volume) * np.exp(-k2 / (4.0 * alpha * alpha)) / k2
    step = max(1, PHASE_CHUNK // max(1, n))
    for start in range(0, len(k), step):
        report(start / len(k), f"reciprocal sum, vectors {start}-"
                               f"{min(len(k), start + step)} of {len(k)}")
        kk = k[start:start + step]
        w = weight[start:start + step]
        phase = pos @ kk.T
        c = np.cos(phase)
        s = np.sin(phase)
        s_re = q @ c
        s_im = q @ s
        phi += c @ (w * s_re) + s @ (w * s_im)
        fld -= (c * (w * s_im) - s * (w * s_re)) @ kk
    report(1.0, "reciprocal sum done")
    return phi, fld, int(len(k))


def choose_parameters(n: int, volume: float, settings: EwaldSettings,
                      alpha_scale: float = 1.0) -> Dict[str, float]:
    """Splitting parameter and cutoffs for a requested accuracy."""
    s = math.sqrt(-math.log(settings.accuracy))
    if settings.alpha_per_A is not None and alpha_scale == 1.0:
        alpha = float(settings.alpha_per_A)
        chosen = "user"
    else:
        rc_balanced = (COST_RATIO * s ** 6 * volume ** 2
                       / (2.0 * math.pi ** 3 * max(n, 1))) ** (1.0 / 6.0)
        rc_balanced = min(max(rc_balanced, MIN_AUTO_REAL_CUTOFF_A), MAX_AUTO_REAL_CUTOFF_A)
        alpha = s / rc_balanced * alpha_scale
        chosen = "measured cost balance of the real and reciprocal sums"
    use_user = alpha_scale == 1.0
    rc = (float(settings.real_cutoff_A) if use_user and settings.real_cutoff_A is not None
          else s / alpha)
    kc = (float(settings.kspace_cutoff_per_A)
          if use_user and settings.kspace_cutoff_per_A is not None else 2.0 * alpha * s)
    return {"alpha_per_A": alpha, "real_cutoff_A": rc, "kspace_cutoff_per_A": kc,
            "alpha_choice": chosen,
            "neglected_real_factor": float(erfc(alpha * rc)),
            "neglected_reciprocal_factor": float(math.exp(-kc * kc / (4.0 * alpha * alpha)))}


def _slab_frame(matrix: np.ndarray, axes: Tuple[int, ...]) -> Tuple[np.ndarray, np.ndarray]:
    a, b = matrix[axes[0]], matrix[axes[1]]
    normal = np.cross(a, b)
    normal /= np.linalg.norm(normal)
    return np.array([a, b]), normal


def _min_lateral_g(lateral: np.ndarray, normal: np.ndarray) -> float:
    full = np.array([lateral[0], lateral[1], normal])
    recip = 2.0 * math.pi * np.linalg.inv(full).T
    b1, b2 = recip[0], recip[1]
    best = math.inf
    for i in range(-2, 3):
        for j in range(-2, 3):
            if i == 0 and j == 0:
                continue
            best = min(best, float(np.linalg.norm(i * b1 + j * b2)))
    return best


def _periodic_sum(pos: np.ndarray, q: np.ndarray, matrix: np.ndarray,
                  params: Dict[str, float], report: _Reporter,
                  ids: Optional[np.ndarray]) -> Dict[str, object]:
    alpha = params["alpha_per_A"]
    phi_r, fld_r, pairs = _real_space(pos, q, matrix, alpha, params["real_cutoff_A"],
                                      report.sub(0.0, 0.45), ids)
    phi_k, fld_k, nk = _reciprocal(pos, q, matrix, alpha, params["kspace_cutoff_per_A"],
                                   report.sub(0.45, 1.0))
    phi_self = -2.0 * alpha / math.sqrt(math.pi) * q
    return {"phi_real": phi_r, "phi_recip": phi_k, "phi_self": phi_self,
            "field": fld_r + fld_k, "pairs": pairs, "n_kvectors": nk}


def _evaluate(structure: Structure, charges: np.ndarray, geometry: Geometry,
              settings: EwaldSettings, report: _Reporter,
              alpha_scale: float = 1.0, gap_scale: float = 1.0) -> dict:
    pos = np.array(structure.positions, dtype=float)
    q = np.asarray(charges, dtype=float)
    n = len(q)
    ids = np.array(structure.ids)
    total = float(q.sum())
    phi_parts: Dict[str, np.ndarray] = {}
    field_total = np.zeros((n, 3))
    parameters: Dict[str, object] = {}

    if geometry.kind == "isolated":
        phi, fld, pairs = _direct(pos, q, report, ids)
        phi_parts["direct"] = phi
        field_total = fld
        parameters.update({"method": "direct pairwise sum", "pairs": pairs})
    elif geometry.kind == "bulk-3d":
        matrix = structure.cell.matrix
        volume = structure.cell.volume
        params = choose_parameters(n, volume, settings, alpha_scale)
        parts = _periodic_sum(pos, q, matrix, params, report, ids)
        phi_parts["real"] = parts["phi_real"]
        phi_parts["reciprocal"] = parts["phi_recip"]
        phi_parts["self"] = parts["phi_self"]
        field_total = parts["field"]
        if settings.background == "uniform":
            phi_parts["background"] = np.full(n, -math.pi * total /
                                              (volume * params["alpha_per_A"] ** 2))
        if settings.surrounding == "vacuum":
            moment = q @ pos
            phi_parts["surface"] = (4.0 * math.pi / (3.0 * volume)) * (pos @ moment)
            field_total = field_total - (4.0 * math.pi / (3.0 * volume)) * moment[None, :]
        parameters.update(params)
        parameters.update({"method": "Ewald summation",
                           "real_space_pairs": parts["pairs"],
                           "n_kvectors_half_space": parts["n_kvectors"],
                           "cell_volume_A3": volume})
    elif geometry.kind == "slab-2d":
        lateral, normal = _slab_frame(structure.cell.matrix, geometry.periodic_axes)
        z = pos @ normal
        thickness = float(z.max() - z.min()) if n else 0.0
        g_min = _min_lateral_g(lateral, normal)
        s2 = -math.log(settings.accuracy)
        gap = max(s2 / g_min, 2.0 * thickness, MIN_SLAB_GAP_A) * gap_scale
        height = thickness + gap
        matrix = np.array([lateral[0], lateral[1], normal * height])
        volume = abs(float(np.linalg.det(matrix)))
        shifted = pos - normal[None, :] * float(z.min())
        params = choose_parameters(n, volume, settings, alpha_scale)
        parts = _periodic_sum(shifted, q, matrix, params, report, ids)
        phi_parts["real"] = parts["phi_real"]
        phi_parts["reciprocal"] = parts["phi_recip"]
        phi_parts["self"] = parts["phi_self"]
        zs = shifted @ normal
        moment_z = float(q @ zs)
        phi_parts["slab_correction"] = (4.0 * math.pi / volume) * moment_z * zs
        field_total = parts["field"] - (4.0 * math.pi / volume) * moment_z * normal[None, :]
        parameters.update(params)
        parameters.update({"method": "Ewald summation with Yeh-Berkowitz slab correction",
                           "real_space_pairs": parts["pairs"],
                           "n_kvectors_half_space": parts["n_kvectors"],
                           "slab_thickness_A": thickness,
                           "internal_vacuum_gap_A": gap,
                           "internal_cell_height_A": height,
                           "internal_cell_volume_A3": volume,
                           "slab_normal": normal.tolist(),
                           "slab_dipole_e_A": moment_z,
                           "min_lateral_G_per_A": g_min,
                           "lateral_image_decay_factor": math.exp(-g_min * gap)})
    else:
        raise UndefinedElectrostatics(geometry.reason)

    k = COULOMB_K_EV_A
    potential = sum(phi_parts.values()) if phi_parts else np.zeros(n)
    components = {name: float(0.5 * k * (q @ part)) for name, part in phi_parts.items()}
    energy = float(sum(components.values()))
    field_v = k * field_total
    return {"energy_eV": energy, "components_eV": components,
            "site_potential_V": k * potential, "site_field_V_A": field_v,
            "forces_eV_A": q[:, None] * field_v, "parameters": parameters}


def _direct(pos: np.ndarray, q: np.ndarray, report: _Reporter,
            ids: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, int]:
    n = len(pos)
    phi = np.zeros(n)
    fld = np.zeros((n, 3))
    step = max(1, PHASE_CHUNK // max(1, n))
    for start in range(0, n, step):
        report(start / max(1, n), f"direct sum, atoms {start}-{min(n, start + step)} of {n}")
        block = pos[start:start + step]
        D = pos[None, :, :] - block[:, None, :]
        d = np.sqrt(np.einsum("ijk,ijk->ij", D, D))
        rows = np.arange(block.shape[0])
        d[rows, start + rows] = np.inf
        if (d < COINCIDENCE_TOLERANCE_A).any():
            a, b = np.argwhere(d < COINCIDENCE_TOLERANCE_A)[0]
            raise _coincidence_error(int(start + a), int(b), ids)
        inv = 1.0 / d
        phi[start:start + step] = inv @ q
        fld[start:start + step] = -np.einsum("ij,ijk->ik", q[None, :] * inv ** 3, D)
    report(1.0, "direct sum done")
    return phi, fld, n * (n - 1)


def refusal(structure: Structure, charges: np.ndarray, geometry: Geometry,
            settings: EwaldSettings) -> str:
    """Why this configuration has no defined electrostatic energy, or ``""``."""
    if not geometry.supported:
        return geometry.reason
    total = float(np.sum(charges)) if len(charges) else 0.0
    neutral = abs(total) <= NEUTRALITY_TOLERANCE_E * max(1, len(charges))
    if geometry.kind == "isolated":
        if settings.background != "none":
            return ("A neutralising background has no meaning for an isolated cluster, "
                    "whose Coulomb energy is finite whatever its net charge. Use "
                    "background 'none'.")
        if settings.surrounding != "tinfoil":
            return ("The surrounding medium applies only to a three-dimensional "
                    "periodic crystal. An isolated cluster is summed directly.")
        return ""
    if geometry.kind == "slab-2d":
        if settings.surrounding != "tinfoil":
            return ("The surrounding medium applies only to a three-dimensional "
                    "periodic crystal. A slab uses the Yeh-Berkowitz correction instead.")
        if not neutral:
            return (f"The slab carries a net point charge of {total:+.6g} e. A charged "
                    "two-dimensional slab has no defined energy under this method: a "
                    "neutralising background would fill the vacuum gap and the energy "
                    "would depend on its arbitrary height. Make the slab neutral.")
        if settings.background != "none":
            return ("A neutralising background is not applied to slabs. The slab must "
                    "be neutral.")
        return ""
    if not neutral:
        if settings.background != "uniform":
            return (f"The cell carries a net point charge of {total:+.6g} e. The Coulomb "
                    "energy of a charged periodic crystal is infinite. Either make the "
                    "cell neutral or explicitly request a uniform neutralising "
                    "background, whose energy is then reported as its own term.")
        if settings.surrounding == "vacuum":
            return ("The vacuum surface term depends on the choice of origin for a "
                    "charged cell and is undefined. Use tin-foil boundary conditions.")
    return ""


def compute(structure: Structure, model: ChargeModel,
            settings: Optional[EwaldSettings] = None,
            progress: Optional[Progress] = None,
            cancelled: Optional[Callable[[], bool]] = None) -> ElectrostaticsOutput:
    """Electrostatic energy, site potentials, fields and forces.

    Raises :class:`ChargeModelError` when the model does not fit the structure,
    :class:`UndefinedElectrostatics` when the configuration has no defined
    electrostatic energy, and :class:`ElectrostaticsCancelled` when the caller
    cancels.
    """
    t0 = time.perf_counter()
    settings = (settings or EwaldSettings()).validate()
    geometry = classify(structure.cell)
    charges = model.charges(structure)
    reason = refusal(structure, charges, geometry, settings)
    if reason:
        raise UndefinedElectrostatics(reason)
    n = len(structure)
    if n == 0:
        raise UndefinedElectrostatics("The structure has no atoms.")
    report = _Reporter(progress, cancelled)
    check_wanted = settings.check_convergence and geometry.kind != "isolated"
    main_report = report.sub(0.0, 0.6 if check_wanted else 1.0)
    primary = _evaluate(structure, charges, geometry, settings, main_report)
    log: List[str] = []
    check: Optional[Dict[str, object]] = None
    converged = True
    message = ""
    if geometry.kind == "isolated":
        message = "Direct pairwise sum: exact up to floating-point rounding."
    elif check_wanted:
        second = _evaluate(structure, charges, geometry, settings, report.sub(0.6, 1.0),
                           alpha_scale=CHECK_ALPHA_SCALE, gap_scale=CHECK_GAP_SCALE)
        d_energy = abs(second["energy_eV"] - primary["energy_eV"])
        d_force = float(np.abs(second["forces_eV_A"] - primary["forces_eV_A"]).max())
        d_potential = float(np.abs(second["site_potential_V"]
                                   - primary["site_potential_V"]).max())
        e_tol = settings.energy_tolerance_eV_per_atom * n
        converged = d_energy <= e_tol and d_force <= settings.force_tolerance_eV_A
        check = {"alpha_per_A": second["parameters"]["alpha_per_A"],
                 "real_cutoff_A": second["parameters"]["real_cutoff_A"],
                 "kspace_cutoff_per_A": second["parameters"]["kspace_cutoff_per_A"],
                 "energy_eV": second["energy_eV"],
                 "energy_difference_eV": d_energy,
                 "energy_tolerance_eV": e_tol,
                 "max_force_difference_eV_A": d_force,
                 "force_tolerance_eV_A": settings.force_tolerance_eV_A,
                 "max_potential_difference_V": d_potential}
        if geometry.kind == "slab-2d":
            check["internal_vacuum_gap_A"] = second["parameters"]["internal_vacuum_gap_A"]
        message = (
            f"Independent split agrees: |dE| = {d_energy:.3g} eV (tolerance {e_tol:.3g}), "
            f"max |dF| = {d_force:.3g} eV/A (tolerance {settings.force_tolerance_eV_A:g})."
            if converged else
            f"Not converged: an independent split differs by |dE| = {d_energy:.3g} eV "
            f"(tolerance {e_tol:.3g}) and max |dF| = {d_force:.3g} eV/A (tolerance "
            f"{settings.force_tolerance_eV_A:g}). Tighten the accuracy or remove the "
            "parameter overrides.")
    else:
        converged = False
        message = ("Convergence was not checked. The parameters bound the largest "
                   "neglected terms, but no independent evaluation confirms the result.")
    if settings.background == "uniform" and geometry.kind == "bulk-3d":
        log.append("A uniform neutralising background is included. The energy of a "
                   "charged periodic cell depends on the cell size (Makov and Payne 1995); "
                   "no finite-size correction is applied.")
    if settings.surrounding == "vacuum":
        log.append("Vacuum surrounding: the surface term depends on which periodic image "
                   "of each atom is stored, as for a finite crystal in vacuum.")
    return ElectrostaticsOutput(
        geometry=geometry, charges_e=charges,
        energy_eV=primary["energy_eV"], components_eV=primary["components_eV"],
        site_potential_V=primary["site_potential_V"],
        site_field_V_A=primary["site_field_V_A"],
        forces_eV_A=primary["forces_eV_A"],
        parameters=primary["parameters"], check=check, converged=converged,
        convergence_message=message, wall_time_s=time.perf_counter() - t0, log=log)
