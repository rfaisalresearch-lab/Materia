"""Surface reconstruction generators.

A reconstruction is produced in two separable stages.  The *generator* builds
the reconstructed connectivity and periodicity from the symmetry of the
truncated slab: which surface atoms pair, dimerise, are removed or are added,
and along which crystallographic direction.  The *relaxation* then finds the
geometry, using whichever interatomic model the material recommends.  No
reconstructed bond length, buckling amplitude or registry shift shipped with
Materia is a hand-entered number: the generator supplies topology only, and
every distance reported afterwards is the output of a solver run in the
session.

Published structural parameters are carried alongside, in the material
definition under ``reconstructions[].reference_geometry``, purely so that a
computed geometry can be compared against measurement.  They are never used to
place an atom.

Declared-but-unavailable reconstructions raise :class:`ReconstructionNotImplemented`,
which carries a machine-readable unsupported record.  Nothing in this module
approximates a reconstruction it cannot build.

Every reconstructed slab records which of three states its geometry is in, and
the three are never conflated:

``GEOMETRY_CONVERGED``
    The relaxation reached its force target.  The geometry is the minimum of
    the model that ran, the record's origin is ``calculated``, and the geometry
    may be compared against published structural data.
``GEOMETRY_NOT_CONVERGED``
    A relaxation ran but stopped early, on its step budget or otherwise.  The
    atoms sit somewhere between the construction guess and the minimum, so no
    distance measured from them is a prediction: the origin is ``estimated``,
    the provenance states the residual force against the target, and
    :func:`compare_to_reference` refuses to score it.
``GEOMETRY_UNRELAXED``
    No relaxation was attempted.  Separations are the generator's starting
    guess and nothing more.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core_model.structure import Structure
from ..materials.schema import MaterialDefinition, Reconstruction
from ..provenance import Fidelity, Origin, Provenance, Result, unsupported
from ..version import __version__


GEOMETRY_CONVERGED = "converged"
GEOMETRY_NOT_CONVERGED = "not-converged"
GEOMETRY_UNRELAXED = "unrelaxed"

GEOMETRY_STATUS_TEXT = {
    GEOMETRY_CONVERGED: "relaxed to a converged geometry",
    GEOMETRY_NOT_CONVERGED: "relaxation attempted but not converged",
    GEOMETRY_UNRELAXED: "not relaxed",
}


class ReconstructionError(ValueError):
    """Raised when a reconstruction cannot be applied to the slab as given."""


class ReconstructionNotImplemented(ReconstructionError):
    """Raised when a reconstruction is declared for a material but has no generator."""

    def __init__(self, material_id: str, reconstruction_id: str, reason: str,
                 reference: str = "", suggested: Optional[Sequence[str]] = None) -> None:
        super().__init__(reason)
        self.material_id = material_id
        self.reconstruction_id = reconstruction_id
        self.reason = reason
        self.reference = reference
        self.suggested = list(suggested or [])

    def as_result(self) -> Result:
        result = unsupported(
            name=f"reconstruction:{self.reconstruction_id}",
            model="structure-builder/reconstruction",
            reason=self.reason,
            suggested_models=self.suggested,
        )
        result.provenance.references = [self.reference] if self.reference else []
        result.provenance.parameters = {
            "material": self.material_id,
            "reconstruction": self.reconstruction_id,
        }
        return result

    def as_dict(self) -> dict:
        return {
            "material": self.material_id,
            "reconstruction": self.reconstruction_id,
            "supported": False,
            "reason": self.reason,
            "reference": self.reference,
            "suggested": list(self.suggested),
            "result": self.as_result().as_dict(),
        }


@dataclass(frozen=True)
class GeneratorInfo:
    """Static description of what one generator can build."""

    id: str
    prototypes: Tuple[str, ...]
    orientation: Tuple[int, int, int]
    periodicity: Tuple[int, int]
    summary: str
    method: str
    references: Tuple[str, ...] = ()
    options: Dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "prototypes": list(self.prototypes),
            "orientation": list(self.orientation),
            "periodicity": list(self.periodicity),
            "summary": self.summary,
            "method": self.method,
            "references": list(self.references),
            "options": dict(self.options),
        }


Generator = Callable[[Structure, MaterialDefinition, dict], dict]

_GENERATORS: Dict[str, Tuple[GeneratorInfo, Generator]] = {}


def register(info: GeneratorInfo, generator: Generator) -> None:
    """Add a generator to the registry, keyed by reconstruction id."""
    _GENERATORS[info.id] = (info, generator)


def generator_info(reconstruction_id: str) -> Optional[GeneratorInfo]:
    entry = _GENERATORS.get(reconstruction_id)
    return entry[0] if entry else None


def registered_reconstructions() -> List[dict]:
    """Every reconstruction this build can actually generate."""
    return [info.as_dict() for info, _ in _GENERATORS.values()]


def _matches(info: GeneratorInfo, declaration: Reconstruction,
             material: MaterialDefinition) -> bool:
    if material.prototype not in info.prototypes:
        return False
    return tuple(int(v) for v in declaration.orientation) == info.orientation


def available_reconstructions(material: MaterialDefinition,
                               miller: Optional[Sequence[int]] = None) -> List[dict]:
    """Reconstructions declared for ``material``, each flagged supported or not.

    The flag is resolved against the generator registry and the material's own
    prototype, not against the ``implemented`` key in the file, so a definition
    cannot claim a capability the build does not have.
    """
    want = tuple(int(v) for v in miller) if miller is not None else None
    out: List[dict] = []
    for declaration in material.reconstructions:
        orientation = tuple(int(v) for v in declaration.orientation)
        if want is not None and orientation != want:
            continue
        entry = _GENERATORS.get(declaration.id)
        info = entry[0] if entry else None
        supported = bool(info and _matches(info, declaration, material)
                         and declaration.implemented)
        record = {
            "id": declaration.id,
            "orientation": list(orientation),
            "description": declaration.description,
            "reference": declaration.reference,
            "note": declaration.note,
            "declared_implemented": bool(declaration.implemented),
            "supported": supported,
            "generator": info.as_dict() if info else None,
            "reference_geometry": dict(declaration.reference_geometry),
        }
        if not supported:
            record["reason"] = _unsupported_reason(material, declaration, info)
        out.append(record)
    return out


def _unsupported_reason(material: MaterialDefinition, declaration: Reconstruction,
                        info: Optional[GeneratorInfo]) -> str:
    if info is None:
        return (
            f"{material.id} declares the {declaration.id} reconstruction but Materia "
            f"{__version__} ships no generator for it. The surface is built as the "
            "unreconstructed truncation and labelled as such; no approximate "
            "reconstructed geometry is produced."
        )
    if material.prototype not in info.prototypes:
        return (
            f"The {declaration.id} generator applies to the "
            f"{', '.join(info.prototypes)} prototype, and {material.id} is "
            f"{material.prototype}."
        )
    return (
        f"{material.id} declares {declaration.id} as not implemented; the generator "
        "was not enabled for this material."
    )


def find(material: MaterialDefinition, reconstruction_id: str) -> Reconstruction:
    """Look up a declaration, raising a listing error if the id is unknown."""
    for declaration in material.reconstructions:
        if declaration.id == reconstruction_id:
            return declaration
    known = ", ".join(r.id for r in material.reconstructions) or "none"
    raise ReconstructionError(
        f"{material.id} declares no reconstruction {reconstruction_id!r}. Declared: {known}."
    )


def _surface_layer(structure: Structure, tolerance_A: float) -> np.ndarray:
    z = structure.positions[:, 2]
    if len(z) == 0:
        raise ReconstructionError("Cannot reconstruct an empty slab")
    return np.where(z >= z.max() - tolerance_A)[0]


def _bond_directions(structure: Structure, index: int) -> List[np.ndarray]:
    origin = structure.positions[index]
    out = []
    for neighbour in structure.neighbors_of(int(structure.ids[index])):
        delta = structure.positions[structure.index_of(neighbour)] - origin
        delta = structure.cell.minimum_image(delta[None, :])[0]
        norm = float(np.linalg.norm(delta))
        if norm > 1e-6:
            out.append(delta / norm)
    return out


def dangling_bond_plane(structure: Structure, index: int) -> np.ndarray:
    """In-plane direction along which the two dangling bonds of a site point.

    For a tetrahedral atom holding two back-bonds ``b1`` and ``b2``, the two
    missing orbitals lie in the plane spanned by ``-(b1 + b2)`` and ``b1 x b2``.
    The cross product is therefore the in-plane axis that separates them, and it
    is the direction along which two such atoms can pair without twisting their
    remaining bonds.  Deriving it from the slab means the result follows the
    crystal orientation instead of an assumed cube axis.
    """
    directions = _bond_directions(structure, index)
    if len(directions) != 2:
        raise ReconstructionError(
            f"Site {int(structure.ids[index])} has {len(directions)} bonds; the dimer "
            "generator needs surface atoms with exactly two back-bonds, which is what "
            "an ideal diamond-structure (100) truncation provides. Check the "
            "termination and the slab thickness."
        )
    axis = np.cross(directions[0], directions[1])
    axis[2] = 0.0
    norm = float(np.linalg.norm(axis))
    if norm < 1e-6:
        raise ReconstructionError(
            "The two back-bonds of a surface site are coplanar with the surface "
            "normal, so no in-plane dimer axis is defined for this termination."
        )
    return axis / norm


def _rows(structure: Structure, indices: np.ndarray, axis: np.ndarray,
          tolerance_A: float = 0.35) -> List[List[int]]:
    """Group surface sites into rows perpendicular to ``axis``."""
    perpendicular = np.cross(np.array([0.0, 0.0, 1.0]), axis)
    coordinate = structure.positions[indices] @ perpendicular
    order = np.argsort(coordinate)
    rows: List[List[int]] = []
    current: List[int] = []
    previous = None
    for position in order:
        value = float(coordinate[position])
        if previous is not None and abs(value - previous) > tolerance_A:
            rows.append(current)
            current = []
        current.append(int(indices[position]))
        previous = value
    if current:
        rows.append(current)
    return rows


def _pair_along_axis(structure: Structure, row: Sequence[int],
                     axis: np.ndarray) -> List[Tuple[int, int]]:
    coordinate = {i: float(structure.positions[i] @ axis) for i in row}
    ordered = sorted(row, key=lambda i: coordinate[i])
    if len(ordered) % 2:
        raise ReconstructionError(
            f"A dimer row holds {len(ordered)} sites. A 2x1 reconstruction needs an "
            "even number of surface sites along the dimer axis, so the in-plane "
            "repeat along that direction must be even (2, 4, 6, ...)."
        )
    return [(ordered[k], ordered[k + 1]) for k in range(0, len(ordered), 2)]


def _dimer_2x1(slab: Structure, material: MaterialDefinition, options: dict) -> dict:
    """Pair the dangling bonds of a diamond-structure (100) truncation.

    Each site of the ideal (100) surface layer carries two dangling bonds lying
    in a plane whose in-plane trace alternates by 90 degrees from layer to
    layer.  Neighbouring sites along that trace can close one dangling bond each
    by moving together, which halves the periodicity in that direction and
    leaves rows of dimers: the 2x1 pattern first reported for Si(100) by Schlier
    and Farnsworth.  The generator establishes the pairing and moves each
    partner to a starting separation; the bond length itself comes from the
    relaxation that follows.
    """
    separation = float(options.get("initial_separation_A", 2.60))
    tolerance = float(options.get("layer_tolerance_A", 0.35))
    if separation <= 0.5:
        raise ReconstructionError("initial_separation_A must be a positive bond-scale distance")

    slab.ensure_bonds()
    surface = _surface_layer(slab, tolerance)
    if len(surface) < 2:
        raise ReconstructionError(
            f"The top layer holds {len(surface)} site(s); a dimer needs at least two."
        )
    axis = dangling_bond_plane(slab, int(surface[0]))
    for index in surface[1:]:
        other = dangling_bond_plane(slab, int(index))
        if abs(float(other @ axis)) < 0.99:
            raise ReconstructionError(
                "Surface sites do not share a common dangling-bond plane, so the "
                "top layer is not a single (100) termination. Rebuild the slab with "
                "termination_mode='auto-stable'."
            )

    rows = _rows(slab, surface, axis, tolerance_A=tolerance)
    pairs: List[Tuple[int, int]] = []
    for row in rows:
        pairs.extend(_pair_along_axis(slab, row, axis))

    positions = slab.positions.copy()
    initial = []
    for left, right in pairs:
        span = float((positions[right] - positions[left]) @ axis)
        if span <= 0:
            raise ReconstructionError("Dimer partners are not ordered along the dimer axis")
        initial.append(span)
        shift = 0.5 * (span - separation)
        positions[left] = positions[left] + shift * axis
        positions[right] = positions[right] - shift * axis
    slab.positions = positions
    slab.invalidate_bonds()

    for left, right in pairs:
        for index in (left, right):
            slab.roles[index] = "dimer"
            slab.labels[index] = f"{slab.symbols()[index]}-dimer"

    unreconstructed = float(np.mean(initial)) if initial else 0.0
    return {
        "pairs": [[int(slab.ids[a]), int(slab.ids[b])] for a, b in pairs],
        "n_dimers": len(pairs),
        "n_rows": len(rows),
        "dimer_axis": [round(float(v), 6) for v in axis],
        "row_axis": [round(float(v), 6) for v in np.cross([0.0, 0.0, 1.0], axis)],
        "initial_separation_A": separation,
        "unreconstructed_spacing_A": unreconstructed,
        "periodicity": [2, 1],
        "topology_only": True,
    }


register(
    GeneratorInfo(
        id="2x1-dimer",
        prototypes=("diamond-cubic",),
        orientation=(1, 0, 0),
        periodicity=(2, 1),
        summary=(
            "Rows of surface dimers formed by pairing the dangling bonds of the "
            "ideal (100) truncation; halves the periodicity along the dimer axis."
        ),
        method=(
            "Dimer partners are chosen from the slab's own symmetry: the in-plane "
            "dimer axis is the cross product of a surface site's two back-bonds, and "
            "sites are paired along it. Only the pairing and a starting separation "
            "are imposed; the geometry is the relaxed minimum of the interatomic "
            "model that follows."
        ),
        references=(
            "R. E. Schlier and H. E. Farnsworth, J. Chem. Phys. 30 (1959) 917",
        ),
        options={
            "initial_separation_A": "Starting partner separation before relaxation "
                                    "(default 2.60 A). The relaxed result is "
                                    "insensitive to it over a wide range.",
            "layer_tolerance_A": "Depth window used to identify the top atomic layer "
                                 "(default 0.35 A).",
        },
    ),
    _dimer_2x1,
)


def verify_periodicity(structure: Structure, axis: Optional[Sequence[float]] = None,
                       spacing_A: Optional[float] = None,
                       tolerance_A: float = 0.05) -> dict:
    """Test the slab against the 1x1 and 2x1 translations along the dimer axis.

    A genuine 2x1 must break the 1x1 translational symmetry it was cut with and
    keep the doubled one.  Both are checked by translating every atom and asking
    whether the translated set coincides with the original, element by element.
    When the cell is exactly two 1x1 repeats wide the doubled translation is the
    lattice vector itself, which is trivially satisfied; that is reported rather
    than counted as evidence.
    """
    record = structure.info.get("reconstruction") or {}
    if axis is None:
        axis = record.get("dimer_axis")
    if spacing_A is None:
        spacing_A = record.get("unreconstructed_spacing_A")
    if axis is None or not spacing_A:
        raise ReconstructionError(
            "No dimer axis or 1x1 spacing available; pass them explicitly."
        )
    axis = np.asarray(axis, dtype=float)
    axis = axis / float(np.linalg.norm(axis))
    length = float(np.linalg.norm(structure.cell.matrix[:2] @ axis))
    repeats = length / float(spacing_A)

    def invariant(multiple: int) -> bool:
        shifted = structure.positions + multiple * float(spacing_A) * axis
        return _sets_coincide(structure, shifted, tolerance_A)

    doubled_is_lattice = abs(repeats - 2.0) < 1e-6
    return {
        "dimer_axis": [float(v) for v in axis],
        "onexone_spacing_A": float(spacing_A),
        "cell_length_along_axis_A": length,
        "onexone_repeats_along_axis": float(repeats),
        "invariant_under_1x1_shift": invariant(1),
        "invariant_under_2x1_shift": invariant(2),
        "2x1_shift_is_lattice_vector": doubled_is_lattice,
        "tolerance_A": tolerance_A,
        "is_2x1": (not invariant(1)) and invariant(2),
    }


def _sets_coincide(structure: Structure, shifted: np.ndarray, tolerance_A: float) -> bool:
    reference = structure.positions
    numbers = structure.numbers
    for k in range(len(shifted)):
        delta = structure.cell.minimum_image(reference - shifted[k])
        distance = np.linalg.norm(delta, axis=1)
        distance[numbers != numbers[k]] = np.inf
        if float(distance.min()) > tolerance_A:
            return False
    return True


def dimer_statistics(structure: Structure) -> dict:
    """Measure the dimers recorded in ``structure.info['reconstruction']``.

    Returns bond lengths, the buckling (vertical offset within a dimer) and the
    row geometry, all measured from the current coordinates.  Callers use this
    to compare a relaxed slab against published structural data.
    """
    record = structure.info.get("reconstruction")
    if not record or not record.get("pairs"):
        raise ReconstructionError(
            "This structure carries no reconstruction record with dimer pairs."
        )
    lengths, buckling = [], []
    for left_id, right_id in record["pairs"]:
        a = structure.positions[structure.index_of(int(left_id))]
        b = structure.positions[structure.index_of(int(right_id))]
        delta = structure.cell.minimum_image((b - a)[None, :])[0]
        lengths.append(float(np.linalg.norm(delta)))
        buckling.append(abs(float(b[2] - a[2])))
    lengths_arr = np.asarray(lengths)
    buckling_arr = np.asarray(buckling)
    return {
        "n_dimers": len(lengths),
        "bond_length_A": float(lengths_arr.mean()),
        "bond_length_spread_A": float(lengths_arr.max() - lengths_arr.min()),
        "bond_lengths_A": [float(v) for v in lengths_arr],
        "buckling_A": float(buckling_arr.mean()),
        "buckling_max_A": float(buckling_arr.max()),
        "contraction_A": float(record.get("unreconstructed_spacing_A", 0.0)
                               - lengths_arr.mean()),
        "unreconstructed_spacing_A": float(record.get("unreconstructed_spacing_A", 0.0)),
    }


def compare_to_reference(structure: Structure, material: MaterialDefinition,
                         reconstruction_id: Optional[str] = None) -> dict:
    """Tabulate measured geometry against the published values in the material file.

    Agreement is never asserted: the comparison reports the measured value, the
    cited value with its stated uncertainty, the signed deviation, and whether
    the measurement falls inside the cited interval.

    A comparison is only meaningful against a converged geometry.  When the
    relaxation did not converge, or none was run, the measured numbers are still
    shown -- hiding them would be its own distortion -- but ``comparable`` is
    false, every ``within_stated_uncertainty`` is ``None`` rather than a
    boolean, and the reason is stated.  Nothing downstream can read a
    half-relaxed bond length as a model result that was checked against
    experiment.
    """
    record = structure.info.get("reconstruction") or {}
    reconstruction_id = reconstruction_id or record.get("id", "")
    declaration = find(material, reconstruction_id)
    reference = declaration.reference_geometry
    measured = dimer_statistics(structure)
    status = geometry_status_of(record)
    comparable = status == GEOMETRY_CONVERGED
    rows = []
    for key, entry in reference.items():
        if key not in measured:
            continue
        value = float(entry["value"])
        uncertainty = float(entry.get("uncertainty", 0.0) or 0.0)
        observed = float(measured[key])
        deviation = observed - value
        rows.append({
            "quantity": key,
            "measured": observed,
            "reference": value,
            "reference_uncertainty": uncertainty,
            "unit": entry.get("unit", "A"),
            "deviation": deviation,
            "relative_deviation": deviation / value if value else None,
            "comparable": comparable,
            "within_stated_uncertainty": (
                bool(uncertainty and abs(deviation) <= uncertainty) if comparable else None
            ),
            "source": entry.get("source", ""),
            "method": entry.get("method", ""),
        })
    relaxation = record.get("relaxation") or {}
    return {
        "material": material.id,
        "reconstruction": reconstruction_id,
        "model": relaxation.get("model") or "unrelaxed",
        "geometry_status": status,
        "comparable": comparable,
        "blocked_reason": "" if comparable else _not_comparable_reason(status, relaxation),
        "rows": rows,
        "note": (
            "Deviations are reported, not corrected. A classical interatomic "
            "potential is not expected to reproduce a reconstruction geometry to "
            "the precision of a diffraction measurement."
        ),
    }


def geometry_status_of(record: dict) -> str:
    """The geometry state of a reconstruction record.

    Records written before ``geometry_status`` existed stored only the
    relaxation's own ``performed`` and ``converged`` flags, so the state is
    derived from those when the key is absent.  A project saved by an earlier
    build therefore reads back with the state it actually had rather than
    defaulting to the most cautious one.
    """
    status = record.get("geometry_status")
    if status:
        return str(status)
    relaxation = record.get("relaxation") or {}
    if not relaxation.get("performed"):
        return GEOMETRY_UNRELAXED
    return GEOMETRY_CONVERGED if relaxation.get("converged") else GEOMETRY_NOT_CONVERGED


def _not_comparable_reason(status: str, relaxation: dict) -> str:
    if status == GEOMETRY_NOT_CONVERGED:
        residual = relaxation.get("max_force_eV_A")
        target = relaxation.get("fmax_target_eV_A")
        detail = ""
        if residual is not None and target is not None:
            detail = (f" It stopped after {relaxation.get('steps')} step(s) with a "
                      f"residual force of {residual:.4g} eV/A against a target of "
                      f"{target:g} eV/A.")
        return (
            "The relaxation did not converge, so this geometry is not an energy "
            "minimum of the model and the numbers below are not a prediction to "
            "compare against measurement." + detail
        )
    return (
        "No relaxation was run, so the separations below are the generator's "
        "construction guess rather than a predicted geometry."
    )


def apply_reconstruction(
    slab: Structure,
    material: MaterialDefinition,
    reconstruction: str,
    *,
    relax: bool = True,
    model: str = "recommended",
    fmax_eV_A: float = 0.005,
    max_steps: int = 800,
    options: Optional[dict] = None,
    callback: Optional[Callable[[int, float, float], bool]] = None,
) -> Structure:
    """Apply ``reconstruction`` to ``slab`` in place and return it.

    The default force tolerance is tighter than the one used for a general
    relaxation because the reported quantity is a bond length quoted to three
    decimals: at 0.02 eV/A the residual forces move it by a few thousandths of
    an angstrom depending on the starting separation, and at 0.005 eV/A that
    scatter falls below 5e-4 A.

    Raises :class:`ReconstructionNotImplemented` when the material declares the
    reconstruction but this build cannot generate it, and
    :class:`ReconstructionError` when the slab cannot carry it.
    """
    declaration = find(material, reconstruction)
    entry = _GENERATORS.get(reconstruction)
    info = entry[0] if entry else None
    if info is None or not _matches(info, declaration, material) or not declaration.implemented:
        raise ReconstructionNotImplemented(
            material.id, reconstruction,
            _unsupported_reason(material, declaration, info),
            reference=declaration.reference,
            suggested=["external:gpaw", "external:quantum-espresso"],
        )

    surface_info = slab.info.get("surface") or {}
    miller = tuple(int(v) for v in surface_info.get("miller", info.orientation))
    if miller != info.orientation:
        raise ReconstructionError(
            f"{reconstruction} applies to the ({''.join(str(v) for v in info.orientation)}) "
            f"surface, and this slab exposes ({''.join(str(v) for v in miller)})."
        )
    if slab.info.get("reconstruction"):
        raise ReconstructionError(
            "This slab already carries the "
            f"{slab.info['reconstruction']['id']} reconstruction."
        )
    relax, model, fmax_eV_A, max_steps = _validated_relaxation_settings(
        relax, model, fmax_eV_A, max_steps)

    restore_point = slab.copy()
    try:
        return _apply(slab, material, reconstruction, declaration, info, entry[1],
                      miller, surface_info, relax, model, fmax_eV_A, max_steps,
                      options, callback)
    except BaseException:
        slab.restore_from(restore_point)
        raise


def _validated_relaxation_settings(relax, model, fmax_eV_A, max_steps):
    """Check the relaxation controls before anything is mutated.

    A bad value must be refused while the slab is still untouched, rather than
    raising out of the middle of a minimisation and leaving atoms that have
    been moved but not recorded.
    """
    if not isinstance(model, str):
        raise ReconstructionError(
            f"model must be a solver name, got {type(model).__name__}."
        )
    try:
        fmax_eV_A = float(fmax_eV_A)
    except (TypeError, ValueError):
        raise ReconstructionError(
            f"fmax_eV_A must be a force in eV/A, got {fmax_eV_A!r}."
        ) from None
    if not np.isfinite(fmax_eV_A) or fmax_eV_A <= 0.0:
        raise ReconstructionError(
            f"fmax_eV_A must be a positive, finite force in eV/A, got {fmax_eV_A!r}."
        )
    if isinstance(max_steps, bool) or not isinstance(max_steps, (int, np.integer)):
        raise ReconstructionError(
            f"max_steps must be a positive whole number of steps, got {max_steps!r}."
        )
    max_steps = int(max_steps)
    if max_steps < 1:
        raise ReconstructionError(
            f"max_steps must be at least 1, got {max_steps}."
        )
    return bool(relax), model, fmax_eV_A, max_steps


def _apply(slab, material, reconstruction, declaration, info, generator,
           miller, surface_info, relax, model, fmax_eV_A, max_steps,
           options, callback):
    """Build the reconstruction and write its records.

    Called only from :func:`apply_reconstruction`, which holds the restore
    point that undoes everything here if any step raises.
    """
    report = generator(slab, material, dict(options or {}))
    report["id"] = reconstruction
    report["material"] = material.id
    report["miller"] = list(miller)
    report["generator"] = info.as_dict()
    report["n_fixed_atoms"] = int(slab.fixed.sum())

    relaxation: Dict[str, Any] = {"model": None, "performed": False}
    if relax:
        relaxation = _relax(slab, material, model, fmax_eV_A, max_steps, callback)
    report["relaxation"] = relaxation
    report["relaxation_attempted"] = bool(relaxation.get("performed"))
    report["relaxed"] = bool(relaxation.get("performed") and relaxation.get("converged"))
    report["geometry_status"] = (
        GEOMETRY_CONVERGED if report["relaxed"]
        else GEOMETRY_NOT_CONVERGED if report["relaxation_attempted"]
        else GEOMETRY_UNRELAXED
    )

    approximations = [
        "Reconstruction topology is imposed by the generator; only the pairing and "
        "an initial separation are prescribed.",
    ]
    if report["geometry_status"] == GEOMETRY_CONVERGED:
        origin = Origin.CALCULATED
        fidelity = Fidelity.TIER1_CLASSICAL
        approximations.append(
            f"Geometry is the {relaxation['model']} energy minimum of this slab, not a "
            "first-principles or measured structure."
        )
        if report["n_fixed_atoms"] == 0:
            approximations.append(
                "No atoms were held fixed, so both slab faces relaxed and the lower "
                "face is not a bulk-terminated reference."
            )
    elif report["geometry_status"] == GEOMETRY_NOT_CONVERGED:
        origin = Origin.ESTIMATED
        fidelity = Fidelity.TIER1_CLASSICAL
        residual = relaxation.get("max_force_eV_A")
        approximations.append(
            f"Relaxation with {relaxation['model']} did not converge: it stopped after "
            f"{relaxation.get('steps')} step(s) with a residual force of "
            f"{residual:.4g} eV/A against a target of {fmax_eV_A:g} eV/A."
            if residual is not None else
            f"Relaxation with {relaxation['model']} did not converge."
        )
        approximations.append(
            "The geometry is therefore NOT an energy minimum of the model and no "
            "distance measured from it is a prediction. Re-run with a larger step "
            "budget or a looser force target before using or comparing it."
        )
    else:
        origin = Origin.ESTIMATED
        fidelity = Fidelity.TIER0_STRUCTURAL
        approximations.append(
            "Not relaxed: partner separation is the construction guess, not a "
            "predicted bond length."
        )
        if relaxation.get("reason"):
            approximations.append(str(relaxation["reason"]))

    provenance = Provenance(
        model=f"structure-builder/reconstruction:{reconstruction}",
        fidelity=fidelity,
        origin=origin,
        approximations=approximations,
        boundary_conditions="2D periodic in x,y; vacuum-terminated in z",
        parameters={
            "material": material.id,
            "reconstruction": reconstruction,
            "miller": list(miller),
            "periodicity": list(info.periodicity),
            "n_dimers": report.get("n_dimers"),
            "initial_separation_A": report.get("initial_separation_A"),
            "relaxation_model": relaxation.get("model"),
            "fmax_eV_A": fmax_eV_A if relax else None,
            "max_steps": max_steps if relax else None,
            "geometry_status": report["geometry_status"],
        },
        references=list(info.references) + ([declaration.reference] if declaration.reference else []),
        dataset=material.source_path,
        dataset_license=material.license,
        notes=info.method,
    )
    report["provenance"] = provenance.as_dict()
    slab.info["reconstruction"] = report

    surface_info["reconstruction"] = reconstruction
    surface_info["periodicity"] = list(info.periodicity)
    slab.info["surface"] = surface_info
    _update_surface_provenance(slab, reconstruction, report["geometry_status"])

    if report["relaxation_attempted"]:
        measured = dimer_statistics(slab)
        measured["geometry_status"] = report["geometry_status"]
        measured["is_energy_minimum"] = report["relaxed"]
        slab.info["reconstruction"]["measured"] = measured
    slab.info["reconstruction"]["periodicity_check"] = verify_periodicity(slab)
    return slab


def _update_surface_provenance(slab: Structure, reconstruction: str, status: str) -> None:
    """Replace the ideal-truncation statements the slab builder recorded."""
    provenance = slab.info.get("provenance")
    if not isinstance(provenance, dict):
        return
    kept = [
        text for text in provenance.get("approximations", [])
        if "no surface reconstruction" not in text
    ]
    tail = {
        GEOMETRY_CONVERGED: " and relaxed to a converged geometry.",
        GEOMETRY_NOT_CONVERGED: (
            " and a relaxation was attempted but did not converge, so the geometry "
            "is not an energy minimum."
        ),
        GEOMETRY_UNRELAXED: " without relaxation.",
    }[status]
    kept.append(
        f"The {reconstruction} reconstruction was applied to the upper face" + tail
    )
    provenance["approximations"] = kept
    notes = provenance.get("notes", "")
    provenance["notes"] = (notes + " " if notes else "") + (
        f"Upper face carries the {reconstruction} reconstruction; see "
        "info['reconstruction'] for its own provenance record."
    )


def _relax(slab: Structure, material: MaterialDefinition, model: str,
           fmax_eV_A: float, max_steps: int,
           callback: Optional[Callable[[int, float, float], bool]]) -> dict:
    solver, name, reason = _relaxation_solver(slab, material, model)
    if solver is None:
        return {"model": name, "performed": False, "reason": reason}
    result = solver.relax(slab, fmax_eV_A=fmax_eV_A, max_steps=max_steps,
                          in_place=True, callback=callback)
    relaxed = result.results.get("relaxed_structure")
    if relaxed is not None and relaxed.unsupported_reason:
        return {"model": name, "performed": False, "reason": relaxed.unsupported_reason}
    energy = result.results.get("energy")
    extra = dict(relaxed.extra) if relaxed is not None else {}
    convergence = relaxed.convergence if relaxed is not None else None
    return {
        "model": solver.name,
        "performed": True,
        "converged": bool(convergence.converged) if convergence else False,
        "steps": int(convergence.iterations) if convergence else None,
        "max_force_eV_A": float(convergence.residual) if convergence and convergence.residual is not None else None,
        "convergence_message": convergence.message if convergence else "",
        "max_displacement_A": extra.get("max_displacement_A"),
        "energy_eV": float(energy.value) if energy is not None and energy.value is not None else None,
        "fmax_target_eV_A": fmax_eV_A,
        "wall_time_s": float(result.wall_time_s),
        "provenance": relaxed.provenance.as_dict() if relaxed is not None else None,
    }


def _relaxation_solver(slab: Structure, material: MaterialDefinition, model: str):
    """Pick the interatomic model used to find the reconstructed geometry."""
    from ..solvers import registry
    from ..solvers.classical import stillinger_weber
    from ..physics.potentials import SW_PARAMETERS
    from ..elements import periodic_table as pt

    if model not in ("recommended", "auto"):
        if model not in registry.available():
            return None, model, (
                f"Unknown relaxation model {model!r}. Registered: "
                f"{', '.join(registry.available())}."
            )
        return registry.create(model), model, ""

    name = registry.resolve_recommended(material, "relax")
    if name is None:
        return None, material.recommended_models.get("relax"), (
            f"{material.id} recommends no registered relaxation model, so the "
            "reconstruction topology is returned unrelaxed."
        )
    if name.startswith("stillinger-weber"):
        elements = sorted({pt.symbol(int(z)) for z in slab.numbers})
        hosts = [e for e in elements if e in SW_PARAMETERS]
        if len(elements) == 1 and hosts:
            return stillinger_weber(hosts[0]), name, ""
        if not hosts:
            return None, name, (
                f"Stillinger-Weber has no parameters for {', '.join(elements)}."
            )
    return registry.create(name), name, ""
