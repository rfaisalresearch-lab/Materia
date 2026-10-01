"""Application service: the single place the interface talks to.

The HTTP layer in :mod:`materia.desktop_ui.server` is a thin translation of
these methods.  Keeping them here means the graphical interface, the Python
API and the tests all drive the same code, and that no simulation logic lives
in a view.
"""

from __future__ import annotations

import base64
import io
import math
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core_model.selection import (
    Selection,
    by_element,
    by_role,
    in_box,
    near_plane,
    neighbors_within,
    within_radius,
)
from ..core_model.structure import Structure
from ..dataio import formats as dataio
from ..elements import periodic_table as pt
from ..elements.configuration import (
    configuration,
    configuration_confidence,
    configuration_string,
    shell_occupancy,
    spin_orbitals,
    term_spin_multiplicity,
    unpaired_electrons,
    valence_electrons,
)
from ..materials.loader import default_library
from ..microscopy.afm import AFMSettings, AFMSimulator
from ..microscopy.noise import NoiseModel
from ..microscopy.stm import STMSettings, STMSimulator
from ..microscopy.tip import TIP_WORK_FUNCTIONS, Tip
from ..multiscale.region import RegionSpec, scale_ladder, format_span
from ..multiscale.wafer import SCALE_LEVELS, STANDARD_WAFERS, WaferSpec
from ..physics.bonds import attach_bonds
from ..physics.neighbors import CoincidentAtoms
from ..physics.potentials import PotentialError
from ..plugin_system import default_manager
from ..project_format.project import Project
from ..project_format.recovery import RecoveryStore
from ..provenance import Result
from ..python_api.api import EAM_SETTINGS, ApiError, Lab, UnsupportedRequest
from ..python_api.execution import ScriptRunner
from ..solvers import registry as solver_registry
from ..structure_builder.reconstruction import (
    ReconstructionError,
    ReconstructionNotImplemented,
    apply_reconstruction,
    available_reconstructions,
    compare_to_reference,
)
from ..solvers.external import availability_report
from ..version import __version__
from ..visualization.palette import apply_palette, list_palettes
from .jobs import Cancelled, JobQueue

MAX_RENDER_ATOMS = 250_000


class Service:
    """Everything the interface can ask for."""

    def __init__(self, project: Optional[Project] = None,
                 recovery_dir: Optional[str] = None) -> None:
        self.library = default_library()
        self.lab = Lab(project or Project("Untitled project", self.library), self.library)
        self.runner = ScriptRunner(self.lab, mode="restricted")
        self.jobs = JobQueue()
        self.plugins = default_manager()
        self.plugins.load_all()
        self.last_scan_key: Optional[str] = None
        self.warnings: List[dict] = []
        self.trusted_confirmed = False
        self.recovery = RecoveryStore(recovery_dir)
        self.recovery_candidate = None if project is not None else self.recovery.info()
        if self.recovery_candidate and self.recovery_candidate.get("corrupt"):
            self.warn(
                f"The autosave could not be read: {self.recovery_candidate['error']}",
                "error", "recovery")
        self._install_exit_hook()

    @property
    def project(self) -> Project:
        return self.lab.project

    @property
    def structure(self) -> Optional[Structure]:
        return self.project.structure

    def _require_structure(self) -> Structure:
        s = self.structure
        if s is None:
            raise ValueError("No active structure. Create a wafer and extract a region "
                             "first, or import a structure file.")
        return s

    def warn(self, text: str, level: str = "warning", source: str = "") -> None:
        self.warnings.append({"text": text, "level": level, "source": source,
                              "time": time.time()})
        del self.warnings[:-200]

    def _autosave(self) -> None:
        try:
            self.recovery.save(self.project)
        except Exception as exc:
            self.warn(f"Autosave failed: {exc}", "error", "recovery")

    def state(self) -> dict:
        s = self.structure
        wafer = self.project.wafer
        return {
            "version": __version__,
            "project": {
                "name": self.project.name,
                "modified": self.project.modified,
                "structures": sorted(self.project.structures),
                "active_structure": self.project.active_structure_key,
                "regions": {k: {"spec": v.spec.as_dict(), "n_atoms": len(v.structure),
                                "wafer_context": v.wafer_context,
                                "applied": _jsonable(v.applied)}
                            for k, v in self.project.regions.items()},
                "active_region": self.project.active_region_id,
                "checkpoints": sorted(self.project.checkpoints),
                "scripts": sorted(self.project.scripts),
                "results": sorted(self.project.results),
                "scans": sorted(self.project.scans),
                "can_undo": self.project.history.can_undo,
                "can_redo": self.project.history.can_redo,
                "undo_label": (self.project.history.undo_labels()[-1]
                               if self.project.history.can_undo else ""),
                "redo_label": (self.project.history.redo_labels()[-1]
                               if self.project.history.can_redo else ""),
                "history": [e.as_dict() for e in self.project.history.log[-200:]],
            },
            "wafer": None if wafer is None else {
                "spec": wafer.spec.as_dict(),
                "material": wafer.material.summary(),
                "radius_mm": wafer.radius_mm,
                "outline": wafer.outline(180).tolist(),
                "step_spacing_nm": (wafer._step_spacing_nm()
                                    if wafer.spec.miscut_deg > 0 else None),
                "provenance": wafer.provenance().as_dict(),
            },
            "structure": None if s is None else {
                "n_atoms": len(s),
                "formula": s.formula(),
                "cell": s.cell.as_dict(),
                "cell_lengths": s.cell.lengths.tolist(),
                "cell_angles": s.cell.angles_deg.tolist(),
                "total_charge": s.total_charge(),
                "total_electrons": s.total_electrons(),
                "total_moment": s.total_magnetic_moment(),
                "elements": sorted({pt.symbol(int(z)) for z in s.numbers}),
                "surface": _jsonable(s.info.get("surface", {})),
                "reconstruction": _jsonable(s.info.get("reconstruction")),
                "info_keys": sorted(s.info),
                "bounds": [s.bounding_box()[0].tolist(), s.bounding_box()[1].tolist()],
                "defects": _jsonable(s.info.get("defects", [])),
            },
            "selection": self.project.selection.as_dict(),
            "scan": self.last_scan_key,
            "scales": [{"level": n, "span_A": sp, "unit": u, "description": d}
                       for n, sp, u, d in SCALE_LEVELS],
            "warnings": self.warnings[-50:],
            "jobs": self.jobs.list(12),
            "script_mode": self.runner.mode,
            "recovery": self.recovery_candidate or {"available": False},
        }

    def materials(self) -> List[dict]:
        out = [d.summary() for d in self.library.all()]
        for d, summary in zip(self.library.all(), out):
            summary["source"] = d.source_path
            summary["aliases"] = list(d.aliases)
        return out

    def material_detail(self, key: str) -> dict:
        d = self.library.get(key)
        return {
            "summary": d.summary(),
            "properties": {k: v.as_dict() for k, v in d.properties.items()},
            "basis": [b.as_dict() for b in d.basis],
            "lattice": d.lattice.as_dict(),
            "terminations": [t.as_dict() for t in d.terminations],
            "reconstructions": available_reconstructions(d),
            "references": d.references,
            "license": d.license,
            "provenance_note": d.provenance_note,
            "source": d.source_path,
            "recommended_models": d.recommended_models,
        }

    def material_errors(self) -> Dict[str, str]:
        return self.library.errors()

    def create_wafer(self, **kwargs) -> dict:
        spec = WaferSpec(**kwargs)
        self.project.create_wafer(spec)
        self._autosave()
        return self.state()

    def wafer_probe(self, x_mm: float, y_mm: float) -> dict:
        if self.project.wafer is None:
            raise ValueError("No wafer defined.")
        return _jsonable(self.project.wafer.surface_at(x_mm, y_mm))

    def wafer_map(self, field: str = "roughness", n: int = 96) -> dict:
        """Coarse sampled map of a procedural wafer field, for the wafer view."""
        w = self.project.wafer
        if w is None:
            raise ValueError("No wafer defined.")
        r = w.radius_mm
        xs = np.linspace(-r, r, n)
        ys = np.linspace(-r, r, n)
        data = np.full((n, n), np.nan)
        for j, y in enumerate(ys):
            for i, x in enumerate(xs):
                if not w.contains(float(x), float(y)):
                    continue
                if field == "roughness":
                    data[j, i] = w._value_noise(float(x), float(y), "roughness",
                                                w.spec.roughness_correlation_nm)
                elif field == "grain":
                    g = w.grain_at(float(x), float(y))
                    data[j, i] = 0.0 if g is None else float(g["grain_id"] % 32)
                else:
                    raise ValueError(f"Unknown wafer field {field!r}")
        return {"x_mm": xs.tolist(), "y_mm": ys.tolist(),
                "data": [[None if np.isnan(v) else float(v) for v in row] for row in data],
                "field": field, "radius_mm": r,
                "note": "Procedural, seeded microstructure; synthetic, not measured."}

    def extract_region(self, **kwargs) -> dict:
        region = self.project.extract_region(RegionSpec(**kwargs))
        attach_bonds(region.structure)
        self._autosave()
        return {"region_id": region.region_id, "n_atoms": len(region.structure),
                "applied": _jsonable(region.applied), "state": self.state()}

    def build_surface(self, material: str, miller: Sequence[int],
                      size: Sequence[int] = (4, 4, 4), vacuum_A: float = 14.0,
                      reconstruction: Optional[str] = None, **kwargs) -> dict:
        definition = self.library.get(material)
        if reconstruction:
            kwargs["reconstruction"] = reconstruction
        try:
            handle = self.lab.materials.load(material).create_surface(
                size=size, vacuum_angstrom=vacuum_A, orientation=miller, **kwargs)
        except (ReconstructionNotImplemented, UnsupportedRequest) as exc:
            record = exc.as_dict()
            self.warn(record["reason"], level="unsupported", source="structure/build")
            return {"ok": False, "unsupported": record, "state": self.state()}
        except ReconstructionError as exc:
            self.warn(str(exc), level="error", source="structure/build")
            return {"ok": False, "error": str(exc),
                    "reconstruction": reconstruction, "state": self.state()}
        attach_bonds(handle.structure)
        key = self.project.add_structure(handle.structure)
        self._autosave()
        out = {"ok": True, "key": key, "n_atoms": len(handle.structure)}
        record = handle.structure.info.get("reconstruction")
        if record:
            out["reconstruction"] = _jsonable(record)
            out["comparison"] = _jsonable(
                compare_to_reference(handle.structure, definition))
        out["state"] = self.state()
        return out

    def reconstruct(self, reconstruction: str, relax: bool = True,
                    model: str = "recommended", **kwargs) -> dict:
        """Apply a reconstruction to the active structure.

        The reconstruction is built on a copy and installed only once it has
        succeeded, so a refusal leaves both the structure and the project
        history exactly as they were.  A success records one undo point of its
        own rather than relabelling whatever came before it.

        Returns an explicit unsupported record for a declared reconstruction
        this build cannot generate, and an error record when the slab cannot
        carry it.  Neither case produces an approximate geometry.
        """
        structure = self._require_structure()
        definition = self._active_material()
        if definition is None:
            return {"ok": False, "error": (
                "The active structure is not associated with a material "
                "definition, so its declared reconstructions are unknown.")}
        working = structure.copy()
        try:
            apply_reconstruction(working, definition, reconstruction,
                                 relax=relax, model=model, **kwargs)
        except ReconstructionNotImplemented as exc:
            self.warn(exc.reason, level="unsupported", source="structure/reconstruct")
            return {"ok": False, "unsupported": exc.as_dict(), "state": self.state()}
        except ReconstructionError as exc:
            self.warn(str(exc), level="error", source="structure/reconstruct")
            return {"ok": False, "error": str(exc), "state": self.state()}
        attach_bonds(working)
        record = working.info["reconstruction"]
        self.project.set_structure(
            working, f"Apply {reconstruction} reconstruction", "structure.reconstruct",
            {"reconstruction": reconstruction, "relax": bool(relax), "model": model},
            result_summary=(
                f"{record['n_dimers']} dimers, geometry "
                f"{record['geometry_status']}" if record.get("n_dimers") is not None
                else record["geometry_status"]),
        )
        self._autosave()
        return {"ok": True, "reconstruction": _jsonable(record),
                "comparison": _jsonable(compare_to_reference(working, definition)),
                "state": self.state()}

    def reconstruction_report(self) -> dict:
        """The active structure's reconstruction record and its comparison to
        published geometry, or a statement that the surface is unreconstructed."""
        structure = self._require_structure()
        record = structure.info.get("reconstruction")
        if not record:
            declared = []
            definition = self._active_material()
            if definition is not None:
                miller = (structure.info.get("surface") or {}).get("miller")
                declared = available_reconstructions(definition, miller)
            return {"reconstructed": False, "declared": _jsonable(declared),
                    "note": ("This surface is the ideal truncation. No "
                             "reconstruction has been applied.")}
        definition = self._active_material()
        out = {"reconstructed": True, "record": _jsonable(record)}
        if definition is not None:
            out["comparison"] = _jsonable(compare_to_reference(structure, definition))
        return out

    def render_payload(self, max_atoms: int = MAX_RENDER_ATOMS,
                       include_bonds: bool = True) -> dict:
        """Compact arrays for the 3-D viewport."""
        s = self._require_structure()
        n = len(s)
        stride = 1
        if n > max_atoms:
            stride = int(np.ceil(n / max_atoms))
        idx = np.arange(0, n, stride)
        radii = np.array([pt.covalent_radius(int(z)) or 1.2 for z in s.numbers[idx]])
        if include_bonds and s.bonds is None and n <= 60000:
            attach_bonds(s)
        bonds: List[int] = []
        if include_bonds and s.bonds is not None and stride == 1:
            index = {int(i): k for k, i in enumerate(s.ids)}
            for b in s.bonds:
                if b.a in index and b.b in index:
                    bonds.extend([index[b.a], index[b.b]])
        sel = set(self.project.selection.ids)
        return {
            "n_total": n,
            "stride": stride,
            "ids": s.ids[idx].tolist(),
            "numbers": s.numbers[idx].tolist(),
            "positions": s.positions[idx].astype(np.float32).ravel().tolist(),
            "radii": radii.astype(np.float32).tolist(),
            "roles": [str(r) for r in s.roles[idx]],
            "selected": [1 if int(i) in sel else 0 for i in s.ids[idx]],
            "bonds": bonds,
            "cell": s.cell.matrix.tolist(),
            "pbc": list(s.cell.pbc),
            "bounds": [s.bounding_box()[0].tolist(), s.bounding_box()[1].tolist()],
            "elements": {pt.symbol(int(z)): int(z) for z in np.unique(s.numbers)},
            "truncated": stride > 1,
            "note": ("Every atom of the region is drawn."
                     if stride == 1 else
                     f"Level of detail active: showing every {stride}th atom "
                     f"({len(idx)} of {n}). Zoom in for the full set."),
        }

    def select(self, mode: str, **kwargs) -> dict:
        s = self._require_structure()
        if mode == "ids":
            sel = Selection([int(i) for i in kwargs.get("ids", [])], "explicit")
        elif mode == "element":
            sel = by_element(s, *kwargs["elements"])
        elif mode == "role":
            sel = by_role(s, *kwargs["roles"])
        elif mode == "radius":
            sel = within_radius(s, kwargs["center"], float(kwargs["radius_A"]))
        elif mode == "neighbors":
            sel = neighbors_within(s, int(kwargs["atom_id"]), float(kwargs.get("radius_A", 3.0)))
        elif mode == "box":
            sel = in_box(s, kwargs["lo"], kwargs["hi"])
        elif mode == "plane":
            sel = near_plane(s, kwargs["miller"], float(kwargs["offset_A"]),
                             float(kwargs.get("tolerance_A", 0.5)))
        elif mode == "all":
            sel = Selection([int(i) for i in s.ids], "all atoms")
        elif mode == "none":
            sel = Selection([], "")
        elif mode == "invert":
            sel = self.project.selection.invert(s)
        else:
            raise ValueError(f"Unknown selection mode {mode!r}")
        if kwargs.get("add"):
            sel = self.project.selection.union(sel)
        self.project.selection = sel
        self._autosave()
        return {"selection": sel.as_dict(), "n": len(sel)}

    def inspect_atom(self, atom_id: int) -> dict:
        s = self._require_structure()
        atom = s.atom(int(atom_id))
        el = atom.element
        charge = int(round(atom.charge))
        shells = configuration(el.number, charge)
        origin, confidence, note = configuration_confidence(el.number)
        iso = el.most_abundant_isotope
        mass_number = atom.mass_number or (iso.mass_number if iso else None)
        isotope = None
        if mass_number:
            try:
                nuc = el.isotope(mass_number)
                isotope = {
                    "mass_number": nuc.mass_number,
                    "protons": el.number,
                    "neutrons": nuc.mass_number - el.number,
                    "atomic_mass_u": nuc.atomic_mass_u,
                    "natural_abundance": nuc.natural_abundance,
                    "spin": nuc.spin,
                    "half_life_s": nuc.half_life_s,
                    "stable": nuc.is_stable,
                }
            except KeyError:
                isotope = None
        if s.bonds is None:
            attach_bonds(s)
        bonds = []
        for b in s.bonds_of(atom.id):
            other = b.b if b.a == atom.id else b.a
            bonds.append({
                "partner_id": other,
                "partner_element": pt.symbol(int(s.numbers[s.index_of(other)])),
                "length_A": b.length_A or s.distance(atom.id, other),
                "order": b.order,
                "origin": b.origin,
            })
        neighbours = sorted(bonds, key=lambda b: b["length_A"])
        angles = []
        for i in range(len(neighbours)):
            for j in range(i + 1, len(neighbours)):
                try:
                    angles.append({
                        "a": neighbours[i]["partner_id"], "c": neighbours[j]["partner_id"],
                        "angle_deg": s.angle(neighbours[i]["partner_id"], atom.id,
                                             neighbours[j]["partner_id"]),
                    })
                except Exception:
                    pass
        force = atom.force
        return {
            "identity": {
                "id": atom.id,
                "element": el.symbol,
                "name": el.name,
                "atomic_number": el.number,
                "category": el.category,
                "group": el.group or None,
                "period": el.period,
                "block": el.block,
                "mass_u": atom.mass,
                "standard_atomic_weight": el.standard_atomic_weight,
                "covalent_radius_A": el.covalent_radius_A,
                "vdw_radius_A": el.vdw_radius_A,
                "electronegativity_pauling": el.electronegativity_pauling,
                "ionization_energy_eV": el.ionization_energy_eV,
                "electron_affinity_eV": el.electron_affinity_eV,
                "role": atom.role,
                "label": atom.label or None,
                "fixed": atom.fixed,
                "position_A": atom.position.tolist(),
                "velocity_A_fs": atom.velocity.tolist(),
                "data_origin": "reference",
            },
            "nucleus": {
                "isotope": isotope,
                "assigned": bool(atom.mass_number),
                "available_isotopes": [
                    {"mass_number": i.mass_number, "abundance": i.natural_abundance,
                     "mass_u": i.atomic_mass_u, "spin": i.spin, "stable": i.is_stable}
                    for i in el.isotopes],
                "note": ("Isotope assigned explicitly." if atom.mass_number else
                         "No isotope assigned: the most abundant natural nuclide is "
                         "shown. A scanning probe does not resolve isotopes."),
            },
            "electrons": {
                "charge_e": atom.charge,
                "partial_charge_e": atom.partial_charge,
                "configuration": configuration_string(el.number, charge),
                "configuration_short": configuration_string(el.number, charge, True),
                "subshells": [{"n": sh.n, "l": sh.l, "label": sh.label,
                               "electrons": sh.electrons, "capacity": sh.capacity,
                               "orbitals": sh.orbital_count,
                               "m_l": list(sh.m_l_values)} for sh in shells],
                "shell_occupancy": shell_occupancy(el.number, charge),
                "valence_electrons": valence_electrons(el.number, charge),
                "unpaired_electrons": unpaired_electrons(el.number, charge),
                "spin_multiplicity": term_spin_multiplicity(el.number, charge),
                "magnetic_moment_muB": atom.magnetic_moment,
                "spin_orbitals": [{"n": o.n, "l": o.l, "m_l": o.m_l, "m_s": o.m_s,
                                   "occupied": o.occupied} for o in
                                  spin_orbitals(el.number, charge)],
                "origin": origin,
                "confidence": confidence,
                "note": note,
                "caveat": ("Occupancies follow the Aufbau principle, the Pauli "
                           "exclusion principle and Hund's rules. The assignment of "
                           "electrons to particular m_l values is a convention: the "
                           "true many-electron state is generally a superposition."),
            },
            "environment": {
                "coordination": len(bonds),
                "bonds": bonds,
                "angles": angles,
                "nearest_neighbour_A": (min(b["length_A"] for b in bonds) if bonds else None),
                "bond_origin": "distance heuristic unless a solver supplied an order",
            },
            "energy": {
                "force_eV_A": [None if np.isnan(v) else float(v) for v in force],
                "force_magnitude_eV_A": (None if np.isnan(force).any()
                                         else float(np.linalg.norm(force))),
                "note": ("Forces are present only after a solver has run. "
                         "A per-atom potential energy is not defined for a "
                         "many-body potential and is therefore not reported."),
            },
            "electrostatics": self._electrostatics_for_atom(s, atom.id),
            "meta": _jsonable(dict(s.atom_meta.get(atom.id, {}))),
        }

    def element_reference(self, key: str) -> dict:
        el = pt.element(key)
        return {
            "symbol": el.symbol, "name": el.name, "number": el.number,
            "category": el.category, "group": el.group or None, "period": el.period,
            "block": el.block,
            "standard_atomic_weight": el.standard_atomic_weight,
            "covalent_radius_A": el.covalent_radius_A,
            "vdw_radius_A": el.vdw_radius_A,
            "electronegativity_pauling": el.electronegativity_pauling,
            "ionization_energy_eV": el.ionization_energy_eV,
            "electron_affinity_eV": el.electron_affinity_eV,
            "configuration": configuration_string(el.number),
            "configuration_short": configuration_string(el.number, abbreviated=True),
            "shell_occupancy": shell_occupancy(el.number),
            "isotopes": [{"mass_number": i.mass_number, "mass_u": i.atomic_mass_u,
                          "abundance": i.natural_abundance, "spin": i.spin,
                          "stable": i.is_stable} for i in el.isotopes],
        }

    def periodic_table(self) -> List[dict]:
        from ..elements.data import ELEMENTS
        return [{"symbol": e.symbol, "number": e.number, "name": e.name,
                 "group": e.group, "period": e.period, "category": e.category,
                 "block": e.block, "mass": e.standard_atomic_weight}
                for e in ELEMENTS]


@dataclass(frozen=True)
class GPAWSubmission:
    """One calculation, frozen at the moment it was asked for.

    A first-principles run takes minutes, and the project can change underneath
    it.  Everything the run depends on is copied here when it is submitted:
    which project owns the answer, which slot the structure came from, the
    exact worker specification, the validated configuration, and a fingerprint
    of both.  The run writes only to ``project``, and its forces are attached
    only if the structure still matches ``digest``.
    """

    run_id: str
    project: Any
    structure_key: Optional[str]
    spec: Dict[str, Any]
    settings: Dict[str, Any]
    digest: str
    apply_forces: bool = False
    timeout_s: Optional[float] = None


@dataclass(frozen=True)
class DFTSubmission:
    """One ground-state experiment, frozen when it was asked for.

    The specification already holds the geometry, every variable and the
    datasets it is pinned to, so nothing the run needs is read from live
    state afterwards.  The run writes only to ``project``.
    """

    run_id: str
    project: Any
    structure_key: Optional[str]
    spec: Any
    reuse_restart: bool = False
    keep_restart: bool = False
    timeout_s: Optional[float] = None


def _jsonable(obj):
    from ..provenance.record import _jsonable as conv
    return conv(obj)


class _InstrumentMixin:

    def scan(self, technique: str = "stm", settings: Optional[dict] = None,
             background: bool = True) -> dict:
        s = self._require_structure()
        settings = dict(settings or {})
        label = f"{technique.upper()} scan"

        def work(job):
            result = self._run_scan(technique, s, settings, job)
            self._autosave()
            return result

        if not background:
            result = self._run_scan(technique, s, settings, _NullJob())
            self._autosave()
            return result
        job = self.jobs.submit("scan", label, work)
        return {"job": job.as_dict()}

    def _run_scan(self, technique: str, structure: Structure, settings: dict, job):
        noise_name = settings.pop("noise", "realistic")
        seed = int(settings.pop("seed", 0))
        noise = {"quiet": NoiseModel.quiet, "realistic": NoiseModel.realistic,
                 "noisy": NoiseModel.noisy}[noise_name](seed)
        tip_material = settings.pop("tip", "W")
        tip = Tip(tip_material,
                  apex_state=settings.pop("apex_state", "s"),
                  radius_A=float(settings.pop("tip_radius_A", 50.0)))
        if technique == "stm":
            from ..microscopy.scan import ScanResult
            from ..python_api.api import ApiError

            material = self._active_material()
            try:
                solver = self.lab.microscope._tb_for(structure, material)
            except ApiError as exc:
                return self._finish_scan(ScanResult.unsupported_result(
                    "microscopy/stm-tersoff-hamann", str(exc),
                    {"technique": "stm", **settings},
                    suggested=["external:gpaw (Tersoff-Hamann from density functional theory)",
                               "external:quantum-espresso with Wannier90"]))
            wf = 4.85
            if material is not None and material.property("work_function"):
                wf = float(material.property("work_function").value)
            cfg = STMSettings(noise=noise, **_filter_kwargs(STMSettings, settings))
            sim = STMSimulator(solver, tip, sample_work_function_eV=wf)
            job.message = f"Tersoff-Hamann STM, {solver.name}"
            scan = sim.scan(structure, cfg)
        elif technique == "afm":
            cfg = AFMSettings(noise=noise, **_filter_kwargs(AFMSettings, settings))
            job.message = "Classical-force AFM"
            scan = AFMSimulator(tip).scan(structure, cfg)
        else:
            raise ValueError(f"Unknown technique {technique!r}; use 'stm' or 'afm'.")
        if job.cancel_flag.is_set():
            raise Cancelled()
        return self._finish_scan(scan)

    def _finish_scan(self, scan) -> dict:
        if not scan.supported:
            self.warn(scan.unsupported_reason, "error", scan.provenance.model)
            return {"supported": False, "reason": scan.unsupported_reason,
                    "suggested_models": scan.suggested_models,
                    "provenance": scan.provenance.as_dict()}
        key = self.project.add_scan(scan)
        self.last_scan_key = key
        return {"supported": True, "key": key, **self.scan_payload(key)}

    def _active_material(self):
        s = self.structure
        if s is None:
            return None
        mid = s.info.get("material_id")
        if mid is None and self.project.wafer is not None:
            mid = self.project.wafer.spec.material_id
        try:
            return self.library.get(mid) if mid else None
        except Exception:
            return None

    def scan_payload(self, key: Optional[str] = None,
                     channel: Optional[str] = None) -> dict:
        key = key or self.last_scan_key
        if key is None or key not in self.project.scans:
            raise ValueError("No scan available. Run a scan first.")
        scan = self.project.scans[key]
        name = channel or scan.primary_channel
        data = np.asarray(scan.channel(name), dtype=np.float32)
        return {
            "key": key,
            "technique": scan.technique,
            "mode": scan.mode,
            "channel": name,
            "channels": scan.channel_names(),
            "units": scan.units,
            "extent_A": list(scan.extent_A),
            "shape": [int(data.shape[0]), int(data.shape[1])],
            "pixel_size_A": list(scan.pixel_size_A()),
            "statistics": scan.statistics(name),
            "data_b64": base64.b64encode(data.tobytes()).decode("ascii"),
            "dtype": "float32",
            "provenance": scan.provenance.as_dict(),
            "convergence": scan.convergence.as_dict() if scan.convergence else None,
            "noise": _jsonable(scan.noise_record),
            "settings": _jsonable(scan.settings),
            "wall_time_s": scan.wall_time_s,
        }

    def scan_filtered(self, key: Optional[str], channel: str, method: str) -> dict:
        key = key or self.last_scan_key
        scan = self.project.scans[key]
        data = np.asarray(scan.filtered(channel, method), dtype=np.float32)
        return {"key": key, "channel": channel, "method": method,
                "shape": list(data.shape),
                "data_b64": base64.b64encode(data.tobytes()).decode("ascii"),
                "statistics": {"min": float(data.min()), "max": float(data.max()),
                               "mean": float(data.mean()), "std": float(data.std())}}

    def scan_features(self, key: Optional[str] = None, **kwargs) -> dict:
        key = key or self.last_scan_key
        scan = self.project.scans[key]
        feats = scan.detect_features(**kwargs)
        return {"key": key, "features": [f.as_dict() for f in feats],
                "kinds": sorted({f.kind for f in feats}),
                "border_rejected": getattr(scan, "_border_rejected", 0),
                "note": ("A maximum in a scanning-probe image is a maximum of the "
                         "measured signal, which is the local density of states for "
                         "STM and the tip-sample force for AFM. It need not sit on a "
                         "nucleus.")}

    def scan_identify(self, x_A: float, y_A: float, key: Optional[str] = None) -> dict:
        key = key or self.last_scan_key
        if key is None:
            raise ValueError("No scan available.")
        return _jsonable(self.project.scans[key].identify_at(float(x_A), float(y_A)))

    def scan_line_profile(self, x0: float, y0: float, x1: float, y1: float,
                          key: Optional[str] = None, channel: Optional[str] = None,
                          n: int = 256) -> dict:
        key = key or self.last_scan_key
        scan = self.project.scans[key]
        data = scan.channel(channel)
        ny, nx = data.shape
        xs = np.linspace(x0, x1, n)
        ys = np.linspace(y0, y1, n)
        cols, rows = [], []
        for x, y in zip(xs, ys):
            c, r = scan.xy_to_pixel(x, y)
            cols.append(np.clip(int(round(c)), 0, nx - 1))
            rows.append(np.clip(int(round(r)), 0, ny - 1))
        values = data[rows, cols]
        distance = np.hypot(xs - x0, ys - y0)
        return {"distance_A": distance.tolist(), "values": values.tolist(),
                "unit": scan.units.get(channel or scan.primary_channel, ""),
                "channel": channel or scan.primary_channel}

    def spectroscopy(self, x_A: float, y_A: float, height_A: float = 5.0,
                     bias_range: Sequence[float] = (-2.0, 2.0)) -> dict:
        s = self._require_structure()
        material = self._active_material()
        solver = self.lab.microscope._tb_for(s, material)
        sim = STMSimulator(solver, Tip("W"))
        z = float(s.positions[:, 2].max()) + height_A
        res = sim.spectroscopy(s, float(x_A), float(y_A), z,
                               bias_range_V=tuple(bias_range))
        if not res.supported:
            return {"supported": False, "reason": res.unsupported_reason,
                    "suggested_models": res.suggested_models}
        self.project.add_result(f"sts::{time.time():.0f}", res)
        self._autosave()
        return {"supported": True, "bias_V": res.value["bias_V"].tolist(),
                "dIdV": res.value["dIdV"].tolist(),
                "fermi_level_eV": res.extra["fermi_level_eV"],
                "provenance": res.provenance.as_dict(), "unit": res.unit}

    def force_curve(self, x_A: float, y_A: float) -> dict:
        s = self._require_structure()
        res = AFMSimulator(Tip("W")).force_curve(s, float(x_A), float(y_A))
        if not res.supported:
            return {"supported": False, "reason": res.unsupported_reason}
        v = res.value
        return {"supported": True,
                "height_A": v["height_above_surface_A"].tolist(),
                "force_nN": v["force_nN"].tolist(),
                "frequency_shift_Hz": v["frequency_shift_Hz"].tolist(),
                "provenance": res.provenance.as_dict()}

    def instrument_options(self) -> dict:
        return {
            "tips": sorted(TIP_WORK_FUNCTIONS),
            "tip_work_functions": TIP_WORK_FUNCTIONS,
            "stm_modes": ["constant-current", "constant-height"],
            "afm_modes": ["contact", "constant-height", "fm-afm",
                          "constant-frequency-shift"],
            "noise_presets": ["quiet", "realistic", "noisy"],
            "palettes": list_palettes(),
            "filters": ["none", "plane", "line", "median", "plane+median",
                        "plane+line", "plane+line+median"],
        }

    def solvers(self) -> dict:
        return {"registered": solver_registry.describe_all(),
                "external": availability_report(),
                "plugins": self.plugins.report()}

    def gpaw_status(self, refresh: bool = False) -> dict:
        """Whether GPAW is usable, and what is missing when it is not."""
        from ..solvers.gpaw_driver import discover

        environment = discover(refresh=bool(refresh))
        return {"environment": environment.as_dict(),
                "presets": self.gpaw_presets()}

    def gpaw_presets(self) -> List[dict]:
        from ..solvers.gpaw_driver import describe_presets

        return describe_presets()

    def gpaw_settings(self, preset: str = "molecule", **overrides) -> dict:
        """Validate a configuration without running it.

        The interface calls this as the user edits, so a bad combination is
        reported before a job is ever created.
        """
        from ..solvers.gpaw_driver import GPAWSettingsError
        from ..solvers.gpaw_driver import preset as build_preset

        structure = self.structure
        try:
            settings = build_preset(preset, structure, **overrides)
        except GPAWSettingsError as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "settings": settings.as_dict(),
                "summary": settings.summary(), "units": settings.units()}

    def _gpaw_submission(self, structure: Structure, configuration: dict,
                         apply_forces: bool = False,
                         timeout_s: Optional[float] = None) -> "GPAWSubmission":
        """Freeze everything one calculation needs, before it is queued.

        The worker specification and the validated configuration are copied and
        fingerprinted here.  Nothing the calculation depends on is read again
        from live state afterwards, so editing the structure while GPAW runs
        cannot change what was computed, only whether the answer may be
        attached to what is now on screen.
        """
        from ..solvers.gpaw_driver.conversion import input_digest, to_worker_spec

        spec = to_worker_spec(structure)
        settings = dict(configuration)
        return GPAWSubmission(
            run_id=f"{int(time.time() * 1000):013d}-{uuid.uuid4().hex[:6]}",
            project=self.project,
            structure_key=self.project.key_of(structure),
            spec=spec,
            settings=settings,
            digest=input_digest(spec, settings),
            apply_forces=bool(apply_forces),
            timeout_s=timeout_s,
        )

    def gpaw_energy(self, settings: Optional[dict] = None, preset: str = "molecule",
                    background: bool = True, apply_forces: bool = False,
                    timeout_s: Optional[float] = None, **overrides) -> dict:
        """Run one GPAW single-point calculation on the active structure."""
        from ..solvers.gpaw_driver import GPAWSettingsError
        from ..solvers.gpaw_driver import preset as build_preset

        structure = self._require_structure()
        try:
            if settings is None:
                configuration = build_preset(preset, structure, **overrides).as_dict()
            else:
                merged = dict(settings)
                merged.update(overrides)
                configuration = build_preset(preset, structure, **merged).as_dict()
        except GPAWSettingsError as exc:
            self.warn(str(exc), level="error", source="solver/gpaw")
            return {"ok": False, "error": str(exc)}

        submission = self._gpaw_submission(structure, configuration, apply_forces,
                                           timeout_s)

        if not background:
            return self._run_gpaw(submission, _NullJob())

        job = self.jobs.submit(
            "gpaw", f"GPAW {configuration['xc']} energy, {len(structure)} atoms",
            (lambda j: self._run_gpaw(submission, j)),
            owner=submission.project)
        return {"job": job.as_dict(), "run_id": submission.run_id,
                "inputs_digest": submission.digest}

    def gpaw_runs(self) -> List[dict]:
        """Every GPAW run held by the active project, newest first."""
        runs: Dict[str, dict] = {}
        for key, result in self.project.results.items():
            if not key.startswith("gpaw::"):
                continue
            parts = key.split("::")
            if len(parts) != 3:
                continue
            run = runs.setdefault(parts[1], {"run_id": parts[1], "keys": []})
            run["keys"].append(key)
            if parts[2] == "energy":
                run["supported"] = bool(result.supported)
                run["value_eV"] = result.value
                run["inputs_digest"] = result.provenance.inputs_digest
                run["xc"] = result.provenance.parameters.get("xc")
                run["converged"] = bool(result.convergence.converged
                                        if result.convergence else False)
                run["status"] = result.extra.get("status", "")
        return [runs[key] for key in sorted(runs, reverse=True)]

    def _run_gpaw(self, submission: "GPAWSubmission", job) -> dict:
        """Run one frozen submission and give its answer to its own project."""
        from ..solvers.gpaw_driver import GPAWSolver
        from ..solvers.gpaw_driver.conversion import (
            apply_forces as write_forces, describe_difference, from_worker_spec,
            input_digest, to_worker_spec,
        )

        solver = GPAWSolver()
        configuration = submission.settings
        maximum = int(configuration.get("max_iterations", 1) or 1)
        snapshot = from_worker_spec(submission.spec)

        def progress(event: dict) -> None:
            if event.get("event") == "scf":
                iteration = int(event.get("iteration", 0))
                job.progress = min(0.99, iteration / max(1, maximum))
                job.message = f"SCF iteration {iteration}"
            elif event.get("event") == "start":
                job.message = f"starting {event.get('formula', '')}"

        out = solver.single_point(snapshot, settings=configuration,
                                  progress=progress,
                                  cancelled=job.cancel_flag.is_set,
                                  timeout_s=submission.timeout_s)
        energy = out.results["energy"]
        owner = submission.project
        for result in out.results.values():
            result.provenance.inputs_digest = submission.digest
            result.extra["run_id"] = submission.run_id
        for key, result in out.results.items():
            owner.add_result(f"gpaw::{submission.run_id}::{key}", result)

        payload = {
            "ok": bool(energy.supported),
            "run_id": submission.run_id,
            "inputs_digest": submission.digest,
            "settings": configuration,
            "converged": bool(out.convergence.converged) if out.convergence else False,
            "iterations": out.convergence.iterations if out.convergence else 0,
            "convergence_message": out.convergence.message if out.convergence else "",
            "status": energy.extra.get("status", "failed"),
            "wall_time_s": out.wall_time_s,
            "log": list(out.log)[:40],
            "energy": _jsonable(energy.as_dict()),
        }
        if energy.supported:
            forces = out.results["forces"]
            payload["forces"] = _jsonable(forces.as_dict(include_value=False))
            payload["max_force_eV_A"] = forces.extra["max_force_eV_A"]
            payload["energy_eV"] = float(energy.value)
            if submission.apply_forces:
                refusal = self._apply_gpaw_forces(submission, forces)
                if refusal:
                    payload["forces_refused"] = refusal
                    self.warn(refusal, level="warning", source="solver/gpaw")
                else:
                    payload["forces_applied"] = True
        else:
            payload["reason"] = energy.unsupported_reason
            self.warn(energy.unsupported_reason, level="unsupported", source="solver/gpaw")

        if owner is self.project:
            self._autosave()
            payload["state"] = self.state()
        else:
            payload["state"] = None
            payload["owner_replaced"] = True
        return payload

    def _apply_gpaw_forces(self, submission: "GPAWSubmission", forces) -> str:
        """Attach forces to the structure they were computed for, or refuse.

        Returns an empty string when the forces were applied, and the reason
        otherwise.  The check is a fingerprint of the whole input, not just the
        atom ids: a structure that has been moved, re-elemented, re-charged or
        reordered since the calculation started is a different calculation.
        """
        from ..solvers.gpaw_driver.conversion import (
            apply_forces as write_forces, describe_difference, input_digest,
            to_worker_spec,
        )

        owner = submission.project
        if owner is not self.project:
            return ("The forces were not attached: the project this calculation "
                    "was started from has been replaced.")
        if submission.structure_key is None:
            return ("The forces were not attached: the calculation was run on a "
                    "structure this project does not hold.")
        current = owner.structures.get(submission.structure_key)
        if current is None:
            return (f"The forces were not attached: structure "
                    f"{submission.structure_key} is no longer in the project.")
        now = to_worker_spec(current)
        if input_digest(now, submission.settings) != submission.digest:
            difference = describe_difference(submission.spec, now) or "the input changed"
            return (
                f"The forces were not attached: {difference} while GPAW was running, "
                "so they were computed for a different structure than the one now in "
                "the project. The energy and its provenance are kept. Re-run the "
                "calculation on the current geometry to get forces for it.")
        before = current.copy()
        before_selection = Selection(list(owner.selection.ids), owner.selection.query)
        write_forces(current, forces.extra["by_atom_id"])
        owner.record_change(
            current, before, before_selection, "GPAW forces", "solver.gpaw.forces",
            {"xc": submission.settings["xc"], "mode": submission.settings["mode"],
             "run_id": submission.run_id, "inputs_digest": submission.digest},
            result_summary=f"max |F| = {forces.extra['max_force_eV_A']:.4f} eV/A")
        return ""

    def _dft_environment(self):
        from ..solvers.gpaw_driver import discover

        return discover()

    def _dft_build(self, variables: Optional[dict] = None, structure=None):
        """A specification for the active structure, plus its check report."""
        from ..experiments.dft import spec as specs

        structure = structure if structure is not None else self._require_structure()
        environment = self._dft_environment()
        spec = specs.build(structure, environment,
                           structure_key=self.project.key_of(structure),
                           **dict(variables or {}))
        return spec, specs.check(spec, environment)

    def dft_status(self) -> dict:
        """GPAW, the specification the active structure would get, and stored runs."""
        from ..experiments.dft import convergence, records, relaxation
        from ..experiments.dft import bands as bands_module
        from ..experiments.dft import eos as eos_module
        from ..experiments.dft import ldos as ldos_module
        from ..experiments.dft import dos as dos_module
        from ..experiments.dft import spec as specs
        from ..experiments.dft.restart import RestartStore
        from ..provenance.classification import describe as describe_classes

        environment = self._dft_environment()
        payload: Dict[str, Any] = {
            "environment": environment.as_dict(),
            "runs": [records.summary(self.project, r) for r in records.run_ids(self.project)],
            "studies": self.dft_studies(),
            "fields": [f.as_dict() for f in specs.FIELDS],
            "study_parameters": convergence.PARAMETERS,
            "study_observables": convergence.OBSERVABLES,
            "classifications": describe_classes(),
            "structure": None,
            "relaxations": [records.relax_summary(self.project, r)
                            for r in records.relax_run_ids(self.project)],
            "relaxation_fields": [f.as_dict() for f in relaxation.FIELDS],
            "dos_runs": [records.dos_summary(self.project, r)
                         for r in records.dos_run_ids(self.project)],
            "dos_fields": [f.as_dict() for f in dos_module.FIELDS],
            "dos_sources": self.dft_dos_sources(),
            "bands_runs": [records.bands_summary(self.project, r)
                           for r in records.bands_run_ids(self.project)],
            "bands_fields": [f.as_dict() for f in bands_module.FIELDS],
            "eos_runs": [records.eos_summary(self.project, r)
                         for r in records.eos_run_ids(self.project)],
            "eos_fields": [f.as_dict() for f in eos_module.FIELDS],
            "ldos_runs": [records.ldos_summary(self.project, r)
                          for r in records.ldos_run_ids(self.project)],
            "ldos_fields": [f.as_dict() for f in ldos_module.FIELDS],
        }
        store = RestartStore()
        entries = store.entries()
        payload["restart_store"] = {"directory": store.directory, "entries": len(entries),
                                    "bytes": sum(int(e.get("size_bytes", 0)) for e in entries)}
        if self.structure is not None:
            spec, report = self._dft_build()
            payload["structure"] = {
                "key": self.project.active_structure_key,
                "formula": self.structure.formula(), "n_atoms": len(self.structure),
                "fixed_atoms": int(np.count_nonzero(self.structure.fixed)),
                "spec": specs.describe(spec, report), "report": report.as_dict(),
            }
        return payload

    def dft_spec(self, variables: Optional[dict] = None) -> dict:
        """Build and check a specification without running anything."""
        from ..experiments.dft import spec as specs

        try:
            spec, report = self._dft_build(variables)
        except specs.SpecError as exc:
            return {"ok": False, "error": str(exc), "blocking": [str(exc)],
                    "by_field": {}, "warnings": []}
        return {"ok": report.ok, "spec": specs.describe(spec, report),
                **report.as_dict()}

    def dft_run(self, variables: Optional[dict] = None, background: bool = True,
                reuse_restart: bool = False, keep_restart: bool = False,
                timeout_s: Optional[float] = None) -> dict:
        """Run one ground-state experiment on the active structure.

        A refused specification returns the reasons and starts no job.  The
        specification and geometry are frozen here, the job writes only to
        the project it was started from, and nothing it produces edits a
        structure or records an undo point.
        """
        from ..experiments.dft import spec as specs
        from ..experiments.dft.run import new_run_id

        try:
            spec, report = self._dft_build(variables)
        except specs.SpecError as exc:
            self.warn(str(exc), level="error", source="solver/dft")
            return {"ok": False, "error": str(exc), "blocking": [str(exc)], "by_field": {}}
        if not report.ok:
            reason = "; ".join(report.blocking)
            self.warn(reason, level="unsupported", source="solver/dft")
            return {"ok": False, "error": reason, "refused": True, **report.as_dict()}
        submission = DFTSubmission(run_id=new_run_id(), project=self.project,
                                   structure_key=spec.structure_key, spec=spec,
                                   reuse_restart=bool(reuse_restart),
                                   keep_restart=bool(keep_restart), timeout_s=timeout_s)
        if not background:
            return self._run_dft(submission, _NullJob())
        job = self.jobs.submit(
            "dft", f"DFT {spec.xc} ground state, {spec.formula}",
            (lambda j: self._run_dft(submission, j)), owner=submission.project)
        return {"job": job.as_dict(), "run_id": submission.run_id,
                "spec_digest": spec.digest}

    def _run_dft(self, submission: "DFTSubmission", job) -> dict:
        """Run one frozen submission and give the outcome to its own project."""
        from ..experiments.dft import records
        from ..experiments.dft.run import execute

        spec = submission.spec
        maximum = max(1, spec.max_scf_iterations)

        def progress(event: dict) -> None:
            kind = event.get("event")
            if kind == "scf":
                iteration = int(event.get("iteration", 0))
                job.progress = min(0.9, iteration / min(maximum, 60))
                change = event.get("energy_change_eV_per_electron")
                job.message = (f"SCF iteration {iteration}, E = "
                               f"{event.get('energy_eV', 0.0):.6f} eV"
                               + (f", dE = {change:.1e} eV/e" if change else ""))
            elif kind == "stage":
                job.progress = max(job.progress, 0.92)
                job.message = f"extracting {event.get('name', '')}"
            elif kind == "start":
                job.message = f"starting {event.get('formula', '')}"

        from ..experiments.dft.spec import SpecRefused

        owner = submission.project
        try:
            outcome = execute(spec, self._dft_environment(), run_id=submission.run_id,
                              progress=progress, cancelled=job.cancel_flag.is_set,
                              timeout_s=submission.timeout_s,
                              reuse_restart=submission.reuse_restart,
                              keep_restart=submission.keep_restart)
        except SpecRefused as exc:
            self.warn(str(exc), level="unsupported", source="solver/dft")
            return {"ok": False, "refused": True, "error": str(exc),
                    "reason": "The specification was refused when the job started, so "
                              "nothing ran and nothing was stored: " + str(exc),
                    **exc.report.as_dict()}
        records.store(owner, outcome)
        if not outcome.converged:
            level = "warning" if outcome.status == "cancelled" else "unsupported"
            self.warn(outcome.reason, level=level, source="solver/dft")
        payload = self.dft_result(submission.run_id, project=owner)
        payload["owner_replaced"] = owner is not self.project
        if owner is self.project:
            self._autosave()
        return payload

    def _dft_relax_build(self, variables: Optional[dict] = None, structure=None):
        from ..experiments.dft import relaxation

        structure = structure if structure is not None else self._require_structure()
        environment = self._dft_environment()
        spec = relaxation.build(structure, environment,
                                structure_key=self.project.key_of(structure),
                                **dict(variables or {}))
        return spec, relaxation.check(spec, environment)

    def dft_relax_spec(self, variables: Optional[dict] = None) -> dict:
        """Build and check a relaxation specification without running anything."""
        from ..experiments.dft import relaxation
        from ..experiments.dft import spec as specs

        try:
            spec, report = self._dft_relax_build(variables)
        except specs.SpecError as exc:
            return {"ok": False, "error": str(exc), "blocking": [str(exc)],
                    "by_field": {}, "warnings": []}
        return {"ok": report.ok, "spec": relaxation.describe(spec, report),
                **report.as_dict()}

    def dft_relax(self, variables: Optional[dict] = None, background: bool = True,
                  apply: bool = True, timeout_s: Optional[float] = None) -> dict:
        """Relax the active structure with GPAW.

        The specification and geometry are frozen here.  A refused
        specification starts nothing.  A relaxation that is cancelled, times
        out or fails stores nothing and changes nothing.  A finished one is
        stored in one step in the project it was started from and, if it
        converged and the structure is unchanged, applied as one undoable
        change.
        """
        from ..experiments.dft import records, relaxation
        from ..experiments.dft import spec as specs
        from ..experiments.dft.run import new_run_id

        try:
            spec, report = self._dft_relax_build(variables)
        except specs.SpecError as exc:
            self.warn(str(exc), level="error", source="solver/dft-relax")
            return {"ok": False, "status": "refused", "refused": True, "error": str(exc),
                    "blocking": [str(exc)], "by_field": {}}
        if not report.ok:
            reason = "; ".join(report.blocking)
            self.warn(reason, level="unsupported", source="solver/dft-relax")
            return {"ok": False, "status": "refused", "refused": True, "error": reason,
                    **report.as_dict()}
        owner = self.project
        run_id = new_run_id()
        maximum = max(1, spec.max_steps)

        def work(job) -> dict:
            def progress(event: dict) -> None:
                kind = event.get("event")
                if kind == "relax":
                    step = int(event.get("step", 0))
                    job.progress = min(0.9, (step + 1) / (maximum + 1))
                    residual = event.get("stress_residual_eV_A3")
                    job.message = (f"ionic step {step}, E = {event.get('energy_free_eV', 0.0):.6f}"
                                   f" eV, max |F| = {event.get('max_force_eV_A', 0.0):.3e} eV/A"
                                   + (f", stress {residual:.2e} eV/A^3"
                                      if residual is not None else ""))
                elif kind == "scf":
                    job.message = (job.message.split(" | ")[0] + " | SCF iteration "
                                   f"{int(event.get('iteration', 0))}")
                elif kind == "stage":
                    job.progress = max(job.progress, 0.92)
                    job.message = f"extracting {event.get('name', '')}"

            try:
                outcome = relaxation.execute(spec, self._dft_environment(), run_id=run_id,
                                             progress=progress,
                                             cancelled=job.cancel_flag.is_set,
                                             timeout_s=timeout_s)
            except specs.SpecRefused as exc:
                self.warn(str(exc), level="unsupported", source="solver/dft-relax")
                return {"ok": False, "status": "refused", "refused": True,
                        "error": str(exc), "run_id": run_id, **exc.report.as_dict()}
            if not outcome.ok:
                self.warn(outcome.reason, level="warning" if outcome.status == "cancelled"
                          else "error", source="solver/dft-relax")
                return {"ok": False, "status": outcome.status, "reason": outcome.reason,
                        "run_id": run_id, "audit": _jsonable(outcome.audit),
                        "owner_replaced": owner is not self.project}
            stored = records.store_relaxation(owner, outcome, apply=apply)
            payload = self.dft_relax_result(run_id, project=owner)
            payload["apply_reason"] = stored["apply_reason"]
            payload["owner_replaced"] = owner is not self.project
            if owner is self.project:
                self._autosave()
                payload["app_state"] = self.state()
            return payload

        if not background:
            return work(_NullJob())
        job = self.jobs.submit(
            "dft-relax", f"DFT {spec.mode} relaxation, {spec.ground_state.formula}",
            work, owner=owner)
        return {"job": job.as_dict(), "run_id": run_id, "spec_digest": spec.digest}

    def dft_relax_runs(self) -> List[dict]:
        from ..experiments.dft import records

        return [records.relax_summary(self.project, r)
                for r in records.relax_run_ids(self.project)]

    def dft_relax_result(self, run_id: Optional[str] = None,
                         project: Optional[Project] = None) -> dict:
        """Everything the interface shows about one relaxation."""
        from ..experiments.dft import records
        from ..experiments.dft import relaxation
        from ..experiments.dft import spec as specs

        project = project or self.project
        ids = records.relax_run_ids(project)
        if run_id is None:
            if not ids:
                return {"ok": False, "error": "No DFT relaxation is stored."}
            run_id = ids[0]
        record = records.relax_record(project, run_id)
        spec = records.relax_spec_of(project, run_id)
        if record is None or spec is None:
            return {"ok": False, "error": f"No DFT relaxation {run_id}."}
        info = records.relax_status(project, run_id)
        get = lambda name: project.results.get(records.relax_key(run_id, name))
        value = dict(record.value or {})
        displacement = dict(value.get("displacement") or {})
        prov = record.provenance
        payload: Dict[str, Any] = {
            "ok": True, "run_id": run_id, "status": record.extra.get("status"),
            "mode": spec.mode, "optimizer": spec.optimizer,
            "run_state": info.get("state"), "current": bool(info.get("current")),
            "applicable": bool(info.get("applicable")), "applied": bool(info.get("applied")),
            "state_reason": info.get("reason"),
            "apply_reason": record.extra.get("apply_reason"),
            "formula": spec.ground_state.formula, "boundary": spec.ground_state.boundary,
            "structure_key": spec.structure_key, "spec_digest": spec.digest,
            "final_spec_digest": value.get("final_ground_state_digest"),
            "input_geometry_digest": value.get("input_geometry_digest"),
            "output_geometry_digest": value.get("output_geometry_digest"),
            "convergence": record.convergence.as_dict() if record.convergence else None,
            "optimizer_steps": value.get("optimizer_steps"),
            "scf_iterations_total": value.get("scf_iterations_total"),
            "stop_reason": value.get("stop_reason"),
            "initial_energy_eV": value.get("initial_energy_eV"),
            "final_energy_eV": value.get("final_energy_eV"),
            "energy_change_eV": value.get("energy_change_eV"),
            "initial_max_force_eV_A": value.get("initial_max_force_eV_A"),
            "final_max_force_eV_A": value.get("final_max_force_eV_A"),
            "final_stress_residual_eV_A3": value.get("final_stress_residual_eV_A3"),
            "displacement": {k: displacement.get(k) for k in ("max_A", "rms_A",
                                                              "max_atom_id")},
            "cell": _jsonable(value.get("cell")),
            "history": _jsonable(value.get("history") or []),
            "fixed_atom_ids": list(value.get("fixed_atom_ids") or []),
            "settings": spec.settings(),
            "fmax_eV_A": spec.fmax_eV_A, "stress_tol_eV_A3": spec.stress_tol_eV_A3,
            "warnings": list(record.extra.get("warnings", [])),
            "wall_time_s": record.extra.get("wall_time_s"),
            "provenance": {
                "model": prov.model, "fidelity": prov.fidelity.value,
                "origin": prov.origin.value, "approximations": list(prov.approximations),
                "boundary_conditions": prov.boundary_conditions,
                "inputs_digest": prov.inputs_digest, "dataset": prov.dataset,
                "references": list(prov.references),
                "software_pinned": prov.parameters.get("software_pinned"),
                "software_reported": prov.parameters.get("software_reported"),
                "paw_datasets": prov.parameters.get("paw_datasets"),
                "gpaw_parameters": prov.parameters.get("gpaw_parameters"),
                "gpaw_parameters_used": prov.parameters.get("gpaw_parameters_used"),
                "relaxation_settings": prov.parameters.get("relaxation_settings"),
                "relaxation_used": prov.parameters.get("relaxation_used"),
            },
        }
        energy = get("energy")
        if energy is not None:
            payload["energy_eV"] = energy.value
            payload["energy_per_atom_eV"] = energy.extra.get("energy_per_atom_eV")
        stress = get("stress")
        if stress is not None:
            payload["stress_eV_A3"] = _jsonable(stress.value)
            payload["pressure_GPa"] = stress.extra.get("pressure_GPa")
        accounting = get("charge_accounting")
        if accounting is not None:
            payload["charge_accounting"] = _jsonable(accounting.value)
        return _jsonable(payload)

    def dft_relax_apply(self, run_id: Optional[str] = None) -> dict:
        """Apply a stored relaxation as one undoable change, or say why not."""
        try:
            message = self.lab.dft.relax_apply(run_id)
        except ApiError as exc:
            self.warn(str(exc), level="unsupported", source="solver/dft-relax")
            return {"ok": False, "error": str(exc)}
        self._autosave()
        return {"ok": True, "message": message, "app_state": self.state()}

    def _dft_dos_build(self, variables: Optional[dict] = None,
                       source: Optional[dict] = None):
        from ..experiments.dft import dos

        source = dict(source or {})
        kind = source.get("kind") or "structure"
        environment = self._dft_environment()
        if kind == "structure":
            structure = self._require_structure()
            spec = dos.build(structure, environment,
                             structure_key=self.project.key_of(structure),
                             **dict(variables or {}))
        else:
            spec = dos.build_from_run(self.project, kind, str(source.get("run_id") or ""),
                                      environment, **dict(variables or {}))
        return spec, dos.check(spec, environment)

    def dft_dos_sources(self) -> List[dict]:
        """Stored converged runs whose geometry a DOS can be taken of."""
        from ..experiments.dft import records

        out = []
        for run_id in records.run_ids(self.project):
            rec = records.record(self.project, run_id)
            spec = records.spec_of(self.project, run_id)
            if rec is not None and rec.supported and spec is not None:
                out.append({"kind": "ground-state", "run_id": run_id,
                            "label": f"ground state {run_id[-6:]}, {spec.formula}, {spec.xc}"})
        for run_id in records.relax_run_ids(self.project):
            rec = records.relax_record(self.project, run_id)
            spec = records.relax_spec_of(self.project, run_id)
            if rec is not None and rec.extra.get("status") == "converged" and spec is not None:
                out.append({"kind": "relaxation", "run_id": run_id,
                            "label": f"relaxed geometry {run_id[-6:]}, "
                                     f"{spec.ground_state.formula}, {spec.mode}"})
        return out

    def dft_dos_spec(self, variables: Optional[dict] = None,
                     source: Optional[dict] = None) -> dict:
        """Build and check a DOS specification without running anything."""
        from ..experiments.dft import dos
        from ..experiments.dft import spec as specs

        try:
            spec, report = self._dft_dos_build(variables, source)
        except specs.SpecError as exc:
            return {"ok": False, "error": str(exc), "blocking": [str(exc)],
                    "by_field": {}, "warnings": []}
        return {"ok": report.ok, "spec": dos.describe(spec, report), **report.as_dict()}

    def dft_dos_run(self, variables: Optional[dict] = None, source: Optional[dict] = None,
                    background: bool = True, timeout_s: Optional[float] = None) -> dict:
        """Compute the DOS and PDOS of the active structure or of a stored run.

        The specification is frozen here.  A refused one starts nothing; a DOS
        that is cancelled, times out or fails stores nothing.  A complete one
        is stored in one step in the project it was started from, with one
        history line that is not an undo point; no structure is changed.
        """
        from ..experiments.dft import dos, records
        from ..experiments.dft import spec as specs
        from ..experiments.dft.run import new_run_id

        try:
            spec, report = self._dft_dos_build(variables, source)
        except specs.SpecError as exc:
            self.warn(str(exc), level="error", source="solver/dft-dos")
            return {"ok": False, "status": "refused", "refused": True, "error": str(exc),
                    "blocking": [str(exc)], "by_field": {}}
        if not report.ok:
            reason = "; ".join(report.blocking)
            self.warn(reason, level="unsupported", source="solver/dft-dos")
            return {"ok": False, "status": "refused", "refused": True, "error": reason,
                    **report.as_dict()}
        owner = self.project
        run_id = new_run_id()

        def work(job) -> dict:
            def progress(event: dict) -> None:
                kind = event.get("event")
                if kind == "scf":
                    job.progress = min(0.6, 0.6 * int(event.get("iteration", 0)) / 40.0)
                    job.message = f"ground state, SCF iteration {int(event.get('iteration', 0))}"
                elif kind == "stage":
                    name = event.get("name", "")
                    job.progress = max(job.progress, {"nscf": 0.65, "dos": 0.9,
                                                      "pdos": 0.95}.get(name, 0.62))
                    job.message = {"nscf": "non-self-consistent step on the DOS grid",
                                   "dos": "evaluating the DOS",
                                   "pdos": "evaluating the projections"}.get(
                        name, f"extracting {name}")

            try:
                outcome = dos.execute(spec, self._dft_environment(), run_id=run_id,
                                      progress=progress, cancelled=job.cancel_flag.is_set,
                                      timeout_s=timeout_s)
            except specs.SpecRefused as exc:
                self.warn(str(exc), level="unsupported", source="solver/dft-dos")
                return {"ok": False, "status": "refused", "refused": True,
                        "error": str(exc), "run_id": run_id, **exc.report.as_dict()}
            if not outcome.ok:
                self.warn(outcome.reason, level="warning" if outcome.status == "cancelled"
                          else "error", source="solver/dft-dos")
                return {"ok": False, "status": outcome.status, "reason": outcome.reason,
                        "run_id": run_id, "audit": _jsonable(outcome.audit),
                        "owner_replaced": owner is not self.project}
            records.store_dos(owner, outcome)
            payload = self.dft_dos_result(run_id, project=owner)
            payload["owner_replaced"] = owner is not self.project
            if owner is self.project:
                self._autosave()
            return payload

        if not background:
            return work(_NullJob())
        job = self.jobs.submit("dft-dos", f"DFT DOS, {spec.ground_state.formula}", work,
                               owner=owner)
        return {"job": job.as_dict(), "run_id": run_id, "spec_digest": spec.digest}

    def dft_dos_runs(self) -> List[dict]:
        from ..experiments.dft import records

        return [records.dos_summary(self.project, r) for r in records.dos_run_ids(self.project)]

    def dft_dos_result(self, run_id: Optional[str] = None, project: Optional[Project] = None,
                       max_points: int = 4001) -> dict:
        """One stored DOS with its curves, checks, state and provenance."""
        from ..experiments.dft import records

        project = project or self.project
        ids = records.dos_run_ids(project)
        if run_id is None:
            if not ids:
                return {"ok": False, "error": "No DOS is stored."}
            run_id = ids[0]
        record = records.dos_record(project, run_id)
        spec = records.dos_spec_of(project, run_id)
        if record is None or spec is None:
            return {"ok": False, "error": f"No DOS {run_id}."}
        info = records.dos_status(project, run_id)
        array = lambda name: project.arrays.get(records.dos_key(run_id, name))
        energies = np.asarray(array("energies").data, dtype=float)
        stride = max(1, int(math.ceil(len(energies) / max(2, int(max_points)))))
        cut = lambda values: np.asarray(values, dtype=float)[..., ::stride].tolist()
        curves: Dict[str, Any] = {"energies_eV": cut(energies),
                                  "total": cut(array("dos_total").data)}
        if array("dos_spin") is not None:
            spin = np.asarray(array("dos_spin").data)
            curves["up"], curves["down"] = cut(spin[0]), cut(spin[1])
        projections = []
        pdos = array("pdos")
        pspin = array("pdos_spin")
        for index, (label, ids_, angular) in enumerate(spec.projections):
            item = {"label": label, "atom_ids": list(ids_), "angular": angular,
                    "total": cut(pdos.data[index]) if pdos is not None else []}
            if pspin is not None:
                item["up"] = cut(pspin.data[index][0])
                item["down"] = cut(pspin.data[index][1])
            projections.append(item)
        value = dict(record.value or {})
        prov = record.provenance
        return _jsonable({
            "ok": True, "run_id": run_id, "status": value.get("status"),
            "run_state": info.get("state"), "current": bool(info.get("current")),
            "state_reason": info.get("reason"), "source": dict(spec.source),
            "formula": spec.ground_state.formula, "boundary": spec.ground_state.boundary,
            "structure_key": spec.structure_key, "spec_digest": spec.digest,
            "settings": spec.settings(), "fermi_level_eV": value.get("fermi_level_eV"),
            "reference": spec.energy_reference, "reference_eV": value.get("reference_eV"),
            "unit": "states/eV per cell", "npoints": spec.npoints,
            "display_stride": stride, "curves": curves, "projections": projections,
            "checks": value.get("checks"), "n_ibz_kpoints": value.get("n_ibz_kpoints"),
            "n_bz_kpoints": value.get("n_bz_kpoints"), "n_spins": value.get("n_spins"),
            "convergence": record.convergence.as_dict() if record.convergence else None,
            "warnings": list(record.extra.get("warnings", [])),
            "arrays": {name: stored.summary() for name, stored in (
                (n, array(n)) for n in ("energies", "dos_total", "dos_spin", "pdos",
                                        "pdos_spin", "eigenvalues", "kpoint_weights"))
                if stored is not None},
            "provenance": {
                "model": prov.model, "fidelity": prov.fidelity.value,
                "origin": prov.origin.value, "approximations": list(prov.approximations),
                "boundary_conditions": prov.boundary_conditions,
                "inputs_digest": prov.inputs_digest, "dataset": prov.dataset,
                "references": list(prov.references),
                "software_pinned": prov.parameters.get("software_pinned"),
                "software_reported": prov.parameters.get("software_reported"),
                "paw_datasets": prov.parameters.get("paw_datasets"),
                "gpaw_parameters": prov.parameters.get("gpaw_parameters"),
                "gpaw_parameters_used": prov.parameters.get("gpaw_parameters_used"),
                "nscf_parameters": prov.parameters.get("nscf_parameters"),
                "nscf_parameters_used": prov.parameters.get("nscf_parameters_used"),
                "dos_settings": prov.parameters.get("dos_settings"),
                "dos_used": prov.parameters.get("dos_used"),
            },
        })

    def dft_dos_export(self, path: str, run_id: Optional[str] = None) -> dict:
        """Write a stored DOS and its projections to CSV, full resolution."""
        from ..experiments.dft import records

        ids = records.dos_run_ids(self.project)
        run_id = run_id or (ids[0] if ids else None)
        spec = records.dos_spec_of(self.project, run_id or "")
        record = records.dos_record(self.project, run_id or "")
        if spec is None or record is None:
            raise ValueError(f"No DOS {run_id} is stored.")
        array = lambda name: self.project.arrays.get(records.dos_key(run_id, name))
        reference = "E-E_F" if spec.energy_reference == "fermi-level" else "E"
        columns: Dict[str, List[Any]] = {
            f"{reference}_eV": [f"{v:.10g}" for v in array("energies").data],
            "total_states_per_eV": [f"{v:.10g}" for v in array("dos_total").data]}
        if array("dos_spin") is not None:
            columns["up_states_per_eV"] = [f"{v:.10g}" for v in array("dos_spin").data[0]]
            columns["down_states_per_eV"] = [f"{v:.10g}" for v in array("dos_spin").data[1]]
        for index, (label, _, _) in enumerate(spec.projections):
            name = label.replace(" ", "_").replace(",", "_")
            columns[f"pdos_{name}_states_per_eV"] = [
                f"{v:.10g}" for v in array("pdos").data[index]]
        value = record.value or {}
        header = (f"DOS {run_id} of {spec.ground_state.formula}, {spec.ground_state.xc}, "
                  f"{spec.broadening} width {spec.width_eV} eV, Fermi level "
                  f"{value.get('fermi_level_eV'):.6f} eV, specification {spec.digest}\n"
                  f"{record.provenance.model}; states per eV per cell, both spins in total")
        return {"path": dataio.write_csv(path, columns, header), "rows": spec.npoints,
                "columns": list(columns)}

    def _dft_bands_build(self, variables: Optional[dict] = None,
                         source: Optional[dict] = None):
        from ..experiments.dft import bands

        source = dict(source or {})
        kind = source.get("kind") or "structure"
        environment = self._dft_environment()
        if kind == "structure":
            structure = self._require_structure()
            spec = bands.build(structure, environment,
                               structure_key=self.project.key_of(structure),
                               **dict(variables or {}))
        else:
            spec = bands.build_from_run(self.project, kind, str(source.get("run_id") or ""),
                                        environment, **dict(variables or {}))
        return spec, bands.check(spec, environment)

    def dft_bands_spec(self, variables: Optional[dict] = None,
                       source: Optional[dict] = None) -> dict:
        """Build and check a band-structure specification without running anything."""
        from ..experiments.dft import bands
        from ..experiments.dft import spec as specs

        try:
            spec, report = self._dft_bands_build(variables, source)
        except specs.SpecError as exc:
            return {"ok": False, "error": str(exc), "blocking": [str(exc)],
                    "by_field": {}, "warnings": []}
        return _jsonable({"ok": report.ok, "spec": bands.describe(spec, report),
                          **report.as_dict()})

    def dft_bands_run(self, variables: Optional[dict] = None, source: Optional[dict] = None,
                      background: bool = True, timeout_s: Optional[float] = None) -> dict:
        """Compute the band structure of the active structure or of a stored run.

        The specification, path included, is frozen here.  A refused one starts
        nothing; one that is cancelled, times out, fails or returns anything
        that does not verify stores nothing.  A complete one is stored in one
        step in the project it was started from, with one history line that
        is not an undo point; no structure is changed.
        """
        from ..experiments.dft import bands, records
        from ..experiments.dft import spec as specs
        from ..experiments.dft.run import new_run_id

        try:
            spec, report = self._dft_bands_build(variables, source)
        except specs.SpecError as exc:
            self.warn(str(exc), level="error", source="solver/dft-bands")
            return {"ok": False, "status": "refused", "refused": True, "error": str(exc),
                    "blocking": [str(exc)], "by_field": {}}
        if not report.ok:
            reason = "; ".join(report.blocking)
            self.warn(reason, level="unsupported", source="solver/dft-bands")
            return {"ok": False, "status": "refused", "refused": True, "error": reason,
                    **report.as_dict()}
        owner = self.project
        run_id = new_run_id()

        def work(job) -> dict:
            def progress(event: dict) -> None:
                kind = event.get("event")
                if kind == "scf":
                    job.progress = min(0.6, 0.6 * int(event.get("iteration", 0)) / 40.0)
                    job.message = f"ground state, SCF iteration {int(event.get('iteration', 0))}"
                elif kind == "stage":
                    name = event.get("name", "")
                    job.progress = max(job.progress, {"bands": 0.7}.get(name, 0.62))
                    job.message = ("non-self-consistent step along the path"
                                   if name == "bands" else f"extracting {name}")

            try:
                outcome = bands.execute(spec, self._dft_environment(), run_id=run_id,
                                        progress=progress, cancelled=job.cancel_flag.is_set,
                                        timeout_s=timeout_s)
            except specs.SpecRefused as exc:
                self.warn(str(exc), level="unsupported", source="solver/dft-bands")
                return {"ok": False, "status": "refused", "refused": True,
                        "error": str(exc), "run_id": run_id, **exc.report.as_dict()}
            if not outcome.ok:
                self.warn(outcome.reason, level="warning" if outcome.status == "cancelled"
                          else "error", source="solver/dft-bands")
                return {"ok": False, "status": outcome.status, "reason": outcome.reason,
                        "run_id": run_id, "audit": _jsonable(outcome.audit),
                        "owner_replaced": owner is not self.project}
            try:
                records.store_bands(owner, outcome)
            except ValueError as exc:
                self.warn(str(exc), level="error", source="solver/dft-bands")
                return {"ok": False, "status": "failed", "reason": str(exc),
                        "run_id": run_id}
            payload = self.dft_bands_result(run_id, project=owner)
            payload["owner_replaced"] = owner is not self.project
            if owner is self.project:
                self._autosave()
            return payload

        if not background:
            return work(_NullJob())
        job = self.jobs.submit("dft-bands", f"DFT band structure, {spec.ground_state.formula}",
                               work, owner=owner)
        return {"job": job.as_dict(), "run_id": run_id, "spec_digest": spec.digest}

    def dft_bands_runs(self) -> List[dict]:
        from ..experiments.dft import records

        return [records.bands_summary(self.project, r)
                for r in records.bands_run_ids(self.project)]

    def dft_bands_result(self, run_id: Optional[str] = None,
                         project: Optional[Project] = None) -> dict:
        """One stored band structure with its bands, path, checks, state and provenance."""
        from ..experiments.dft import records

        project = project or self.project
        ids = records.bands_run_ids(project)
        if run_id is None:
            if not ids:
                return {"ok": False, "error": "No band structure is stored."}
            run_id = ids[0]
        record = records.bands_record(project, run_id)
        spec = records.bands_spec_of(project, run_id)
        if record is None or spec is None:
            return {"ok": False, "error": f"No band structure {run_id}."}
        info = records.bands_status(project, run_id)
        if info.get("state") == "corrupt":
            return {"ok": False, "status": "corrupt", "run_id": run_id,
                    "run_state": "corrupt", "error": info.get("reason")}
        array = lambda name: project.arrays.get(records.bands_key(run_id, name))
        value = dict(record.value or {})
        fermi = float(value.get("fermi_level_eV"))
        eigen = np.asarray(array("eigenvalues").data, dtype=float)
        prov = record.provenance
        return _jsonable({
            "ok": True, "run_id": run_id, "status": value.get("status"),
            "run_state": info.get("state"), "current": bool(info.get("current")),
            "state_reason": info.get("reason"), "source": dict(spec.source),
            "formula": spec.ground_state.formula, "boundary": spec.ground_state.boundary,
            "xc": spec.ground_state.xc, "structure_key": spec.structure_key,
            "spec_digest": spec.digest, "settings": spec.settings(),
            "fermi_level_eV": fermi, "reference": spec.energy_reference,
            "n_spins": int(eigen.shape[0]), "n_kpoints": int(eigen.shape[1]),
            "n_bands": int(eigen.shape[2]),
            "distance_invA": np.asarray(array("distance").data, dtype=float),
            "kpoints_frac": np.asarray(array("kpoints_frac").data, dtype=float),
            "bands_relative_eV": np.transpose(eigen - fermi, (0, 2, 1)),
            "labels": value.get("labels"), "breaks": value.get("breaks"),
            "ticks": value.get("ticks"), "path": value.get("path"),
            "special_points": value.get("special_points"),
            "segment_intervals": value.get("segment_intervals"),
            "path_length_invA": value.get("path_length_invA"),
            "reciprocal_cell_invA": value.get("reciprocal_cell_invA"),
            "band_edges": value.get("band_edges"), "checks": value.get("checks"),
            "units": value.get("units"), "occupations": value.get("occupations"),
            "orbital_character": value.get("orbital_character"),
            "experimental_comparison": value.get("experimental_comparison"),
            "started_unix": value.get("started_unix"),
            "finished_unix": value.get("finished_unix"),
            "wall_time_s": value.get("wall_time_s"),
            "array_sha256": value.get("array_sha256"),
            "convergence": record.convergence.as_dict() if record.convergence else None,
            "warnings": list(record.extra.get("warnings", [])),
            "arrays": {name: array(name).summary() for name in records.BANDS_ARRAYS},
            "provenance": {
                "model": prov.model, "fidelity": prov.fidelity.value,
                "origin": prov.origin.value, "approximations": list(prov.approximations),
                "boundary_conditions": prov.boundary_conditions,
                "inputs_digest": prov.inputs_digest, "dataset": prov.dataset,
                "references": list(prov.references), "created_unix": prov.created_unix,
                "software_pinned": prov.parameters.get("software_pinned"),
                "software_reported": prov.parameters.get("software_reported"),
                "paw_datasets": prov.parameters.get("paw_datasets"),
                "gpaw_parameters": prov.parameters.get("gpaw_parameters"),
                "gpaw_parameters_used": prov.parameters.get("gpaw_parameters_used"),
                "nscf_parameters": prov.parameters.get("nscf_parameters"),
                "nscf_parameters_used": prov.parameters.get("nscf_parameters_used"),
                "band_settings": prov.parameters.get("band_settings"),
                "bands_used": prov.parameters.get("bands_used"),
                "path_origin": prov.parameters.get("path_origin"),
                "tolerances": dict(prov.tolerances),
                "scf_iterations": prov.parameters.get("scf_iterations"),
            },
        })

    def dft_bands_export(self, path: str, run_id: Optional[str] = None) -> dict:
        """Write a stored band structure to CSV: one row per spin and k-point, full precision."""
        from ..experiments.dft import records

        ids = records.bands_run_ids(self.project)
        run_id = run_id or (ids[0] if ids else None)
        spec = records.bands_spec_of(self.project, run_id or "")
        record = records.bands_record(self.project, run_id or "")
        if spec is None or record is None:
            raise ValueError(f"No band structure {run_id} is stored.")
        damage = records.bands_integrity(self.project, run_id)
        if damage:
            raise ValueError(f"Band structure {run_id} is not exported: {damage}")
        array = lambda name: np.asarray(
            self.project.arrays[records.bands_key(run_id, name)].data, dtype=float)
        value = record.value or {}
        fermi = float(value["fermi_level_eV"])
        eigen, kfrac, distance = array("eigenvalues"), array("kpoints_frac"), array("distance")
        labels = list(value.get("labels") or [""] * len(distance))
        breaks = set(value.get("breaks") or [])
        columns: Dict[str, List[Any]] = {k: [] for k in (
            "spin", "k_index", "distance_invA", "k1_frac", "k2_frac", "k3_frac", "label",
            "branch_start")}
        for n in range(eigen.shape[2]):
            columns[f"band_{n + 1}_E-E_F_eV"] = []
        spins = ["up", "down"] if eigen.shape[0] == 2 else ["both"]
        for s, spin in enumerate(spins):
            for k in range(eigen.shape[1]):
                columns["spin"].append(spin)
                columns["k_index"].append(k)
                columns["distance_invA"].append(f"{distance[k]:.12g}")
                for i in range(3):
                    columns[f"k{i + 1}_frac"].append(f"{kfrac[k, i]:.12g}")
                columns["label"].append(labels[k] or "")
                columns["branch_start"].append(1 if k in breaks else 0)
                for n in range(eigen.shape[2]):
                    columns[f"band_{n + 1}_E-E_F_eV"].append(f"{eigen[s, k, n] - fermi:.10g}")
        gs = spec.ground_state
        header = (f"Band structure {run_id} of {gs.formula}, {gs.xc}, path {value.get('path')}, "
                  f"{spec.n_kpoints} k-points, Fermi level {fermi:.8f} eV (absolute GPAW), "
                  f"specification {spec.digest}\n"
                  f"{record.provenance.model}; energies in eV relative to the self-consistent "
                  "Fermi level; distance in 1/A with 2 pi; k in fractional reciprocal "
                  "coordinates; branch_start marks a jump with no distance")
        return {"path": dataio.write_csv(path, columns, header),
                "rows": len(columns["spin"]), "columns": list(columns)}

    def _dft_eos_build(self, variables: Optional[dict] = None, source: Optional[dict] = None):
        from ..experiments.dft import eos

        source = dict(source or {})
        kind = source.get("kind") or "structure"
        environment = self._dft_environment()
        if kind == "structure":
            structure = self._require_structure()
            spec = eos.build(structure, environment,
                             structure_key=self.project.key_of(structure),
                             **dict(variables or {}))
        else:
            spec = eos.build_from_run(self.project, kind, str(source.get("run_id") or ""),
                                      environment, **dict(variables or {}))
        return spec, eos.check(spec, environment)

    def dft_eos_spec(self, variables: Optional[dict] = None,
                     source: Optional[dict] = None) -> dict:
        """Build and check an equation-of-state specification without running anything."""
        from ..experiments.dft import eos
        from ..experiments.dft import spec as specs

        try:
            spec, report = self._dft_eos_build(variables, source)
        except specs.SpecError as exc:
            return {"ok": False, "error": str(exc), "blocking": [str(exc)],
                    "by_field": {}, "warnings": []}
        return _jsonable({"ok": report.ok, "spec": eos.describe(spec, report),
                          **report.as_dict()})

    def dft_eos_run(self, variables: Optional[dict] = None, source: Optional[dict] = None,
                    background: bool = True, timeout_s: Optional[float] = None) -> dict:
        """Compute the equation of state of the active structure or of a stored run.

        Every volume uses the reference's k-point grid and cutoff.  A refused
        specification starts nothing; a run that is cancelled, fails or whose
        fit does not verify stores nothing.  A complete one is stored in one
        step with one history line that is not an undo point.
        """
        from ..experiments.dft import eos, records
        from ..experiments.dft import spec as specs
        from ..experiments.dft.run import new_run_id

        try:
            spec, report = self._dft_eos_build(variables, source)
        except specs.SpecError as exc:
            self.warn(str(exc), level="error", source="solver/dft-eos")
            return {"ok": False, "status": "refused", "refused": True, "error": str(exc),
                    "blocking": [str(exc)], "by_field": {}}
        if not report.ok:
            reason = "; ".join(report.blocking)
            self.warn(reason, level="unsupported", source="solver/dft-eos")
            return {"ok": False, "status": "refused", "refused": True, "error": reason,
                    **report.as_dict()}
        owner = self.project
        run_id = new_run_id()

        def work(job) -> dict:
            def progress(fraction: float, message: str) -> None:
                job.progress = max(job.progress, min(0.99, float(fraction)))
                job.message = message

            try:
                outcome = eos.execute(spec, self._dft_environment(), run_id=run_id,
                                      progress=progress, cancelled=job.cancel_flag.is_set,
                                      timeout_s=timeout_s)
            except specs.SpecRefused as exc:
                self.warn(str(exc), level="unsupported", source="solver/dft-eos")
                return {"ok": False, "status": "refused", "refused": True,
                        "error": str(exc), "run_id": run_id, **exc.report.as_dict()}
            if not outcome.ok:
                self.warn(outcome.reason, level="warning" if outcome.status == "cancelled"
                          else "error", source="solver/dft-eos")
                return {"ok": False, "status": outcome.status, "reason": outcome.reason,
                        "run_id": run_id, "points": _jsonable(outcome.points),
                        "owner_replaced": owner is not self.project}
            try:
                records.store_eos(owner, outcome)
            except ValueError as exc:
                self.warn(str(exc), level="error", source="solver/dft-eos")
                return {"ok": False, "status": "failed", "reason": str(exc), "run_id": run_id}
            payload = self.dft_eos_result(run_id, project=owner)
            payload["owner_replaced"] = owner is not self.project
            if owner is self.project:
                self._autosave()
            return payload

        if not background:
            return work(_NullJob())
        job = self.jobs.submit("dft-eos", f"DFT equation of state, {spec.ground_state.formula}",
                               work, owner=owner)
        return {"job": job.as_dict(), "run_id": run_id, "spec_digest": spec.digest}

    def dft_eos_runs(self) -> List[dict]:
        from ..experiments.dft import records

        return [records.eos_summary(self.project, r) for r in records.eos_run_ids(self.project)]

    def dft_eos_result(self, run_id: Optional[str] = None,
                       project: Optional[Project] = None) -> dict:
        """One stored equation of state with its points, fit, checks, state and provenance."""
        from ..experiments.dft import eos, records

        project = project or self.project
        ids = records.eos_run_ids(project)
        if run_id is None:
            if not ids:
                return {"ok": False, "error": "No equation of state is stored."}
            run_id = ids[0]
        record = records.eos_record(project, run_id)
        spec = records.eos_spec_of(project, run_id)
        if record is None or spec is None:
            return {"ok": False, "error": f"No equation of state {run_id}."}
        info = records.eos_status(project, run_id)
        if info.get("state") == "corrupt":
            return {"ok": False, "status": "corrupt", "run_id": run_id,
                    "run_state": "corrupt", "error": info.get("reason")}
        value = dict(record.value or {})
        volumes = np.asarray(project.arrays[records.eos_key(run_id, "volumes")].data,
                             dtype=float)
        curve_v = np.linspace(volumes.min(), volumes.max(), 121)
        v0 = value["V0_A3"]
        b0 = value["B0_GPa"] / eos.EV_A3_TO_GPA
        free = value["free_energy_fit"]
        free_b0 = free["B0_GPa"] / eos.EV_A3_TO_GPA
        curve = {"volumes_A3": curve_v,
                 "energies_eV": eos.birch_murnaghan(curve_v, value["E0_eV"], v0, b0,
                                                    value["B1"]),
                 "pressures_GPa": eos.birch_murnaghan_pressure(curve_v, v0, b0, value["B1"])
                 * eos.EV_A3_TO_GPA,
                 "free_energies_eV": eos.birch_murnaghan(curve_v, free["E0_eV"], free["V0_A3"],
                                                         free_b0, free["B1"]),
                 "free_pressures_GPa": eos.birch_murnaghan_pressure(
                     curve_v, free["V0_A3"], free_b0, free["B1"]) * eos.EV_A3_TO_GPA}
        prov = record.provenance
        return _jsonable({
            "ok": True, "run_id": run_id, "status": value.get("status"),
            "run_state": info.get("state"), "current": bool(info.get("current")),
            "applicable": bool(info.get("applicable")), "state_reason": info.get("reason"),
            "source": dict(spec.source), "formula": spec.ground_state.formula,
            "xc": spec.ground_state.xc, "structure_key": spec.structure_key,
            "spec_digest": spec.digest, "settings": spec.settings(),
            "n_atoms": value.get("n_atoms"), "E0_eV": value.get("E0_eV"),
            "energy_definition": value.get("energy_definition"),
            "occupations": value.get("occupations"),
            "free_energy_fit": value.get("free_energy_fit"),
            "E0_eV_per_atom": value.get("E0_eV_per_atom"), "V0_A3": v0,
            "V0_A3_per_atom": value.get("V0_A3_per_atom"), "B0_GPa": value.get("B0_GPa"),
            "B1": value.get("B1"), "linear_scale": value.get("linear_scale"),
            "equilibrium_cell_lengths_A": value.get("equilibrium_cell_lengths_A"),
            "reference_volume_A3": value.get("reference_volume_A3"),
            "points": value.get("points"), "fit_curve": curve, "checks": value.get("checks"),
            "kpoints": value.get("kpoints"), "cutoff_eV": value.get("cutoff_eV"),
            "units": value.get("units"), "started_unix": value.get("started_unix"),
            "finished_unix": value.get("finished_unix"), "wall_time_s": value.get("wall_time_s"),
            "array_sha256": value.get("array_sha256"),
            "experimental_comparison": value.get("experimental_comparison"),
            "convergence": record.convergence.as_dict() if record.convergence else None,
            "warnings": list(record.extra.get("warnings", [])),
            "provenance": {
                "model": prov.model, "fidelity": prov.fidelity.value,
                "origin": prov.origin.value, "approximations": list(prov.approximations),
                "inputs_digest": prov.inputs_digest, "dataset": prov.dataset,
                "references": list(prov.references), "created_unix": prov.created_unix,
                "software_pinned": prov.parameters.get("software_pinned"),
                "software_reported": prov.parameters.get("software_reported"),
                "paw_datasets": prov.parameters.get("paw_datasets"),
                "gpaw_parameters": prov.parameters.get("gpaw_parameters"),
                "point_spec_digests": prov.parameters.get("point_spec_digests"),
                "point_run_ids": prov.parameters.get("point_run_ids"),
                "tolerances": dict(prov.tolerances),
            },
        })

    def dft_eos_export(self, path: str, run_id: Optional[str] = None) -> dict:
        """Write the points of a stored equation of state, with the fit, to CSV."""
        from ..experiments.dft import eos, records

        ids = records.eos_run_ids(self.project)
        run_id = run_id or (ids[0] if ids else None)
        spec = records.eos_spec_of(self.project, run_id or "")
        record = records.eos_record(self.project, run_id or "")
        if spec is None or record is None:
            raise ValueError(f"No equation of state {run_id} is stored.")
        damage = records.eos_integrity(self.project, run_id)
        if damage:
            raise ValueError(f"Equation of state {run_id} is not exported: {damage}")
        value = record.value
        b0 = value["B0_GPa"] / eos.EV_A3_TO_GPA
        free = value["free_energy_fit"]
        free_b0 = free["B0_GPa"] / eos.EV_A3_TO_GPA
        columns: Dict[str, List[Any]] = {k: [] for k in (
            "V_over_Vref", "volume_A3", "zero_width_energy_eV", "free_energy_eV",
            "fit_zero_width_energy_eV", "fit_pressure_0K_GPa", "free_energy_fit_pressure_GPa",
            "stress_pressure_GPa", "max_force_eV_A")}
        for row in value["points"]:
            fit_energy = eos.birch_murnaghan(row["volume_A3"], value["E0_eV"], value["V0_A3"],
                                             b0, value["B1"])
            fit_pressure = eos.birch_murnaghan_pressure(row["volume_A3"], value["V0_A3"], b0,
                                                        value["B1"]) * eos.EV_A3_TO_GPA
            free_pressure = eos.birch_murnaghan_pressure(
                row["volume_A3"], free["V0_A3"], free_b0, free["B1"]) * eos.EV_A3_TO_GPA
            columns["V_over_Vref"].append(f"{row['scale']:.12g}")
            columns["volume_A3"].append(f"{row['volume_A3']:.12g}")
            columns["zero_width_energy_eV"].append(f"{row['energy_eV']:.12g}")
            columns["free_energy_eV"].append(f"{row['free_energy_eV']:.12g}")
            columns["fit_zero_width_energy_eV"].append(f"{float(fit_energy):.12g}")
            columns["fit_pressure_0K_GPa"].append(f"{float(fit_pressure):.10g}")
            columns["free_energy_fit_pressure_GPa"].append(f"{float(free_pressure):.10g}")
            columns["stress_pressure_GPa"].append(
                "" if row["pressure_GPa"] is None else f"{row['pressure_GPa']:.10g}")
            columns["max_force_eV_A"].append(
                "" if row["max_force_eV_A"] is None else f"{row['max_force_eV_A']:.10g}")
        gs = spec.ground_state
        header = (f"Equation of state {run_id} of {gs.formula}, {gs.xc}, "
                  f"{'x'.join(map(str, gs.kpoints))} k-points and {gs.cutoff_eV:g} eV at every "
                  f"volume, specification {spec.digest}\n"
                  f"Birch-Murnaghan fit of the zero-width extrapolated energy (0 K static "
                  f"lattice): V0 {value['V0_A3']:.8g} A^3 per cell, B0 {value['B0_GPa']:.6g} GPa, "
                  f"B' {value['B1']:.6g}, E0 {value['E0_eV']:.10g} eV per cell\n"
                  f"Free energy E - TS at {gs.occupations} width {gs.smearing_eV:g} eV fitted "
                  f"separately: V0 {free['V0_A3']:.8g} A^3, B0 {free['B0_GPa']:.6g} GPa; GPAW's "
                  "stress is -dF/dV of that free energy")
        return {"path": dataio.write_csv(path, columns, header),
                "rows": len(value["points"]), "columns": list(columns)}

    def dft_eos_apply(self, run_id: Optional[str] = None) -> dict:
        """Scale the structure to a stored fitted equilibrium volume, as one undoable change."""
        from ..experiments.dft import records

        ids = records.eos_run_ids(self.project)
        run_id = run_id or (ids[0] if ids else None)
        try:
            message = records.apply_eos(self.project, run_id or "")
        except records.ApplyRefused as exc:
            self.warn(f"Not applied: {exc}", level="unsupported", source="solver/dft-eos")
            return {"ok": False, "error": f"Not applied: {exc}"}
        self._autosave()
        return {"ok": True, "message": message, "app_state": self.state()}

    def _dft_ldos_build(self, variables: Optional[dict] = None, source: Optional[dict] = None):
        from ..experiments.dft import ldos

        source = dict(source or {})
        kind = source.get("kind") or "structure"
        environment = self._dft_environment()
        if kind == "structure":
            structure = self._require_structure()
            spec = ldos.build(structure, environment,
                              structure_key=self.project.key_of(structure),
                              **dict(variables or {}))
        else:
            spec = ldos.build_from_run(self.project, kind, str(source.get("run_id") or ""),
                                       environment, **dict(variables or {}))
        return spec, ldos.check(spec, environment)

    def dft_ldos_spec(self, variables: Optional[dict] = None,
                      source: Optional[dict] = None) -> dict:
        """Build and check an LDOS specification without running anything."""
        from ..experiments.dft import ldos
        from ..experiments.dft import spec as specs

        try:
            spec, report = self._dft_ldos_build(variables, source)
        except specs.SpecError as exc:
            return {"ok": False, "error": str(exc), "blocking": [str(exc)],
                    "by_field": {}, "warnings": []}
        return _jsonable({"ok": report.ok, "spec": ldos.describe(spec, report),
                          **report.as_dict()})

    def dft_ldos_run(self, variables: Optional[dict] = None, source: Optional[dict] = None,
                     background: bool = True, timeout_s: Optional[float] = None) -> dict:
        """Compute the spatial LDOS of the active structure or of a stored run.

        A refused specification starts nothing; a run that is cancelled, times
        out, fails or does not verify stores nothing.  A complete one is stored
        in one step with one history line that is not an undo point.
        """
        from ..experiments.dft import ldos, records
        from ..experiments.dft import spec as specs
        from ..experiments.dft.run import new_run_id

        try:
            spec, report = self._dft_ldos_build(variables, source)
        except specs.SpecError as exc:
            self.warn(str(exc), level="error", source="solver/dft-ldos")
            return {"ok": False, "status": "refused", "refused": True, "error": str(exc),
                    "blocking": [str(exc)], "by_field": {}}
        if not report.ok:
            reason = "; ".join(report.blocking)
            self.warn(reason, level="unsupported", source="solver/dft-ldos")
            return {"ok": False, "status": "refused", "refused": True, "error": reason,
                    **report.as_dict()}
        owner = self.project
        run_id = new_run_id()

        def work(job) -> dict:
            def progress(event: dict) -> None:
                kind = event.get("event")
                if kind == "scf":
                    job.progress = min(0.6, 0.6 * int(event.get("iteration", 0)) / 40.0)
                    job.message = f"ground state, SCF iteration {int(event.get('iteration', 0))}"
                elif kind == "stage":
                    name = event.get("name", "")
                    job.progress = max(job.progress, {"nscf": 0.65, "ldos": 0.9}.get(name, 0.62))
                    job.message = {"nscf": "non-self-consistent step, symmetry off",
                                   "ldos": "accumulating the map"}.get(name, f"extracting {name}")

            try:
                outcome = ldos.execute(spec, self._dft_environment(), run_id=run_id,
                                       progress=progress, cancelled=job.cancel_flag.is_set,
                                       timeout_s=timeout_s)
            except specs.SpecRefused as exc:
                self.warn(str(exc), level="unsupported", source="solver/dft-ldos")
                return {"ok": False, "status": "refused", "refused": True,
                        "error": str(exc), "run_id": run_id, **exc.report.as_dict()}
            if not outcome.ok:
                self.warn(outcome.reason, level="warning" if outcome.status == "cancelled"
                          else "error", source="solver/dft-ldos")
                return {"ok": False, "status": outcome.status, "reason": outcome.reason,
                        "run_id": run_id, "audit": _jsonable(outcome.audit),
                        "owner_replaced": owner is not self.project}
            try:
                records.store_ldos(owner, outcome)
            except ValueError as exc:
                self.warn(str(exc), level="error", source="solver/dft-ldos")
                return {"ok": False, "status": "failed", "reason": str(exc), "run_id": run_id}
            payload = self.dft_ldos_result(run_id, project=owner)
            payload["owner_replaced"] = owner is not self.project
            if owner is self.project:
                self._autosave()
            return payload

        if not background:
            return work(_NullJob())
        job = self.jobs.submit("dft-ldos", f"DFT LDOS, {spec.ground_state.formula}", work,
                               owner=owner)
        return {"job": job.as_dict(), "run_id": run_id, "spec_digest": spec.digest}

    def dft_ldos_runs(self) -> List[dict]:
        from ..experiments.dft import records

        return [records.ldos_summary(self.project, r)
                for r in records.ldos_run_ids(self.project)]

    def _ldos_stored(self, run_id: Optional[str], project: Optional[Project] = None):
        from ..experiments.dft import records

        project = project or self.project
        ids = records.ldos_run_ids(project)
        run_id = run_id or (ids[0] if ids else None)
        record = records.ldos_record(project, run_id or "")
        spec = records.ldos_spec_of(project, run_id or "")
        if record is None or spec is None:
            raise ValueError(f"No LDOS {run_id} is stored." if run_id else
                             "No LDOS is stored.")
        damage = records.ldos_integrity(project, run_id)
        if damage:
            raise ValueError(f"LDOS {run_id} refused: {damage}")
        return project, run_id, record, spec

    def dft_ldos_result(self, run_id: Optional[str] = None,
                        project: Optional[Project] = None) -> dict:
        """One stored LDOS: summary, checks, state, a profile along c and provenance."""
        from ..experiments.dft import records

        project = project or self.project
        ids = records.ldos_run_ids(project)
        if run_id is None:
            if not ids:
                return {"ok": False, "error": "No LDOS is stored."}
            run_id = ids[0]
        record = records.ldos_record(project, run_id)
        spec = records.ldos_spec_of(project, run_id)
        if record is None or spec is None:
            return {"ok": False, "error": f"No LDOS {run_id}."}
        info = records.ldos_status(project, run_id)
        if info.get("state") == "corrupt":
            return {"ok": False, "status": "corrupt", "run_id": run_id,
                    "run_state": "corrupt", "error": info.get("reason")}
        value = dict(record.value or {})
        data = np.asarray(project.arrays[records.ldos_key(run_id, "ldos")].data, dtype=float)
        cell = np.asarray(value["cell_A"], dtype=float)
        profile = data.mean(axis=(0, 1))
        coordinate = np.arange(len(profile)) * float(np.linalg.norm(cell[2])) / len(profile)
        prov = record.provenance
        return _jsonable({
            "ok": True, "run_id": run_id, "status": value.get("status"),
            "run_state": info.get("state"), "current": bool(info.get("current")),
            "state_reason": info.get("reason"), "source": dict(spec.source),
            "formula": spec.ground_state.formula, "xc": spec.ground_state.xc,
            "boundary": spec.ground_state.boundary, "structure_key": spec.structure_key,
            "spec_digest": spec.digest, "settings": spec.settings(),
            "fermi_level_eV": value.get("fermi_level_eV"), "window_eV": value.get("window_eV"),
            "window_absolute_eV": value.get("window_absolute_eV"),
            "states_in_window": value.get("states_in_window"),
            "map_integral_states": value.get("map_integral_states"),
            "pseudo_norm_ratio": value.get("pseudo_norm_ratio"),
            "augmentation_radii_A": value.get("augmentation_radii_A"),
            "max_augmentation_radius_A": value.get("max_augmentation_radius_A"),
            "grid_shape": value.get("grid_shape"), "cell_A": value.get("cell_A"),
            "n_kpoints": value.get("n_kpoints"), "n_bands": value.get("n_bands"),
            "n_spins": value.get("n_spins"), "spin_channels": value.get("spin_channels"),
            "max_ldos": value.get("max_ldos"), "stm": value.get("stm"),
            "profile": {"c_A": coordinate, "planar_average": profile},
            "highest_band_above_EF_eV": value.get("highest_band_above_EF_eV"),
            "array_sha256": value.get("array_sha256"),
            "started_unix": value.get("started_unix"),
            "finished_unix": value.get("finished_unix"),
            "wall_time_s": value.get("wall_time_s"), "units": value.get("units"),
            "experimental_comparison": value.get("experimental_comparison"),
            "convergence": record.convergence.as_dict() if record.convergence else None,
            "warnings": list(record.extra.get("warnings", [])),
            "provenance": {
                "model": prov.model, "fidelity": prov.fidelity.value,
                "origin": prov.origin.value, "approximations": list(prov.approximations),
                "inputs_digest": prov.inputs_digest, "dataset": prov.dataset,
                "references": list(prov.references), "created_unix": prov.created_unix,
                "software_pinned": prov.parameters.get("software_pinned"),
                "software_reported": prov.parameters.get("software_reported"),
                "paw_datasets": prov.parameters.get("paw_datasets"),
                "gpaw_parameters": prov.parameters.get("gpaw_parameters"),
                "gpaw_parameters_used": prov.parameters.get("gpaw_parameters_used"),
                "nscf_parameters": prov.parameters.get("nscf_parameters"),
                "nscf_parameters_used": prov.parameters.get("nscf_parameters_used"),
                "ldos_settings": prov.parameters.get("ldos_settings"),
                "ldos_used": prov.parameters.get("ldos_used"),
            },
        })

    def dft_ldos_slice(self, run_id: Optional[str] = None, index: Optional[int] = None) -> dict:
        """One plane of the stored map perpendicular to c, for display."""
        try:
            project, run_id, record, spec = self._ldos_stored(run_id)
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        from ..experiments.dft import records

        data = np.asarray(project.arrays[records.ldos_key(run_id, "ldos")].data, dtype=float)
        nz = data.shape[2]
        cell = np.asarray(record.value["cell_A"], dtype=float)
        if index is None:
            index = int(np.argmax(data.mean(axis=(0, 1))))
        index = int(index)
        if not 0 <= index < nz:
            return {"ok": False, "error": f"The plane index must be 0 to {nz - 1}."}
        plane = data[:, :, index]
        return _jsonable({"ok": True, "run_id": run_id, "index": index, "n_planes": nz,
                          "c_A": index * float(np.linalg.norm(cell[2])) / nz,
                          "shape": list(plane.shape), "values": plane.T.ravel(),
                          "min": float(plane.min()), "max": float(plane.max()),
                          "unit": "states/A^3", "lut": _silver_lut()})

    def dft_ldos_image(self, run_id: Optional[str] = None, mode: str = "constant-height",
                       height_A: Optional[float] = None,
                       isovalue: Optional[float] = None) -> dict:
        """A Tersoff-Hamann image from a stored LDOS, or the reason it cannot be taken."""
        from ..experiments.dft import ldos, records

        try:
            project, run_id, record, spec = self._ldos_stored(run_id)
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        supported, reason = ldos.stm_supported(spec.ground_state)
        if not supported:
            return {"ok": False, "error": reason}
        data = np.asarray(project.arrays[records.ldos_key(run_id, "ldos")].data, dtype=float)
        try:
            image = ldos.stm_image(data, np.asarray(record.value["cell_A"], dtype=float),
                                   np.asarray(spec.ground_state.positions_A, dtype=float),
                                   mode, height_A=height_A, isovalue=isovalue,
                                   augmentation_radius_A=float(
                                       record.value["max_augmentation_radius_A"]))
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        values = np.asarray(image["values"], dtype=float)
        return _jsonable({"ok": True, "run_id": run_id, "mode": mode,
                          "height_A": image.get("height_A"), "isovalue": image.get("isovalue"),
                          "shape": list(values.shape), "values": values.T.ravel(),
                          "min": float(values.min()), "max": float(values.max()),
                          "unit": image["unit"], "valid_heights_A": image["valid_heights_A"],
                          "surface_z_A": image["surface_z_A"],
                          "cell_ab_A": np.asarray(record.value["cell_A"])[:2, :2],
                          "window_eV": record.value["window_eV"],
                          "lut": _silver_lut()})

    def dft_ldos_export(self, path: str, run_id: Optional[str] = None) -> dict:
        """Write the stored map as a Gaussian Cube file, with its window and specification."""
        from ..experiments.dft import records
        from ..experiments.dft.convergence import structure_of

        project, run_id, record, spec = self._ldos_stored(run_id)
        data = np.asarray(project.arrays[records.ldos_key(run_id, "ldos")].data, dtype=float)
        low, high = record.value["window_eV"]
        comment = (f"LDOS {run_id} {spec.ground_state.formula} {low:g} to {high:g} eV about "
                   f"E_F, states/A^3 both spins, specification {spec.digest[:16]}")
        written = dataio.write_cube(path, structure_of(spec.ground_state), data,
                                    comment=comment)
        return {"path": written, "shape": list(data.shape), "unit": "states/A^3"}

    def dft_ldos_image_export(self, path: str, run_id: Optional[str] = None,
                              mode: str = "constant-height", height_A: Optional[float] = None,
                              isovalue: Optional[float] = None) -> dict:
        """Write a Tersoff-Hamann image to CSV: x, y and the value at every grid column."""
        image = self.dft_ldos_image(run_id, mode, height_A, isovalue)
        if not image.get("ok"):
            raise ValueError(image.get("error"))
        nx, ny = image["shape"]
        values = np.asarray(image["values"], dtype=float).reshape(ny, nx).T
        cell = np.asarray(image["cell_ab_A"], dtype=float)
        columns: Dict[str, List[Any]] = {"i": [], "j": [], "x_A": [], "y_A": [],
                                         "value": []}
        for i in range(nx):
            for j in range(ny):
                x, y = np.array([i / nx, j / ny]) @ cell
                columns["i"].append(i)
                columns["j"].append(j)
                columns["x_A"].append(f"{x:.10g}")
                columns["y_A"].append(f"{y:.10g}")
                columns["value"].append(f"{values[i, j]:.12g}")
        what = (f"constant height {image['height_A']:g} A above the topmost atom, LDOS in "
                "states/A^3" if mode == "constant-height" else
                f"constant LDOS {image['isovalue']:g} states/A^3, height in A above the "
                "topmost atom")
        low, high = image["window_eV"]
        header = (f"Tersoff-Hamann image of LDOS {image['run_id']}, {low:g} to {high:g} eV about "
                  f"E_F, {what}\nThe value is proportional to the tunnelling current; no "
                  "conversion to amperes is made")
        return {"path": dataio.write_csv(path, columns, header), "rows": nx * ny,
                "columns": list(columns)}

    def dft_runs(self) -> List[dict]:
        from ..experiments.dft import records

        return [records.summary(self.project, r) for r in records.run_ids(self.project)]

    def dft_result(self, run_id: Optional[str] = None,
                   project: Optional[Project] = None) -> dict:
        """Everything the interface shows about one run."""
        from ..experiments.dft import convergence, records

        project = project or self.project
        ids = records.run_ids(project)
        if run_id is None:
            if not ids:
                return {"ok": False, "error": "No DFT ground-state run is stored."}
            run_id = ids[0]
        record = records.record(project, run_id)
        if record is None:
            return {"ok": False, "error": f"No DFT run {run_id}."}
        spec = records.spec_of(project, run_id)
        state = records.status(project, run_id)
        prov = record.provenance
        get = lambda name: project.results.get(records.key(run_id, name))
        payload: Dict[str, Any] = {
            "ok": bool(record.supported), "run_id": run_id,
            "status": record.extra.get("status"), "state": state.get("state"),
            "current": bool(state.get("current")), "state_reason": state.get("reason"),
            "formula": spec.formula if spec else None,
            "boundary": spec.boundary if spec else None,
            "structure_key": spec.structure_key if spec else None,
            "spec_digest": spec.digest if spec else None,
            "classification": record.extra.get("classification"),
            "warnings": list(record.extra.get("warnings", [])),
            "software_warnings": list(record.extra.get("software_warnings", [])),
            "convergence": record.convergence.as_dict() if record.convergence else None,
            "iterations": record.extra.get("iterations"),
            "wall_time_s": record.extra.get("wall_time_s"),
            "study_id": record.extra.get("study_id"),
            "derived_from": record.extra.get("derived_from"),
            "provenance": {
                "model": prov.model, "fidelity": prov.fidelity.value,
                "origin": prov.origin.value, "approximations": list(prov.approximations),
                "boundary_conditions": prov.boundary_conditions,
                "tolerances": dict(prov.tolerances), "inputs_digest": prov.inputs_digest,
                "dataset": prov.dataset, "dataset_license": prov.dataset_license,
                "references": list(prov.references), "notes": prov.notes,
                "software_pinned": prov.parameters.get("software_pinned"),
                "software_reported": prov.parameters.get("software_reported"),
                "paw_datasets": prov.parameters.get("paw_datasets"),
                "gpaw_parameters": prov.parameters.get("gpaw_parameters"),
                "grid": prov.parameters.get("grid"),
                "n_bands": prov.parameters.get("n_bands"),
                "n_ibz_kpoints": prov.parameters.get("n_ibz_kpoints"),
                "symmetry_operations": prov.parameters.get("symmetry_operations"),
                "restart": prov.parameters.get("restart"),
                "units": prov.parameters.get("units"),
            },
        }
        if spec is not None:
            from ..experiments.dft import spec as specs

            payload["spec"] = {"settings": spec.settings(), "digest": spec.digest,
                               "gpaw_parameters": specs.gpaw_parameters(spec),
                               "mass_numbers": list(spec.mass_numbers),
                               "n_atoms": spec.n_atoms}
        if not record.supported:
            payload["reason"] = record.unsupported_reason
            payload["scf_history"] = _jsonable(record.extra.get("scf_history", []))
            return payload
        energy = get("energy")
        payload.update({
            "energy_eV": energy.value,
            "extrapolated_energy_eV": energy.extra.get("extrapolated_energy_eV"),
            "energy_per_atom_eV": energy.extra.get("energy_per_atom_eV"),
            "energy_contributions_eV": energy.extra.get("contributions_eV"),
            "reference_energy_eV": energy.extra.get("reference_energy_eV"),
        })
        forces = get("forces")
        if forces is not None:
            payload["max_force_eV_A"] = forces.extra.get("max_force_eV_A")
            payload["net_force_eV_A"] = forces.extra.get("net_force_eV_A")
            payload["forces_by_atom_id"] = forces.extra.get("by_atom_id")
        stress = get("stress")
        if stress is not None:
            payload["stress_eV_A3"] = _jsonable(stress.value)
            payload["pressure_GPa"] = stress.extra.get("pressure_GPa")
        fermi = get("fermi_level")
        if fermi is not None:
            payload["fermi_level_eV"] = fermi.value
            payload["homo_eV"] = fermi.extra.get("homo_eV")
            payload["lumo_eV"] = fermi.extra.get("lumo_eV")
        moment = get("magnetic_moment")
        if moment is not None:
            payload["magnetic_moment_muB"] = moment.value
            payload["local_moments_muB"] = moment.extra.get("local_moments_by_atom_id")
        eigen = get("eigenvalues")
        if eigen is not None:
            payload["band_edges"] = eigen.extra.get("band_edges")
        payload["charge_accounting"] = _jsonable(get("charge_accounting").value)
        payload["scf_history"] = _jsonable(get("scf_history").value)
        arrays = {}
        for name in ("density", "spin_density", "electrostatic_potential", "eigenvalues",
                     "occupations"):
            stored = project.arrays.get(records.key(run_id, name))
            if stored is not None:
                arrays[name] = stored.summary()
        payload["arrays"] = _jsonable(arrays)
        if spec is not None:
            studies = [v for k, v in project.results.items()
                       if k.startswith(records.STUDY_PREFIX)]
            payload["evidence"] = convergence.evidence(spec, studies)
        return payload

    def dft_array(self, run_id: Optional[str] = None, name: str = "density",
                  axis: int = 2, reduce: str = "planar-average") -> dict:
        """A one-dimensional view of a stored grid: the average over two axes."""
        from ..experiments.dft import records

        ids = records.run_ids(self.project)
        run_id = run_id or (ids[0] if ids else None)
        stored = self.project.arrays.get(records.key(run_id or "", name))
        if stored is None:
            return {"ok": False, "error": f"No {name} array stored for run {run_id}."}
        if stored.kind != "volumetric":
            return {"ok": False, "error": f"{name} is not a volumetric grid."}
        axis = int(axis)
        if axis not in (0, 1, 2) or reduce != "planar-average":
            return {"ok": False, "error": "axis must be 0, 1 or 2 and reduce "
                                          "'planar-average'."}
        data = np.asarray(stored.data)
        others = tuple(i for i in range(3) if i != axis)
        profile = data.mean(axis=others)
        cell = np.asarray(stored.meta.get("cell_A"), dtype=float)
        length = float(np.linalg.norm(cell[axis]))
        coordinate = np.arange(len(profile)) * length / len(profile)
        return {"ok": True, "run_id": run_id, "name": name, "axis": "abc"[axis],
                "unit": stored.unit, "coordinate_A": coordinate.tolist(),
                "values": profile.tolist(), "shape": stored.shape,
                "description": stored.description}

    def dft_study(self, parameter: str, values: Sequence[Any],
                  observable: str = "energy_per_atom_eV", tolerance: float = 1.0e-3,
                  variables: Optional[dict] = None, background: bool = True) -> dict:
        """Start a convergence study on the active structure."""
        from ..experiments.dft import convergence
        from ..experiments.dft import spec as specs

        try:
            spec, report = self._dft_build(variables)
        except specs.SpecError as exc:
            return {"ok": False, "error": str(exc)}
        if not report.ok:
            return {"ok": False, "error": "; ".join(report.blocking), **report.as_dict()}
        try:
            study = convergence.StudySpec(base=spec, parameter=str(parameter),
                                          values=tuple(values), observable=str(observable),
                                          tolerance=float(tolerance))
            plan = convergence.plan(study, self._dft_environment())
        except (convergence.StudyError, specs.SpecError, ValueError, TypeError) as exc:
            self.warn(str(exc), level="unsupported", source="solver/dft")
            return {"ok": False, "error": str(exc)}
        owner = self.project
        study_id = f"{int(time.time() * 1000):013d}-{uuid.uuid4().hex[:6]}"

        def work(job) -> dict:
            from ..experiments.dft import records

            def progress(fraction: float, message: str) -> None:
                job.progress = min(0.99, fraction)
                job.message = message

            def keep(run, point) -> None:
                records.store(owner, run)

            outcome = convergence.run_study(
                study, self._dft_environment(), progress=progress,
                cancelled=job.cancel_flag.is_set, on_run=keep, study_id=study_id)
            owner.add_result(f"{records.STUDY_PREFIX}{outcome.study_id}", outcome.record())
            if owner is self.project:
                self._autosave()
            result = self.dft_study_result(outcome.study_id, project=owner)
            result["owner_replaced"] = owner is not self.project
            return result

        if not background:
            return work(_NullJob())
        job = self.jobs.submit("dft-study", f"Convergence of {parameter}, {len(plan)} runs",
                               work, owner=owner)
        return {"job": job.as_dict(), "study_id": study_id, "points": len(plan)}

    def dft_studies(self) -> List[dict]:
        from ..experiments.dft import records

        out = []
        for key, item in sorted(self.project.results.items(), reverse=True):
            if key.startswith(records.STUDY_PREFIX):
                value = item.value or {}
                out.append({"study_id": item.extra.get("study_id"),
                            "parameter": value.get("parameter"),
                            "observable": value.get("observable"),
                            "tolerance_met": value.get("tolerance_met"),
                            "residual": value.get("residual"),
                            "status": item.extra.get("status")})
        return out

    def dft_study_result(self, study_id: Optional[str] = None,
                         project: Optional[Project] = None) -> dict:
        from ..experiments.dft import records

        project = project or self.project
        keys = sorted((k for k in project.results if k.startswith(records.STUDY_PREFIX)),
                      reverse=True)
        if study_id is None:
            if not keys:
                return {"ok": False, "error": "No convergence study is stored."}
            key = keys[0]
        else:
            key = f"{records.STUDY_PREFIX}{study_id}"
        item = project.results.get(key)
        if item is None:
            return {"ok": False, "error": f"No convergence study {study_id}."}
        return {"ok": True, "study_id": item.extra.get("study_id"),
                "status": item.extra.get("status"), **_jsonable(item.value),
                "note": item.provenance.notes,
                "restart_policy": item.provenance.parameters.get("restart_policy"),
                "parameter_info": item.provenance.parameters.get("parameter_info"),
                "wall_time_s": item.extra.get("wall_time_s")}

    def claims_list(self) -> dict:
        from ..provenance.classification import describe as describe_classes

        return {"claims": [c.as_dict() for c in self.project.claims.values()],
                "classifications": describe_classes()}

    def electrostatics_status(self, settings: Optional[dict] = None) -> dict:
        """Charge model, geometry and readiness of the active structure."""
        s = self.structure
        if s is None:
            return {"structure": False, "ready": False,
                    "blocking": ["No active structure."], "runs": []}
        try:
            out = self.lab.electrostatics.status(s, settings or {})
        except ApiError as exc:
            return {"structure": True, "ready": False, "blocking": [str(exc)], "runs": []}
        out["structure"] = True
        out["runs"] = self.lab.electrostatics.runs()[:12]
        return out

    def electrostatics_assign(self, kind: str, by_element: Optional[dict] = None,
                              by_atom_id: Optional[dict] = None,
                              source: str = "") -> dict:
        """Record a point-charge model on the active structure, undoably."""
        s = self._require_structure()
        try:
            if kind == "formal-point-ion":
                info = self.lab.electrostatics.assign(s, formal=True, source=source)
            elif kind == "per-element":
                info = self.lab.electrostatics.assign(
                    s, by_element={str(k): float(v) for k, v in (by_element or {}).items()},
                    source=source)
            elif kind == "per-atom":
                info = self.lab.electrostatics.assign(
                    s, by_atom_id={int(k): float(v) for k, v in (by_atom_id or {}).items()},
                    source=source)
            else:
                raise ApiError(f"Unknown charge model {kind!r}. Available: per-element, "
                               "per-atom, formal-point-ion.")
        except (ApiError, ValueError, TypeError) as exc:
            self.warn(str(exc), "error", "electrostatics")
            return {"ok": False, "error": str(exc)}
        self._autosave()
        return {"ok": True, **_jsonable(info), "state": self.state()}

    def electrostatics_clear(self) -> dict:
        s = self._require_structure()
        removed = self.lab.electrostatics.clear(s)
        self._autosave()
        return {"ok": True, "removed": removed, "state": self.state()}

    def electrostatics_run(self, settings: Optional[dict] = None,
                           background: bool = True) -> dict:
        """Compute point-charge electrostatics for the active structure.

        The structure and settings are frozen here, before the job is queued,
        so edits made while it runs change only whether the answer still
        describes what is on screen, never what was computed.
        """
        s = self._require_structure()
        try:
            status = self.lab.electrostatics.status(s, settings or {})
        except ApiError as exc:
            self.warn(str(exc), "error", "electrostatics")
            return {"ok": False, "error": str(exc)}
        if not status["ready"]:
            reason = "; ".join(status["blocking"])
            self.warn(reason, "unsupported", "electrostatics")
            return {"ok": False, "error": reason, "status": status}
        owner = self.project
        snapshot = s.copy()
        structure_key = owner.key_of(s)
        frozen = dict(status["settings"])
        run_id = f"{int(time.time() * 1000):013d}-{uuid.uuid4().hex[:6]}"

        def work(job):
            return self._run_electrostatics(owner, snapshot, structure_key, frozen,
                                            run_id, job)

        if not background:
            return work(_NullJob())
        job = self.jobs.submit("electrostatics",
                               f"Electrostatics, {len(snapshot)} atoms", work, owner=owner)
        return {"job": job.as_dict(), "run_id": run_id}

    def _run_electrostatics(self, owner: Project, snapshot: Structure,
                            structure_key: Optional[str], settings: dict,
                            run_id: str, job) -> dict:
        def progress(fraction: float, message: str) -> bool:
            job.progress = min(0.99, float(fraction))
            job.message = message
            return not job.cancel_flag.is_set()

        self.lab.electrostatics.compute(
            snapshot, settings=settings, progress=progress,
            cancelled=job.cancel_flag.is_set, run_id=run_id, project=owner,
            structure_key=structure_key)
        payload = self.electrostatics_result(run_id, project=owner)
        energy = owner.results.get(f"electrostatics::{run_id}::energy")
        if energy is not None and not energy.supported:
            level = "warning" if energy.extra.get("status") == "cancelled" else "unsupported"
            self.warn(energy.unsupported_reason, level, "electrostatics")
        if owner is self.project:
            self._autosave()
            payload["state"] = self.state()
        else:
            payload["owner_replaced"] = True
            payload["state"] = None
        return payload

    def electrostatics_result(self, run_id: Optional[str] = None,
                              project: Optional[Project] = None) -> dict:
        """A stored electrostatics run, summarised for display."""
        from ..physics.electrostatics import EwaldSettings
        from ..solvers.electrostatics import structure_digest

        project = project or self.project
        prefix = "electrostatics::"
        if run_id is None:
            runs = sorted({k.split("::")[1] for k in project.results
                           if k.startswith(prefix) and k.count("::") == 2}, reverse=True)
            if not runs:
                return {"ok": False, "error": "No electrostatics run is stored."}
            run_id = runs[0]
        energy = project.results.get(f"{prefix}{run_id}::energy")
        if energy is None:
            return {"ok": False, "error": f"No electrostatics run {run_id}."}
        payload = {"ok": bool(energy.supported), "run_id": run_id,
                   "status": energy.extra.get("status", ""),
                   "structure_key": energy.extra.get("structure_key")}
        if not energy.supported:
            payload["reason"] = energy.unsupported_reason
            return payload
        prov = energy.provenance
        parameters = dict(prov.parameters)
        payload.update({
            "energy_eV": energy.value,
            "energy_per_atom_eV": energy.extra.get("energy_per_atom_eV"),
            "uncertainty_eV": energy.uncertainty,
            "components_eV": energy.extra.get("components_eV", {}),
            "charge_accounting": energy.extra.get("charge_accounting", {}),
            "geometry": energy.extra.get("geometry", {}),
            "check": energy.extra.get("check", {}),
            "convergence": energy.convergence.as_dict() if energy.convergence else None,
            "origin": prov.origin.value,
            "fidelity": prov.fidelity.value,
            "model": prov.model,
            "boundary_conditions": prov.boundary_conditions,
            "approximations": list(prov.approximations),
            "references": list(prov.references),
            "tolerances": dict(prov.tolerances),
            "inputs_digest": prov.inputs_digest,
            "parameters": {k: parameters.get(k) for k in (
                "method", "alpha_per_A", "real_cutoff_A", "kspace_cutoff_per_A",
                "alpha_choice", "neglected_real_factor", "neglected_reciprocal_factor",
                "real_space_pairs", "n_kvectors_half_space", "pairs",
                "slab_thickness_A", "internal_vacuum_gap_A", "internal_cell_height_A",
                "slab_dipole_e_A", "lateral_image_decay_factor", "cell_volume_A3")
                if parameters.get(k) is not None},
            "charge_model": parameters.get("charge_model"),
            "settings": parameters.get("settings"),
            "log": energy.extra.get("log", []),
        })
        forces = project.results.get(f"{prefix}{run_id}::forces")
        field = project.results.get(f"{prefix}{run_id}::site_field")
        potential = project.results.get(f"{prefix}{run_id}::site_potential")
        charges = project.results.get(f"{prefix}{run_id}::point_charges")
        if forces is not None:
            payload["max_force_eV_A"] = forces.extra.get("max_force_eV_A")
            payload["net_force_eV_A"] = forces.extra.get("net_force_eV_A")
        if field is not None:
            payload["max_field_V_A"] = field.extra.get("max_field_V_A")
        ids = energy.extra.get("atom_ids") or []
        stored = (potential is not None and potential.value is not None
                  and charges is not None and charges.value is not None)
        payload["per_atom_stored"] = bool(stored)
        if stored:
            s_for = project.structures.get(payload["structure_key"] or "")
            symbols = {}
            if s_for is not None:
                symbols = {int(i): pt.symbol(int(z)) for i, z in zip(s_for.ids, s_for.numbers)}
            phi = np.asarray(potential.value, dtype=float)
            q = np.asarray(charges.value, dtype=float)
            groups: Dict[str, List[int]] = {}
            for k, atom_id in enumerate(ids):
                label = symbols.get(int(atom_id), "?")
                groups.setdefault(f"{label} {q[k]:+.4g} e", []).append(k)
            payload["site_summary"] = [
                {"group": g, "count": len(ks),
                 "potential_mean_V": float(phi[ks].mean()),
                 "potential_min_V": float(phi[ks].min()),
                 "potential_max_V": float(phi[ks].max())}
                for g, ks in sorted(groups.items())]
        current = False
        s_now = project.structures.get(payload["structure_key"] or "")
        if s_now is not None and parameters.get("settings"):
            try:
                settings = EwaldSettings.from_dict(parameters["settings"])
                current = structure_digest(s_now, settings) == prov.inputs_digest
            except ValueError:
                current = False
        payload["current"] = bool(current)
        return payload

    def _electrostatics_for_atom(self, s: Structure, atom_id: int) -> Optional[dict]:
        """Electrostatic quantities at one atom from the newest run that is current."""
        from ..physics.electrostatics import EwaldSettings
        from ..solvers.electrostatics import structure_digest

        key = self.project.key_of(s)
        if key is None:
            return None
        runs = sorted({k.split("::")[1] for k, r in self.project.results.items()
                       if k.startswith("electrostatics::") and k.endswith("::energy")
                       and r.supported and r.extra.get("structure_key") == key},
                      reverse=True)
        digests: Dict[str, Optional[str]] = {}
        for run_id in runs:
            energy = self.project.results[f"electrostatics::{run_id}::energy"]
            try:
                settings = EwaldSettings.from_dict(energy.provenance.parameters.get("settings"))
            except ValueError:
                continue
            key_settings = repr(sorted(settings.as_dict().items()))
            if key_settings not in digests:
                digests[key_settings] = structure_digest(s, settings)
            if digests[key_settings] != energy.provenance.inputs_digest:
                continue
            ids = [int(i) for i in energy.extra.get("atom_ids", [])]
            if int(atom_id) not in ids:
                return None
            k = ids.index(int(atom_id))
            out = {"run_id": run_id, "origin": energy.provenance.origin.value,
                   "model": energy.provenance.model,
                   "boundary_conditions": energy.provenance.boundary_conditions,
                   "charge_model": energy.provenance.parameters.get("charge_model", {})
                   .get("kind")}
            for name, unit in (("point_charges", "e"), ("site_potential", "V"),
                               ("site_field", "V/A"), ("forces", "eV/A")):
                result = self.project.results.get(f"electrostatics::{run_id}::{name}")
                if result is None or result.value is None:
                    out[name] = None
                    continue
                value = np.asarray(result.value, dtype=float)[k]
                out[name] = value.tolist() if value.ndim else float(value)
            out["units"] = {"point_charges": "e", "site_potential": "V",
                            "site_field": "V/A", "forces": "eV/A"}
            return out
        return None

    def eam_status(self) -> dict:
        """Shipped EAM potentials, user files, and what applies to the active structure."""
        from ..physics import eam as eam_module

        out = {"potentials": _jsonable(self.lab.eam.catalog()),
               "user_directory": str(eam_module.user_directory()),
               "user_files": self.lab.eam.user_files(),
               "tasks": {k: dict(v) for k, v in EAM_SETTINGS.items()},
               "structure": None, "runs": []}
        s = self.structure
        if s is None:
            return out
        present = sorted({pt.symbol(int(z)) for z in s.numbers})
        chosen, reason = eam_module.select_shipped(present)
        blocking = ""
        if chosen is not None:
            ok, why = eam_module.load_shipped(chosen).supports(s)
            blocking = "" if ok else why
        out["structure"] = {"elements": present, "n_atoms": len(s),
                            "periodicity": list(s.cell.pbc),
                            "selected": chosen, "reason": reason, "blocking": blocking}
        out["runs"] = self.lab.eam.runs()[:12]
        return out

    def eam_run(self, task: str = "energy", potential: Optional[str] = None,
                settings: Optional[dict] = None, background: bool = True) -> dict:
        """Run an EAM energy, relaxation or dynamics job on the active structure.

        The structure is copied and fingerprinted here.  The job works on the
        copy; its result is applied to the live structure, as one undoable
        change, only if that structure is still exactly what was submitted.
        """
        from ..python_api.api import structure_state_digest

        s = self._require_structure()
        potential = potential or None
        try:
            parsed = self.lab.eam.settings(task, settings or {})
            chosen = self.lab.eam.potential(potential, s)
        except ApiError as exc:
            self.warn(str(exc), "unsupported", "solver/eam")
            return {"ok": False, "error": str(exc)}
        ok, why = chosen.supports(s)
        if not ok:
            self.warn(why, "unsupported", "solver/eam")
            return {"ok": False, "error": why}
        owner = self.project
        snapshot = s.copy()
        structure_key = owner.key_of(s)
        digest = structure_state_digest(s)
        run_id = f"{int(time.time() * 1000):013d}-{uuid.uuid4().hex[:6]}"
        potential_ref = chosen.identity.id if chosen.identity.shipped else chosen.identity.path

        def work(job):
            def progress(fraction: float, message: str) -> None:
                job.progress = float(fraction)
                job.message = message

            self.lab.eam.run(snapshot, task, potential=potential_ref, settings=parsed,
                             progress=progress, cancelled=job.cancel_flag.is_set,
                             run_id=run_id, project=owner, structure_key=structure_key,
                             apply=True, input_digest=digest)
            payload = self.eam_result(run_id, project=owner)
            energy = owner.results.get(f"eam::{run_id}::energy")
            if energy is not None and not energy.supported:
                self.warn(energy.unsupported_reason,
                          "warning" if energy.extra.get("status") == "cancelled"
                          else "unsupported", "solver/eam")
            if owner is self.project:
                self._autosave()
                payload["state"] = self.state()
            else:
                payload["owner_replaced"] = True
                payload["state"] = None
            return payload

        if not background:
            return work(_NullJob())
        job = self.jobs.submit("eam", f"EAM {task}, {chosen.name}, {len(snapshot)} atoms",
                               work, owner=owner)
        return {"job": job.as_dict(), "run_id": run_id}

    def eam_result(self, run_id: Optional[str] = None,
                   project: Optional[Project] = None) -> dict:
        """A stored EAM run, summarised for display, with its staleness."""
        from ..physics import eam as eam_module
        from ..python_api.api import structure_state_digest

        project = project or self.project
        runs = sorted({k.split("::")[1] for k in project.results
                       if k.startswith("eam::") and k.count("::") == 2}, reverse=True)
        if run_id is None:
            if not runs:
                return {"ok": False, "error": "No EAM run is stored."}
            run_id = runs[0]
        energy = project.results.get(f"eam::{run_id}::energy")
        if energy is None:
            return {"ok": False, "error": f"No EAM run {run_id}."}
        payload = {"ok": bool(energy.supported), "run_id": run_id,
                   "task": energy.extra.get("task"), "status": energy.extra.get("status", ""),
                   "model": energy.provenance.model}
        if not energy.supported:
            payload["reason"] = energy.unsupported_reason
            return payload
        prov = energy.provenance
        identity = dict(prov.parameters.get("potential", {}))
        forces = project.results.get(f"eam::{run_id}::forces")
        key = energy.extra.get("structure_key")
        live = project.structures.get(key or "")
        current = (live is not None and structure_state_digest(live)
                   == energy.extra.get("output_state_digest"))
        matches = None
        if identity.get("shipped"):
            try:
                matches = eam_module.load_shipped(identity["id"]).identity.sha256 \
                    == identity.get("sha256")
            except Exception:
                matches = False
        payload.update({
            "origin": prov.origin.value, "fidelity": prov.fidelity.value,
            "energy_eV": energy.value,
            "energy_per_atom_eV": energy.value / max(1, energy.extra.get("n_atoms", 1)),
            "n_atoms": energy.extra.get("n_atoms"),
            "convergence": energy.convergence.as_dict() if energy.convergence else None,
            "applied": energy.extra.get("applied"),
            "apply_reason": energy.extra.get("apply_reason"),
            "current": bool(current), "structure_key": key,
            "wall_time_s": energy.extra.get("wall_time_s"),
            "rho_extrapolated": energy.extra.get("rho_extrapolated", 0),
            "neighbour_list_builds": energy.extra.get("neighbour_list_builds"),
            "potential": {k: identity.get(k) for k in (
                "id", "file", "sha256", "family", "elements", "license", "license_text",
                "source_url", "openkim_id", "openkim_doi", "citations", "shipped",
                "notes", "retrieved")},
            "potential_file_matches": matches,
            "cutoff_A": prov.parameters.get("file_header", {}).get("cutoff_A"),
            "cutoff_residuals": prov.parameters.get("cutoff_residuals"),
            "task_settings": prov.parameters.get("task_settings"),
            "approximations": list(prov.approximations),
            "boundary_conditions": prov.boundary_conditions,
            "initial_energy_eV": energy.extra.get("initial_energy_eV"),
            "energy_change_eV": energy.extra.get("energy_change_eV"),
            "max_force_eV_A": (float(np.linalg.norm(np.asarray(forces.value), axis=1).max())
                               if forces is not None and forces.value is not None
                               and len(forces.value) else None),
        })
        history = project.results.get(f"eam::{run_id}::relaxation_history")
        if history is not None and history.value:
            payload["steps"] = len(history.value.get("step", []))
        temperature = project.results.get(f"eam::{run_id}::mean_temperature")
        if temperature is not None:
            payload["mean_temperature_K"] = temperature.value
            payload["temperature_stddev_K"] = temperature.uncertainty
        trajectory = project.results.get(f"eam::{run_id}::trajectory")
        if trajectory is not None and trajectory.value:
            payload["steps"] = len(trajectory.value.get("step", []))
            payload["frames"] = int(trajectory.value.get("frames", 0))
            payload["duration_fs"] = (float(trajectory.value.get("time_fs", [0.0])[-1])
                                      if trajectory.value.get("time_fs") else 0.0)
            payload["trajectory_available"] = bool(
                trajectory.value.get("positions_array") in project.arrays)
            if trajectory.convergence:
                payload["energy_drift_eV"] = trajectory.convergence.residual
        return payload

    def eam_trajectory(self, run_id: Optional[str] = None,
                       frame: Optional[int] = None,
                       project: Optional[Project] = None) -> dict:
        """Trajectory metadata or one renderable frame from a stored EAM or LAMMPS run.

        The trajectory contract is shared: a LAMMPS run identifier is served
        from the LAMMPS record, so the same player replays either backend.
        """
        project = project or self.project
        if run_id is not None and f"lammps::{run_id}::run" in project.results:
            return self.lammps_trajectory(run_id, frame, project=project)
        result = self.eam_result(run_id, project=project)
        if not result.get("ok"):
            return result
        run_id = result["run_id"]
        trajectory = project.results.get(f"eam::{run_id}::trajectory")
        if trajectory is None or not trajectory.value:
            return {"ok": False, "error": f"EAM run {run_id} has no trajectory."}
        return self._trajectory_frames(project, run_id, trajectory.value, frame, "EAM")

    def _trajectory_frames(self, project: Project, run_id: str, values: dict,
                           frame: Optional[int], backend: str) -> dict:
        from ..core_model.cell import Cell
        from ..experiments.collisions import fragments

        positions_key = values.get("positions_array")
        velocities_key = values.get("velocities_array")
        positions = project.arrays.get(positions_key or "")
        velocities = project.arrays.get(velocities_key or "")
        if positions is None or velocities is None:
            return {"ok": False, "error": f"{backend} run {run_id} has no stored frames."}
        count = int(positions.data.shape[0])
        metadata = {
            "ok": True, "run_id": run_id, "frames": count, "backend": backend.lower(),
            "steps": [int(v) for v in values.get("step", [])],
            "times_fs": [float(v) for v in values.get("time_fs", [])],
            "potential_eV": [float(v) for v in values.get("potential_eV", [])],
            "kinetic_eV": [float(v) for v in values.get("kinetic_eV", [])],
            "total_eV": [float(v) for v in values.get("total_eV", [])],
            "temperature_K": [float(v) for v in values.get("temperature_K", [])],
            "collision": dict(positions.meta.get("collision", {})),
        }
        if frame is None:
            return metadata
        try:
            index = int(frame)
        except (TypeError, ValueError):
            return {"ok": False, "error": "Trajectory frame must be an integer."}
        if index < 0 or index >= count:
            return {"ok": False, "error": f"Trajectory frame must be from 0 to {count - 1}."}
        points = np.asarray(positions.data[index], dtype=float)
        speeds = np.asarray(velocities.data[index], dtype=float)
        meta = positions.meta
        numbers = np.asarray(meta.get("numbers", []), dtype=np.int32)
        ids = np.asarray(meta.get("atom_ids", []), dtype=np.int64)
        roles = list(meta.get("roles", ["bulk"] * len(numbers)))
        if len(numbers) != len(points) or len(ids) != len(points):
            return {"ok": False, "error": "Trajectory atom metadata is inconsistent."}
        structure = Structure(
            numbers, points,
            Cell(np.asarray(meta.get("cell"), dtype=float), tuple(meta.get("pbc", (False,) * 3))),
            ids=ids, roles=roles)
        labels, pieces = fragments(structure, points)
        lo, hi = structure.bounding_box()
        metadata.update({
            "frame": index,
            "positions": points.astype(np.float32).ravel().tolist(),
            "velocities": speeds.astype(np.float32).ravel().tolist(),
            "ids": ids.tolist(), "numbers": numbers.tolist(), "roles": roles,
            "fragment_labels": labels.tolist(), "fragments": pieces,
            "bounds": [lo.tolist(), hi.tolist()],
            "cell": structure.cell.matrix.tolist(), "pbc": list(structure.cell.pbc),
            "elements": {pt.symbol(int(z)): int(z) for z in np.unique(numbers)},
        })
        return metadata

    def lammps_status(self, refresh: bool = False) -> dict:
        """The LAMMPS found, its version and packages, and what applies to the structure."""
        from ..physics import eam as eam_module
        from ..solvers.lammps import discover
        from ..solvers.lammps import spec as specs

        environment = discover(refresh=refresh)
        out = {"environment": environment.as_dict(),
               "tasks": {k: dict(v) for k, v in specs.DEFAULTS.items()},
               "ensembles": list(specs.ENSEMBLES), "min_styles": list(specs.MIN_STYLES),
               "outputs": list(specs.OUTPUTS), "notes": dict(specs.FIELD_NOTES),
               "structure": None, "runs": []}
        s = self.structure
        if s is not None:
            present = sorted({pt.symbol(int(z)) for z in s.numbers})
            chosen, reason = eam_module.select_shipped(present)
            blocking: List[str] = []
            if chosen is None:
                blocking.append(reason)
            else:
                try:
                    spec = self.lab.lammps.spec(s, "energy", chosen)
                    blocking = specs.check(spec, environment).messages()
                except ApiError as exc:
                    blocking.append(str(exc))
            out["structure"] = {"elements": present, "n_atoms": len(s),
                                "periodicity": list(s.cell.pbc), "selected": chosen,
                                "reason": reason, "blocking": blocking}
        out["runs"] = self.lab.lammps.runs()[:12]
        return out

    def lammps_spec(self, task: str = "energy", potential: Optional[str] = None,
                    settings: Optional[dict] = None) -> dict:
        """Freeze and check a specification, and show the native input, without running."""
        from ..solvers.lammps import native
        from ..solvers.lammps import spec as specs

        s = self._require_structure()
        try:
            spec = self.lab.lammps.spec(s, task, potential or None, settings or {})
        except ApiError as exc:
            return {"ok": False, "error": str(exc)}
        described = specs.describe(spec)
        data = native.render_data(spec)
        return {"ok": described["check"]["ok"], "spec": described,
                "native_input": {"in.lammps": native.render_input(spec),
                                 "structure.data_sha256": native.sha256_text(data),
                                 "structure.data_bytes": len(data.encode())}}

    def lammps_run(self, task: str = "energy", potential: Optional[str] = None,
                   settings: Optional[dict] = None, background: bool = True,
                   apply: bool = True, timeout_s: Optional[float] = None) -> dict:
        """Run LAMMPS on a frozen copy of the active structure.

        Nothing reaches the project unless the run finishes and every output
        check passes; then the results are stored in one step and, if the
        structure has not changed, applied as one undoable change.  A refused,
        failed or cancelled run returns its reason and changes nothing.
        """
        from ..solvers.lammps import records
        from ..solvers.lammps import spec as specs
        from ..solvers.lammps.run import execute, new_run_id

        s = self._require_structure()
        try:
            spec = self.lab.lammps.spec(s, task, potential or None, settings or {})
        except ApiError as exc:
            self.warn(str(exc), "unsupported", "solver/lammps")
            return {"ok": False, "status": "refused", "error": str(exc)}
        report = specs.check(spec)
        if not report.ok:
            message = "; ".join(report.messages())
            self.warn(message, "unsupported", "solver/lammps")
            return {"ok": False, "status": "refused", "error": message,
                    "check": report.as_dict()}
        owner = self.project
        run_id = new_run_id()

        def work(job):
            def progress(fraction: float, message: str) -> None:
                job.progress = float(fraction)
                job.message = message

            try:
                outcome = execute(spec, progress=progress, cancelled=job.cancel_flag.is_set,
                                  run_id=run_id, timeout_s=timeout_s)
            except specs.SpecRefused as exc:
                self.warn(str(exc), "unsupported", "solver/lammps")
                return {"ok": False, "status": "refused", "error": str(exc), "run_id": run_id}
            if not outcome.ok:
                self.warn(outcome.reason, "warning" if outcome.status == "cancelled"
                          else "error", "solver/lammps")
                return {"ok": False, "status": outcome.status, "reason": outcome.reason,
                        "run_id": run_id,
                        "log_tail": str(outcome.audit.get("log_tail", ""))[-4000:],
                        "state": self.state() if owner is self.project else None}
            stored = records.store(owner, outcome, apply=apply)
            payload = self.lammps_result(run_id, project=owner)
            payload["apply_reason"] = stored["apply_reason"]
            if owner is self.project:
                self._autosave()
                payload["state"] = self.state()
            else:
                payload["owner_replaced"] = True
                payload["state"] = None
            return payload

        if not background:
            return work(_NullJob())
        job = self.jobs.submit("lammps", f"LAMMPS {task}, {spec.potential.get('id')}, "
                                         f"{spec.n_atoms} atoms", work, owner=owner)
        return {"job": job.as_dict(), "run_id": run_id}

    def lammps_runs(self) -> List[dict]:
        return self.lab.lammps.runs()

    def lammps_result(self, run_id: Optional[str] = None,
                      project: Optional[Project] = None) -> dict:
        """A stored LAMMPS run, summarised for display, with its state."""
        from ..solvers.lammps import records
        from ..solvers.lammps.run import key
        from ..solvers.lammps.spec import file_sha256

        project = project or self.project
        ids = records.run_ids(project)
        if run_id is None:
            if not ids:
                return {"ok": False, "error": "No LAMMPS run is stored."}
            run_id = ids[0]
        record = records.record(project, run_id)
        spec = records.spec_of(project, run_id)
        if record is None or spec is None:
            return {"ok": False, "error": f"No LAMMPS run {run_id}."}
        get = lambda name: project.results.get(key(run_id, name))
        energy = get("energy")
        forces = get("forces")
        info = records.state(project, run_id)
        audit = dict(record.value or {})
        potential = dict(spec.potential)
        path = potential.get("path") or ""
        matches = None
        if path:
            try:
                matches = file_sha256(path) == potential.get("sha256")
            except OSError:
                matches = False
        payload = {
            "ok": True, "backend": "lammps", "run_id": run_id, "task": spec.task,
            "status": record.extra.get("status"), "model": record.provenance.model,
            "origin": energy.provenance.origin.value, "fidelity": energy.provenance.fidelity.value,
            "energy_eV": energy.value, "energy_per_atom_eV": energy.value / max(1, spec.n_atoms),
            "n_atoms": spec.n_atoms,
            "convergence": energy.convergence.as_dict() if energy.convergence else None,
            "current": bool(info.get("current")), "run_state": info.get("state"),
            "state_reason": info.get("reason"), "applied": bool(info.get("applied")),
            "applicable": bool(info.get("applicable")),
            "apply_reason": record.extra.get("apply_reason"),
            "structure_key": spec.structure_key, "wall_time_s": audit.get("wall_time_s"),
            "potential": {k: potential.get(k) for k in (
                "id", "file_name", "sha256", "style", "elements", "license", "shipped",
                "citations", "cutoff_A")},
            "potential_file_matches": matches,
            "task_settings": spec.settings(), "outputs": list(spec.outputs),
            "units": spec.units, "boundary": list(spec.boundary),
            "approximations": list(energy.provenance.approximations),
            "boundary_conditions": energy.provenance.boundary_conditions,
            "max_force_eV_A": energy.extra.get("max_force_eV_A"),
            "lammps": {k: spec.lammps.get(k) for k in ("kind", "path", "version",
                                                        "packages", "git_info")},
            "command_line": audit.get("command_line"),
            "inputs": audit.get("inputs"), "input_script": audit.get("input_script"),
            "log_tail": str(audit.get("log_tail", ""))[-4000:],
            "log_warnings": audit.get("log_warnings", []),
            "spec_digest": spec.digest,
        }
        if forces is not None and payload["max_force_eV_A"] is None and forces.value is not None:
            payload["max_force_eV_A"] = float(np.linalg.norm(np.asarray(forces.value),
                                                             axis=1).max())
        stress = get("stress")
        if stress is not None:
            payload["stress_eV_A3"] = stress.value
            payload["pressure_bar"] = stress.extra.get("pressure_bar")
        relaxation = get("relaxation")
        if relaxation is not None and relaxation.value:
            payload["steps"] = relaxation.value.get("iterations")
            payload["stopping_criterion"] = relaxation.value.get("stopping_criterion")
            initial = relaxation.value.get("energy_initial_eV")
            if initial is not None:
                payload["initial_energy_eV"] = initial
                payload["energy_change_eV"] = energy.value - initial
        temperature = get("mean_temperature")
        if temperature is not None:
            payload["mean_temperature_K"] = temperature.value
            payload["temperature_stddev_K"] = temperature.uncertainty
        trajectory = get("trajectory")
        if trajectory is not None and trajectory.value:
            values = trajectory.value
            payload["steps"] = int(spec.steps or 0)
            payload["frames"] = int(values.get("frames", 0))
            payload["duration_fs"] = (float(values.get("time_fs", [0.0])[-1])
                                      if values.get("time_fs") else 0.0)
            payload["trajectory_available"] = bool(
                values.get("positions_array") in project.arrays)
            if trajectory.convergence:
                payload["energy_drift_eV"] = trajectory.convergence.residual
        return payload

    def lammps_trajectory(self, run_id: Optional[str] = None, frame: Optional[int] = None,
                          project: Optional[Project] = None) -> dict:
        """Trajectory metadata or one frame from a stored LAMMPS run, in the shared contract."""
        project = project or self.project
        result = self.lammps_result(run_id, project=project)
        if not result.get("ok"):
            return result
        run_id = result["run_id"]
        trajectory = project.results.get(f"lammps::{run_id}::trajectory")
        if trajectory is None or not trajectory.value:
            return {"ok": False, "error": f"LAMMPS run {run_id} has no trajectory."}
        return self._trajectory_frames(project, run_id, trajectory.value, frame, "LAMMPS")

    def lammps_apply(self, run_id: Optional[str] = None) -> dict:
        """Apply a stored run to its structure as one undoable change, or say why not."""
        try:
            message = self.lab.lammps.apply(run_id)
        except ApiError as exc:
            self.warn(str(exc), "unsupported", "solver/lammps")
            return {"ok": False, "error": str(exc)}
        self._autosave()
        return {"ok": True, "message": message, "state": self.state()}

    def solve(self, task: str = "electronic", model: str = "recommended",
              background: bool = True, **kwargs) -> dict:
        s = self._require_structure()

        def work(job):
            result = self._run_solver(task, model, s, kwargs, job)
            self._autosave()
            return result

        if not background:
            result = self._run_solver(task, model, s, kwargs, _NullJob())
            self._autosave()
            return result
        job = self.jobs.submit("solver", f"{task} ({model})", work)
        return {"job": job.as_dict()}

    def _run_solver(self, task: str, model: str, s: Structure, kwargs: dict, job) -> dict:
        material = self._active_material()
        try:
            solver = self.lab._resolve_solver(model, s, material,
                                              "relax" if task == "relax" else task)
        except Exception as exc:
            self.warn(str(exc), "error", "solver")
            return {"supported": False, "reason": str(exc)}

        from ..solvers.base import Capability
        needed = {"relax": Capability.RELAX, "md": Capability.DYNAMICS,
                  "dynamics": Capability.DYNAMICS, "energy": Capability.ENERGY}.get(task)
        if needed is not None and not solver.can(needed):
            refusal = solver.require(needed)
            self.warn(refusal.unsupported_reason, "unsupported", "solver")
            return {"supported": False, "reason": refusal.unsupported_reason,
                    "suggested_models": refusal.suggested_models}

        if task in ("relax", "md", "dynamics"):
            report = solver.supports(s)
            if not report.ok:
                reason = "; ".join(report.blocking)
                self.warn(reason, "unsupported", "solver")
                return {"supported": False, "reason": reason}
        before_state = s.copy()
        before_selection = Selection(list(self.project.selection.ids),
                                     self.project.selection.query)

        def refused(exc: Exception) -> dict:
            s.restore_from(before_state)
            reason = (f"The {task} run was refused part way and the structure was put "
                      f"back as it was: {exc}")
            self.warn(reason, "unsupported", "solver")
            return {"supported": False, "reason": reason}

        if task == "relax":
            def cb(step, energy, fmax):
                job.progress = min(0.99, step / max(kwargs.get("steps", 400), 1))
                job.message = f"step {step}: E = {energy:.4f} eV, max|F| = {fmax:.4f} eV/A"
                return not job.cancel_flag.is_set()
            before = s.positions.copy()
            try:
                out = solver.relax(s, fmax_eV_A=float(kwargs.get("fmax", 0.02)),
                                   max_steps=int(kwargs.get("steps", 400)),
                                   in_place=True, callback=cb)
            except (PotentialError, CoincidentAtoms) as exc:
                return refused(exc)
            self.project.record_change(s, before_state, before_selection,
                                       f"Relax with {solver.name}", "solver.relax",
                                       {"model": solver.name, **kwargs})
            for key, res in out.results.items():
                self.project.add_result(f"relax::{key}", res)
            moved = float(np.linalg.norm(s.positions - before, axis=1).max())
            s.invalidate_bonds()
            return {"supported": True, "task": task, "solver": out.solver,
                    "converged": out.convergence.converged,
                    "convergence": out.convergence.as_dict(),
                    "energy_eV": out.results["energy"].value,
                    "energy_change_eV": out.results["energy"].extra.get("energy_change_eV"),
                    "max_displacement_A": moved,
                    "history": _jsonable(out.results["relaxation_history"].value),
                    "checks": _jsonable(out.results["energy"].extra.get("potential_checks")),
                    "log": list(out.log),
                    "provenance": out.results["energy"].provenance.as_dict()}

        if task in ("md", "dynamics"):
            def cb(step, energy, temp):
                job.progress = min(0.99, step / max(kwargs.get("steps", 200), 1))
                job.message = f"step {step}: T = {temp:.1f} K"
                return not job.cancel_flag.is_set()
            try:
                out = solver.dynamics(s, in_place=True, callback=cb,
                                      **_filter_call(solver.dynamics, kwargs))
            except (PotentialError, CoincidentAtoms) as exc:
                return refused(exc)
            self.project.record_change(s, before_state, before_selection,
                                       f"Dynamics with {solver.name}", "solver.dynamics",
                                       {"model": solver.name, **kwargs})
            for key, res in out.results.items():
                self.project.add_result(f"md::{key}", res)
            s.invalidate_bonds()
            traj = out.results["trajectory"].value
            return {"supported": True, "task": task, "solver": out.solver,
                    "trajectory": _jsonable(traj),
                    "mean_temperature_K": (out.results["mean_temperature"].value
                                           if "mean_temperature" in out.results else None),
                    "checks": _jsonable(out.results["energy"].extra.get("potential_checks")),
                    "log": list(out.log),
                    "provenance": out.results["trajectory"].provenance.as_dict()}

        if task == "energy":
            out = solver.single_point(s)
            res = out.results["energy"]
            self.project.add_result("energy", res)
            if not res.supported:
                return {"supported": False, "reason": res.unsupported_reason,
                        "suggested_models": res.suggested_models}
            return {"supported": True, "task": task, "solver": out.solver,
                    "energy_eV": res.value,
                    "energy_per_atom_eV": res.extra.get("energy_per_atom_eV"),
                    "max_force_eV_A": out.results["forces"].extra.get("max_force_eV_A"),
                    "checks": _jsonable(res.extra.get("potential_checks")),
                    "log": list(out.log),
                    "provenance": res.provenance.as_dict()}

        if task in ("electronic", "eigenstates"):
            from .history_note import note_operation
            note_operation(self.project, f"Electronic structure with {solver.name}",
                           "solver.electronic", {"model": solver.name})
            out = solver.eigenstates(s, **_filter_call(solver.eigenstates, kwargs))
            for key, res in out.results.items():
                self.project.add_result(f"electronic::{key}", res)
            if not out.results["eigenvalues"].supported:
                r = out.results["eigenvalues"]
                return {"supported": False, "reason": r.unsupported_reason,
                        "suggested_models": r.suggested_models}
            dos = out.results["total_dos"].value
            ldos = out.results["local_dos"].value
            payload = {
                "supported": True, "task": task, "solver": out.solver,
                "fermi_level_eV": out.results["fermi_level"].value,
                "fermi_note": out.results["fermi_level"].extra,
                "eigenvalues": out.results["eigenvalues"].value.tolist(),
                "dos": {"energy_eV": dos["energy_eV"].tolist(),
                        "dos": dos["dos"].tolist()},
                "ldos": {"energy_eV": ldos["energy_eV"].tolist(),
                         "atom_ids": ldos["atom_ids"]},
                "partial_charges": out.results["partial_charges"].value.tolist(),
                "provenance": out.results["eigenvalues"].provenance.as_dict(),
                "log": out.log,
            }
            if "hl_gap" in out.results:
                payload["hl_gap_eV"] = out.results["hl_gap"].value
                payload["hl_gap_note"] = out.results["hl_gap"].extra
            if "self_consistency" in out.results:
                r = out.results["self_consistency"]
                payload["self_consistency"] = {
                    "supported": False, "reason": r.unsupported_reason,
                    "suggested_models": r.suggested_models}
            self._last_ldos = ldos
            return payload

        if task == "band_structure":
            out = solver.band_structure(s, **_filter_call(solver.band_structure, kwargs))
            res = out.results["band_structure"]
            if not res.supported:
                return {"supported": False, "reason": res.unsupported_reason,
                        "suggested_models": res.suggested_models}
            v = res.value
            gap = out.results["band_gap"]
            return {"supported": True, "task": task, "solver": out.solver,
                    "k_coord": v["k_coord"], "bands_eV": v["bands_eV"].tolist(),
                    "ticks": v["ticks"], "labels": v["labels"],
                    "band_gap_eV": gap.value, "band_gap_note": gap.extra,
                    "fermi_level_eV": out.results["fermi_level"].value,
                    "provenance": res.provenance.as_dict()}

        return {"supported": False,
                "reason": f"Unknown task {task!r}. Available: energy, relax, md, "
                          "electronic, band_structure."}

    def atom_ldos(self, atom_id: int) -> dict:
        ldos = getattr(self, "_last_ldos", None)
        if ldos is None:
            return {"supported": False,
                    "reason": "No local density of states has been computed yet. "
                              "Run an electronic-structure calculation first."}
        ids = list(ldos["atom_ids"])
        if int(atom_id) not in ids:
            return {"supported": False,
                    "reason": f"Atom {atom_id} was not in the last electronic "
                              "calculation. Re-run it after the structure changed."}
        k = ids.index(int(atom_id))
        return {"supported": True, "energy_eV": ldos["energy_eV"].tolist(),
                "ldos": ldos["ldos"][:, k].tolist(), "atom_id": int(atom_id)}

    def edit(self, op: str, **kwargs) -> dict:
        from ..structure_builder import defects
        s = self._require_structure()
        sel = self.project.selection
        if op == "substitute":
            ids = kwargs.get("ids") or sel.ids
            element = kwargs["element"]
            if not ids:
                raise ValueError("Select at least one atom to substitute.")
            self.project.begin(f"Substitute {element} ({len(ids)} atoms)",
                               "edit.substitute", {"ids": list(ids), "element": element})
            recs = [defects.substitute_atom(s, int(i), element) for i in ids]
        elif op == "vacancy":
            ids = kwargs.get("ids") or sel.ids
            if not ids:
                raise ValueError("Select at least one atom to remove.")
            self.project.begin(f"Create {len(ids)} vacancy(ies)", "edit.vacancy",
                               {"ids": list(ids)})
            recs = [defects.create_vacancy(s, int(i)) for i in sorted(ids, reverse=True)]
            self.project.selection = Selection([], "")
        elif op == "interstitial":
            self.project.begin(f"Add {kwargs['element']} interstitial",
                               "edit.interstitial", kwargs)
            recs = [defects.add_interstitial(s, kwargs["element"], kwargs["position"],
                                             **_filter_call(defects.add_interstitial,
                                                            kwargs, {"element", "position"}))]
        elif op == "adatom":
            self.project.begin(f"Add {kwargs['element']} adatom", "edit.adatom", kwargs)
            height = kwargs.get("height_A")
            recs = [defects.add_adatom(s, kwargs["element"], kwargs["xy"],
                                       None if height is None else float(height))]
        elif op == "move":
            aid, delta = int(kwargs["atom_id"]), np.asarray(kwargs["delta_A"], float)
            self.project.begin(f"Move atom #{aid}", "edit.move", kwargs)
            s.set_position(aid, s.positions[s.index_of(aid)] + delta)
            recs = [{"type": "move", "atom_id": aid, "delta_A": delta.tolist()}]
        elif op == "set_charge":
            ids = kwargs.get("ids") or sel.ids
            q = float(kwargs["charge"])
            self.project.begin(f"Set charge {q:+g} e on {len(ids)} atom(s)",
                               "edit.charge", {"ids": list(ids), "charge": q})
            for i in ids:
                s.formal_charges[s.index_of(int(i))] = q
            recs = [{"type": "charge", "ids": list(ids), "charge": q}]
        elif op == "set_spin":
            ids = kwargs.get("ids") or sel.ids
            spin = float(kwargs["spin"])
            self.project.begin(f"Set spin S = {spin:g} on {len(ids)} atom(s)",
                               "edit.spin", {"ids": list(ids), "spin": spin})
            for i in ids:
                s.magnetic_moments[s.index_of(int(i))] = 2.0 * spin
            recs = [{"type": "spin", "ids": list(ids), "spin": spin}]
        elif op == "set_isotope":
            ids = kwargs.get("ids") or sel.ids
            a = int(kwargs["mass_number"])
            self.project.begin(f"Set isotope A = {a}", "edit.isotope",
                               {"ids": list(ids), "mass_number": a})
            for i in ids:
                k = s.index_of(int(i))
                pt.element(int(s.numbers[k])).isotope(a)
                s.mass_numbers[k] = a
            recs = [{"type": "isotope", "ids": list(ids), "mass_number": a}]
        elif op == "fix":
            ids = kwargs.get("ids") or sel.ids
            value = bool(kwargs.get("fixed", True))
            self.project.begin(("Fix" if value else "Release") + f" {len(ids)} atom(s)",
                               "edit.fix", {"ids": list(ids), "fixed": value})
            for i in ids:
                s.fixed[s.index_of(int(i))] = value
            recs = [{"type": "fix", "ids": list(ids), "fixed": value}]
        elif op == "strain":
            from ..structure_builder.lattice import apply_strain
            self.project.begin("Apply strain", "edit.strain", kwargs)
            new = apply_strain(s, np.asarray(kwargs["strain"], dtype=float))
            self.project.structures[self.project.active_structure_key] = new
            recs = [{"type": "strain", "strain": np.asarray(kwargs["strain"]).tolist()}]
            s = new
        elif op == "passivate":
            from ..structure_builder.surface import passivate as _pass
            self.project.begin("Passivate surface", "edit.passivate", kwargs)
            n = _pass(s, kwargs.get("element", "H"), side=kwargs.get("side", "bottom"))
            recs = [{"type": "passivate", "added": n}]
        else:
            raise ValueError(
                f"Unknown edit operation {op!r}. Available: substitute, vacancy, "
                "interstitial, adatom, move, set_charge, set_spin, set_isotope, "
                "fix, strain, passivate.")
        attach_bonds(s)
        self.project.selection = self.project.selection.prune(s)
        from ..physics.potentials import overlap_problem

        overlap = overlap_problem(s)
        if overlap is not None:
            self.warn(f"{overlap} Energy, relaxation and dynamics will be refused until "
                      "this is corrected; undo restores the previous structure.",
                      "warning", "geometry")
        charge = s.total_charge()
        if abs(charge) > 1e-9:
            self.warn(
                f"The structure now carries a net charge of {charge:+g} e. "
                "Classical potentials ignore it entirely, and the tight-binding "
                "model is restricted to neutral systems; a charged calculation "
                "needs a self-consistent solver with a compensating background.",
                "warning", "charge bookkeeping")
        self._autosave()
        return {"records": _jsonable(recs), "state": self.state()}

    def run_script(self, code: str, background: bool = True) -> dict:
        if background:
            def work(job):
                result = self.runner.run(code).as_dict()
                self._autosave()
                return result
            job = self.jobs.submit("script", "Python script", work)
            return {"job": job.as_dict()}
        result = self.runner.run(code).as_dict()
        self._autosave()
        return result

    def set_script_mode(self, mode: str, confirm: bool = False) -> dict:
        if mode == "trusted" and not confirm:
            return {"ok": False, "requires_confirmation": True,
                    "message": ("Trusted mode removes the import and builtins "
                                "restrictions. A trusted script can read, write and "
                                "delete any file your user account can, start "
                                "processes and open network connections. Restricted "
                                "mode is a guard rail against accidents, not a "
                                "security boundary, but it does stop casual mistakes. "
                                "Continue only if you wrote or reviewed the script.")}
        self.runner.set_mode(mode)
        if mode == "trusted":
            self.trusted_confirmed = True
            self.project.history.log.append(
                __import__("materia.project_format.history", fromlist=["LogEntry"])
                .LogEntry(operation="script.mode", label="Python console set to trusted mode",
                          parameters={"mode": mode}, undoable=False))
            self._autosave()
        return {"ok": True, "mode": self.runner.mode}

    def script_environment(self) -> dict:
        from ..python_api.execution import ALLOWED_MODULES, MODES
        return {"mode": self.runner.mode, "modes": list(MODES),
                "allowed_modules": sorted(ALLOWED_MODULES),
                "namespace": sorted(k for k in self.runner.globals
                                    if not k.startswith("__"))}

    def _install_exit_hook(self) -> None:
        """Last line of defence: stop background work if the process exits any
        other way than through a close handler."""
        import atexit

        atexit.register(self._atexit_shutdown)

    def _atexit_shutdown(self) -> None:
        try:
            if not self.jobs.is_shut_down:
                self.shutdown(grace_s=5.0)
        except BaseException:
            pass

    def shutdown(self, grace_s: float = 10.0) -> dict:
        """Stop all background work. Idempotent, and safe to call from any path.

        Every way Materia can close -- the native window, the command line, an
        interpreter shutting down -- goes through here, so a GPAW subprocess can
        never outlive the application that started it.
        """
        report = self.jobs.shutdown(grace_s=grace_s)
        return report

    def job_status(self, job_id: str, include_result: bool = True) -> dict:
        job = self.jobs.get(job_id)
        if job is None:
            raise ValueError(f"Unknown job {job_id}")
        finished = job.status in ("done", "cancelled", "failed")
        return job.as_dict(include_result=include_result and finished)

    def cancel_job(self, job_id: str) -> dict:
        return {"cancelled": self.jobs.cancel(job_id)}

    def undo(self) -> dict:
        label = self.project.undo()
        self._autosave()
        return {"label": label, "state": self.state()}

    def redo(self) -> dict:
        label = self.project.redo()
        self._autosave()
        return {"label": label, "state": self.state()}

    def save_project(self, path: str) -> dict:
        saved = self.project.save(path)
        self.recovery.discard()
        self.recovery_candidate = None
        return {"path": saved}

    def _replace_project(self, project: Project, how: str) -> None:
        """Swap in a different project, stopping work started for the old one.

        A calculation belongs to the project it was submitted from.  When that
        project is replaced there is nowhere for its answer to go, so every
        unfinished job owned by it is cancelled and says why.  Nothing a
        running job produces can reach the incoming project.
        """
        outgoing = self.lab.project if self.lab is not None else None
        if outgoing is not None:
            stopped = self.jobs.cancel_owned_by(
                outgoing,
                f"Cancelled because the project was replaced ({how}).")
            if stopped:
                self.warn(
                    f"{len(stopped)} running calculation(s) were cancelled because "
                    f"the project was replaced ({how}). Their results were not "
                    "carried over.", "warning", "jobs")
        self.lab = Lab(project, self.library)

    def open_project(self, path: str) -> dict:
        project = Project.load(path, library=self.library)
        self._replace_project(project, "opened from disk")
        self.runner = ScriptRunner(self.lab, mode=self.runner.mode)
        self.last_scan_key = None
        self.recovery.discard()
        self.recovery_candidate = None
        return self.state()

    def new_project(self, name: str = "Untitled project") -> dict:
        self.warnings.clear()
        self._replace_project(Project(name, self.library), "new project")
        self.runner = ScriptRunner(self.lab, mode=self.runner.mode)
        self.last_scan_key = None
        self.recovery.discard()
        self.recovery_candidate = None
        return self.state()

    def checkpoint(self, name: str, note: str = "") -> dict:
        self.project.save_checkpoint(name, note)
        self._autosave()
        return self.state()

    def restore_checkpoint(self, name: str) -> dict:
        self.project.restore_checkpoint(name)
        self._autosave()
        return self.state()

    def recover_project(self) -> dict:
        project = self.recovery.load(library=self.library)
        self._replace_project(project, "recovered from autosave")
        self.runner = ScriptRunner(self.lab, mode=self.runner.mode)
        self.last_scan_key = next(reversed(project.scans), None)
        self.recovery_candidate = None
        self.warn("Recovered the latest autosaved project.", "info", "recovery")
        return self.state()

    def discard_recovery(self) -> dict:
        discarded = self.recovery.discard()
        self.recovery_candidate = None
        return {"discarded": discarded, "state": self.state()}

    def provenance_text(self) -> str:
        return self.project.provenance_text()

    def export_structure(self, path: str, fmt: Optional[str] = None) -> dict:
        s = self._require_structure()
        return {"path": dataio.write_structure(path, s, fmt)}

    def export_image(self, path: str, channel: Optional[str] = None,
                     palette: str = "silver", key: Optional[str] = None,
                     **kwargs) -> dict:
        key = key or self.last_scan_key
        if key is None:
            raise ValueError("No scan to export.")
        scan = self.project.scans[key]
        return {"path": self.lab.io.write_image(path, scan, channel, palette, **kwargs)}

    def export_csv(self, path: str, what: str = "atoms") -> dict:
        if what == "atoms":
            return {"path": dataio.write_csv(path, dataio.structure_table(
                self._require_structure()), self.project.name)}
        if what == "scan":
            key = self.last_scan_key
            scan = self.project.scans[key]
            data = scan.channel()
            cols = {"row": [], "col": [], "x_A": [], "y_A": [], "value": []}
            for r in range(data.shape[0]):
                for c in range(data.shape[1]):
                    x, y = scan.pixel_to_xy(c, r)
                    cols["row"].append(r)
                    cols["col"].append(c)
                    cols["x_A"].append(f"{x:.4f}")
                    cols["y_A"].append(f"{y:.4f}")
                    cols["value"].append(f"{data[r, c]:.6g}")
            return {"path": dataio.write_csv(path, cols, scan.provenance.summary())}
        raise ValueError(f"Unknown export target {what!r}; use 'atoms' or 'scan'.")

    def export_npz(self, path: str, parts: Optional[Sequence[str]] = None) -> dict:
        """Write the project's numerical data to one verified NumPy archive."""
        from ..dataio import npz

        try:
            out = npz.export(path, self.project, tuple(parts) if parts else npz.PARTS)
        except npz.NpzExportError as exc:
            self.warn(str(exc), "error", "export")
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "path": out["path"], "entries": out["entries"],
                "bytes": out["bytes"], "skipped": out["skipped"]}

    def import_structure(self, path: str) -> dict:
        s = dataio.read_structure(path)
        attach_bonds(s)
        key = self.project.add_structure(s)
        self._autosave()
        return {"key": key, "n_atoms": len(s), "state": self.state()}


def _silver_lut() -> List[List[int]]:
    """The scanning-probe palette, 256 RGB entries, for maps drawn in the interface."""
    from ..visualization.palette import get as get_palette

    return get_palette("silver").lut().tolist()


class _NullJob:
    class _Flag:
        @staticmethod
        def is_set():
            return False
    cancel_flag = _Flag()
    progress = 0.0
    message = ""


def _filter_kwargs(cls, kwargs: dict) -> dict:
    import dataclasses
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(kwargs) - names)
    if unknown:
        raise ValueError(f"Unknown settings for {cls.__name__}: {', '.join(unknown)}. "
                         f"Accepted: {sorted(names)}")
    out = dict(kwargs)
    for key in ("resolution", "kgrid"):
        if key in out and out[key] is not None:
            out[key] = tuple(int(v) for v in out[key])
    if out.get("window_A"):
        out["window_A"] = tuple(float(v) for v in out["window_A"])
    return out


def _filter_call(fn, kwargs: dict, drop: Optional[set] = None) -> dict:
    import inspect
    sig = inspect.signature(fn)
    allowed = {k for k in sig.parameters if k not in ("self", "structure", "kwargs")}
    return {k: v for k, v in kwargs.items()
            if k in allowed and (not drop or k not in drop)}


for _name in dir(_InstrumentMixin):
    if not _name.startswith("__"):
        setattr(Service, _name, getattr(_InstrumentMixin, _name))
