"""The public Python API.

Scripts written in the embedded editor, in a notebook, or in a plain ``.py``
file all see the same namespace.  Nothing here is a stub: every call runs the
same code paths the graphical interface uses.

Namespace
---------
``project``      the active :class:`~materia.project_format.Project`
``materials``    the material registry
``build``        crystal, surface and defect construction
``solvers``      the solver registry and factories
``microscope``   virtual STM and AFM
``view``         hand results to the interface (or collect them headlessly)
``measure``      distances, angles, coordination, strain
``alloys``       exact-composition random alloys and partial occupancy sampling
``mechanics``    external atomic forces and harmonic positional restraints
``search``       deterministic fixed-cell basin hopping and ranked minima
``compare``      boundary-aware before-and-after structural analysis
``ensembles``    classical model-force and energy-change disagreement
``phonons``      finite-displacement harmonic modes and zero-point energy
``neb``          minimum-energy paths and climbing-image reaction barriers
``electrostatics`` point-charge Coulomb energies, forces and site potentials
``eam``          embedded-atom potentials for metals: energy, relaxation, dynamics
``collisions``   prepare and run classical atomistic impact experiments
``lammps``       LAMMPS out of process: frozen specifications, validated results
``dft``          GPAW first principles: ground-state experiments, convergence studies,
                 fixed-cell or variable-cell relaxation, DOS and projected DOS, and
                 band structure along a reciprocal-space path, equations of state, and
                 spatial LDOS with Tersoff-Hamann STM images
``claims``       classified statements and the evidence they cite
``io``           import and export
``np``           NumPy

Example
-------
>>> si = materials.load("silicon", orientation="111")            # doctest: +SKIP
>>> surface = si.create_surface(size=(6, 6, 4), vacuum_angstrom=15)
>>> surface.add_dopant("phosphorus", site=surface.nearest_site((10.0, 10.0, 20.0)))
>>> surface.relax(model="recommended")
>>> scan = microscope.stm_scan(surface, bias_volts=0.8, current_nA=0.5,
...                            resolution=(256, 256))
>>> view.display(scan, palette="silver")
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

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
from ..core_model.structure import AtomView, Structure
from ..dataio import formats as _io
from ..elements import periodic_table as _pt
from ..materials.loader import MaterialLibrary, default_library
from ..materials.schema import MaterialDefinition
from ..multiscale.region import RegionSpec, extract_region
from ..multiscale.wafer import DeviceRegion, Layer, Wafer, WaferSpec
from ..physics.bonds import attach_bonds, perceive_bonds
from ..physics.neighbors import neighbor_list
from ..physics.strain import local_strain
from ..project_format.project import Project
from ..provenance import Result, unsupported
from ..solvers import registry as _registry
from ..solvers.base import Capability
from ..structure_builder import defects as _defects
from ..structure_builder.lattice import apply_strain, bulk, lattice_planes
from ..structure_builder.surface import make_surface, passivate


class ApiError(Exception):
    pass


class UnsupportedRequest(ApiError):
    """A request the program declines to answer rather than approximate.

    ``record`` is the machine-readable unsupported payload: reason, citation
    and the suggested models that could answer it.
    """

    def __init__(self, message: str, record: dict) -> None:
        super().__init__(message)
        self.unsupported = dict(record)

    def as_dict(self) -> dict:
        return dict(self.unsupported)


class SurfaceHandle:
    """A structure plus the material it came from, with convenience methods."""

    def __init__(self, structure: Structure, material: MaterialDefinition,
                 lab: "Lab") -> None:
        self.structure = structure
        self.material = material
        self._lab = lab

    def __len__(self) -> int:
        return len(self.structure)

    def __repr__(self) -> str:
        info = self.structure.info.get("surface", {})
        miller = "".join(str(v) for v in info.get("miller", []))
        return (f"<Surface {self.material.name}({miller}) "
                f"{len(self.structure)} atoms, {self.structure.formula()}>")

    @property
    def atoms(self) -> "AtomCollection":
        return AtomCollection(self.structure, self._lab)

    @property
    def cell(self):
        return self.structure.cell

    def info(self) -> dict:
        return dict(self.structure.info.get("surface", {}))

    def nearest_site(self, point: Sequence[float], element: Optional[str] = None) -> int:
        return _defects.nearest_site(self.structure, point, element)

    def bonds(self):
        attach_bonds(self.structure)
        return self.structure.bonds

    def add_dopant(self, element: str, site: Optional[int] = None,
                   near: Optional[Sequence[float]] = None,
                   host_element: Optional[str] = None) -> dict:
        self._lab._checkpoint(f"Add {element} dopant", "defect.dopant",
                              {"element": element, "site": site})
        return _defects.add_dopant(self.structure, element, site, near, host_element)

    def substitute(self, site: int, element: str, **kwargs) -> dict:
        self._lab._checkpoint(f"Substitute {element}", "defect.substitute",
                              {"site": site, "element": element})
        return _defects.substitute_atom(self.structure, site, element, **kwargs)

    def create_vacancy(self, site: int) -> dict:
        self._lab._checkpoint("Create vacancy", "defect.vacancy", {"site": site})
        return _defects.create_vacancy(self.structure, site)

    def add_interstitial(self, element: str, position: Sequence[float], **kwargs) -> dict:
        self._lab._checkpoint(f"Add {element} interstitial", "defect.interstitial",
                              {"element": element, "position": list(position)})
        return _defects.add_interstitial(self.structure, element, position, **kwargs)

    def add_adatom(self, element: str, xy: Sequence[float],
                   height_A: Optional[float] = None) -> dict:
        self._lab._checkpoint(f"Add {element} adatom", "defect.adatom",
                              {"element": element, "xy": list(xy)})
        return _defects.add_adatom(self.structure, element, xy, height_A)

    def reconstruct(self, reconstruction: str, *, relax: bool = True,
                    model: str = "recommended", fmax: float = 0.005,
                    steps: int = 800, **options) -> dict:
        """Apply a declared surface reconstruction to the upper face.

        Returns the reconstruction record: generator, pairing, relaxation,
        measured geometry and the 1x1/2x1 symmetry check.  A reconstruction the
        material declares but this build cannot generate raises
        :class:`UnsupportedRequest` carrying the machine-readable unsupported
        record; nothing approximate is returned in its place.

        A request that is refused, for any reason, leaves the structure and the
        project history exactly as they were: the undo point is recorded after
        the change has been made, not before it is attempted.

        Where that undo point lands follows what the project holds.  A handle on
        the active structure records one in the usual way.  A handle on a
        structure the project holds but has not activated records one bound to
        that slot, so undo puts *that* structure back and leaves the active one
        alone.  A detached handle -- built with ``activate=False`` and never
        registered -- records nothing, because the project does not own the
        object and must not write another structure's history on its behalf.
        """
        from ..core_model.selection import Selection
        from ..structure_builder.reconstruction import (
            ReconstructionNotImplemented, apply_reconstruction,
        )

        project = self._lab.project
        before = self.structure.copy()
        before_selection = Selection(list(project.selection.ids), project.selection.query)
        try:
            apply_reconstruction(self.structure, self.material, reconstruction,
                                 relax=relax, model=model, fmax_eV_A=fmax,
                                 max_steps=steps, options=options or None)
        except ReconstructionNotImplemented as exc:
            raise UnsupportedRequest(exc.reason, exc.as_dict()) from None
        self.structure.invalidate_bonds()
        record = self.structure.info["reconstruction"]
        project.record_change(
            self.structure, before, before_selection,
            f"Reconstruct {reconstruction}", "structure.reconstruct",
            {"reconstruction": reconstruction, "relax": relax, "model": model},
            result_summary=f"geometry {record['geometry_status']}")
        return dict(record)

    def reconstruction(self) -> Optional[dict]:
        """The reconstruction record carried by this structure, if any."""
        record = self.structure.info.get("reconstruction")
        return dict(record) if record else None

    def compare_reconstruction(self) -> dict:
        """Measured reconstruction geometry against the published values cited
        in the material definition.  Deviations are reported, never corrected."""
        from ..structure_builder.reconstruction import compare_to_reference

        return compare_to_reference(self.structure, self.material)

    def passivate(self, element: str = "H", side: str = "bottom") -> int:
        self._lab._checkpoint(f"Passivate with {element}", "structure.passivate",
                              {"element": element, "side": side})
        return passivate(self.structure, element, side=side)

    def strain(self, tensor) -> "SurfaceHandle":
        self._lab._checkpoint("Apply strain", "structure.strain",
                              {"strain": np.asarray(tensor).tolist()})
        self.structure = apply_strain(self.structure, tensor)
        return self

    def fix_below(self, z_A: float) -> int:
        mask = self.structure.positions[:, 2] <= z_A
        self.structure.fixed[:] = mask
        return int(mask.sum())

    def apply_force(self, atom_id: int, force_eV_A: Sequence[float],
                    reference_position_A: Optional[Sequence[float]] = None) -> dict:
        return self._lab.mechanics.apply_force(
            self, atom_id, force_eV_A,
            reference_position_A=reference_position_A)

    def restrain(self, atom_id: int, spring_eV_A2: float,
                 target_position_A: Optional[Sequence[float]] = None) -> dict:
        return self._lab.mechanics.restrain(
            self, atom_id, spring_eV_A2,
            target_position_A=target_position_A)

    def clear_biases(self, atom_ids=None, kind: Optional[str] = None) -> int:
        return self._lab.mechanics.clear(self, atom_ids=atom_ids, kind=kind)

    def biases(self) -> dict:
        return self._lab.mechanics.list(self)

    def search_minima(self, model: str = "recommended", settings=None,
                      activate_best: bool = False, **kwargs):
        return self._lab.search.basin_hopping(
            self, model=model, settings=settings,
            activate_best=activate_best, **kwargs)

    def compare_to(self, other, **options):
        return self._lab.compare.structures(self, other, **options)

    def relax(self, model: str = "recommended", **kwargs) -> Result:
        return self._lab.relax(self.structure, model=model, material=self.material, **kwargs)

    def dynamics(self, model: str = "recommended", **kwargs) -> Result:
        return self._lab.dynamics(self.structure, model=model, material=self.material, **kwargs)

    def alloy(self, composition: Mapping[str, float], *, atom_ids=None,
              seed: int = 0, activate: bool = True) -> "SurfaceHandle":
        return self._lab.alloys.random(
            self, composition, atom_ids=atom_ids, seed=seed, activate=activate)

    def phonons(self, model: str = "recommended", settings=None, **kwargs):
        return self._lab.phonons(
            self.structure, model=model, material=self.material,
            settings=settings, **kwargs)

    def phonon_dispersion(self, model: str = "recommended", settings=None, **kwargs):
        return self._lab.phonon_dispersion(
            self.structure, model=model, material=self.material,
            settings=settings, **kwargs)

    def solve(self, model: str = "recommended", **kwargs):
        return self._lab.solve(self.structure, model=model, material=self.material, **kwargs)

    def energy(self, model: str = "recommended") -> Result:
        return self._lab.energy(self.structure, model=model, material=self.material)

    def save(self, path: str, fmt: Optional[str] = None) -> str:
        return _io.write_structure(path, self.structure, fmt)

    def copy(self) -> "SurfaceHandle":
        return SurfaceHandle(self.structure.copy(), self.material, self._lab)


class MaterialHandle:
    """A material definition with construction helpers."""

    def __init__(self, definition: MaterialDefinition, lab: "Lab",
                 orientation: Optional[Sequence[int]] = None) -> None:
        self.definition = definition
        self._lab = lab
        self.orientation = tuple(int(v) for v in (orientation
                                                  or definition.orientations[0]
                                                  if definition.orientations else (0, 0, 1)))

    def __repr__(self) -> str:
        return (f"<Material {self.definition.name} ({self.definition.formula}), "
                f"{self.definition.prototype}, default orientation "
                f"{''.join(str(v) for v in self.orientation)}>")

    @property
    def id(self) -> str:
        return self.definition.id

    def summary(self) -> dict:
        return self.definition.summary()

    def property(self, key: str):
        p = self.definition.property(key)
        if p is None:
            raise ApiError(
                f"{self.definition.id} has no tabulated property {key!r}. "
                f"Available: {sorted(self.definition.properties)}"
            )
        return p

    def bulk(self, repeat: Sequence[int] = (1, 1, 1),
             activate: bool = True, occupancy_seed: Optional[int] = None) -> SurfaceHandle:
        partial = any(site.occupancy < 1.0 for site in self.definition.basis)
        if partial and occupancy_seed is None:
            raise ApiError(
                f"{self.definition.id} contains partially occupied sites. Pass an explicit "
                "occupancy_seed to sample a finite supercell reproducibly.")
        if partial:
            from ..structure_builder.alloys import sample_partial_occupancy
            structure, _ = sample_partial_occupancy(
                self.definition, repeat, seed=int(occupancy_seed))
        else:
            structure = bulk(self.definition, repeat)
        if activate:
            self._lab.project.add_structure(structure, activate=True, log=True)
            self._lab.project.history.log[-1].label = (
                f"Build {self.definition.id} bulk cell")
        return SurfaceHandle(structure, self.definition, self._lab)

    def create_surface(self, size: Sequence[int] = (4, 4, 4),
                       vacuum_angstrom: float = 14.0,
                       orientation: Optional[Sequence[int]] = None,
                       activate: bool = True,
                       **kwargs) -> SurfaceHandle:
        """Build a slab and, by default, make it the project's active structure.

        Registering it means checkpoints, undo, the project tree and the
        interface all see the structure a script is working on, rather than the
        script operating on something the project knows nothing about.
        """
        miller = tuple(int(v) for v in (orientation or self.orientation))
        from ..structure_builder.reconstruction import ReconstructionNotImplemented

        try:
            st = make_surface(self.definition, miller, size=tuple(int(v) for v in size),
                              vacuum_A=vacuum_angstrom, **kwargs)
        except ReconstructionNotImplemented as exc:
            raise UnsupportedRequest(exc.reason, exc.as_dict()) from None
        st.info.setdefault("material_id", self.definition.id)
        reconstruction = kwargs.get("reconstruction")
        if activate:
            self._lab.project.add_structure(st, activate=True, log=True)
            self._lab.project.history.log[-1].label = (
                f"Build {self.definition.id} "
                f"({''.join(str(v) for v in miller)}) surface"
                + (f", {reconstruction}" if reconstruction else ""))
        return SurfaceHandle(st, self.definition, self._lab)

    def reconstructions(self, miller: Optional[Sequence[int]] = None) -> List[dict]:
        """Reconstructions declared for this material, each flagged supported or
        not by the generator registry rather than by the file's own claim."""
        from ..structure_builder.reconstruction import available_reconstructions

        return available_reconstructions(self.definition, miller)

    def interplanar_spacing(self, miller: Sequence[int]) -> float:
        return lattice_planes(self.definition, miller)


class AtomCollection:
    """Query helpers over the atoms of a structure."""

    def __init__(self, structure: Structure, lab: "Lab") -> None:
        self._s = structure
        self._lab = lab

    def __len__(self) -> int:
        return len(self._s)

    def __iter__(self):
        return iter(self._s)

    def __getitem__(self, atom_id: int) -> AtomView:
        return self._s.atom(int(atom_id))

    def selected(self) -> List[AtomView]:
        return self._lab.project.selection.atoms(self._s)

    def selected_one(self) -> AtomView:
        return self._lab.project.selection.one(self._s)

    def by_element(self, *symbols: str) -> Selection:
        return by_element(self._s, *symbols)

    def by_role(self, *roles: str) -> Selection:
        return by_role(self._s, *roles)

    def within(self, center: Sequence[float], radius_A: float) -> Selection:
        return within_radius(self._s, center, radius_A)

    def neighbors_of(self, atom_id: int, radius_A: float = 3.0) -> Selection:
        return neighbors_within(self._s, atom_id, radius_A)

    def in_box(self, lo: Sequence[float], hi: Sequence[float]) -> Selection:
        return in_box(self._s, lo, hi)

    def on_plane(self, miller: Sequence[int], offset_A: float,
                 tolerance_A: float = 0.5) -> Selection:
        return near_plane(self._s, miller, offset_A, tolerance_A)

    def table(self) -> dict:
        return _io.structure_table(self._s)


class MaterialsNamespace:
    def __init__(self, lab: "Lab") -> None:
        self._lab = lab

    def load(self, key: str, orientation: Optional[Union[str, Sequence[int]]] = None
             ) -> MaterialHandle:
        definition = self._lab.library.get(key)
        miller = None
        if orientation is not None:
            miller = (tuple(int(c) for c in str(orientation))
                      if isinstance(orientation, str) else tuple(int(v) for v in orientation))
        return MaterialHandle(definition, self._lab, miller)

    def list(self) -> List[str]:
        return self._lab.library.ids()

    def search(self, text: str) -> List[dict]:
        return [d.summary() for d in self._lab.library.search(text)]

    def summary(self, key: str) -> dict:
        return self._lab.library.get(key).summary()

    def add_from_file(self, path: str) -> str:
        import json
        d = self._lab.library.register_raw(json.loads(open(path).read()), path)
        return d.id

    def search_paths(self) -> List[str]:
        return [str(p) for p in self._lab.library.search_paths]


class ViewNamespace:
    """Where scripts hand results to the interface.

    Headless scripts still work: everything is collected in
    :attr:`items` and returned with the script result.
    """

    def __init__(self, lab: "Lab") -> None:
        self._lab = lab
        self.items: List[dict] = []
        self.sink: Optional[Callable[[dict], None]] = None

    def _emit(self, payload: dict) -> dict:
        """Record a display item and return a compact acknowledgement.

        The full payload goes to the interface; what comes back to the script
        is short, so echoing the result of ``view.display`` in a console does
        not print a page of metadata.
        """
        payload["t"] = time.time()
        self.items.append(payload)
        if self.sink:
            self.sink(payload)
        return {k: payload[k] for k in ("kind", "key", "title") if k in payload}

    def display(self, obj: Any, palette: str = "silver", title: str = "",
                channel: Optional[str] = None, **kwargs) -> dict:
        from ..microscopy.scan import ScanResult
        if isinstance(obj, ScanResult):
            key = self._lab.project.add_scan(obj)
            return self._emit({"kind": "scan", "key": key, "palette": palette,
                               "channel": channel or obj.primary_channel,
                               "title": title or f"{obj.technique} {obj.mode}",
                               "summary": obj.as_dict()})
        if isinstance(obj, Result):
            return self.show(obj, title=title)
        if isinstance(obj, (Structure, SurfaceHandle)):
            st = obj.structure if isinstance(obj, SurfaceHandle) else obj
            key = self._lab.project.add_structure(st, activate=True)
            return self._emit({"kind": "structure", "key": key,
                               "title": title or st.formula(),
                               "n_atoms": len(st)})
        if isinstance(obj, np.ndarray) and obj.ndim == 2:
            return self._emit({"kind": "image", "title": title or "array",
                               "shape": list(obj.shape), "palette": palette,
                               "data": obj.tolist()})
        return self._emit({"kind": "value", "title": title or type(obj).__name__,
                           "repr": repr(obj)[:4000]})

    show = display

    def plot(self, x: Sequence[float], y: Sequence[float], title: str = "",
             xlabel: str = "", ylabel: str = "", kind: str = "line") -> dict:
        return self._emit({"kind": "plot", "plot_type": kind, "title": title,
                           "xlabel": xlabel, "ylabel": ylabel,
                           "x": list(map(float, np.asarray(x).ravel())),
                           "y": list(map(float, np.asarray(y).ravel()))})

    def table(self, columns: Dict[str, Sequence[Any]], title: str = "") -> dict:
        return self._emit({"kind": "table", "title": title,
                           "columns": {k: list(v) for k, v in columns.items()}})

    def message(self, text: str, level: str = "info") -> dict:
        return self._emit({"kind": "message", "level": level, "text": str(text)})


class MicroscopeNamespace:
    def __init__(self, lab: "Lab") -> None:
        self._lab = lab

    @staticmethod
    def _structure(target) -> Structure:
        if isinstance(target, SurfaceHandle):
            return target.structure
        if isinstance(target, Structure):
            return target
        raise ApiError("Expected a surface or a structure")

    def _tb_for(self, structure: Structure, material: Optional[MaterialDefinition]):
        from ..solvers.tight_binding import MODELS, TightBinding, model_for_material
        from ..elements import periodic_table as pt
        name = None
        if material is not None:
            name = model_for_material(material.id)
        present = {pt.symbol(int(z)) for z in structure.numbers}
        if name is not None and not (present <= set(MODELS[name].species)):
            extra = sorted(present - set(MODELS[name].species))
            from ..solvers.tight_binding import HARRISON_TERM_VALUES
            if all(e in HARRISON_TERM_VALUES for e in extra):
                return TightBinding(name, impurities=extra)
            name = None
        if name is None:
            for key, model in MODELS.items():
                if present <= set(model.species):
                    name = key
                    break
        if name is None:
            from ..solvers.tight_binding import HARRISON_TERM_VALUES
            for key, model in MODELS.items():
                extra = sorted(present - set(model.species))
                if (present & set(model.species)) and all(
                        e in HARRISON_TERM_VALUES for e in extra):
                    return TightBinding(key, impurities=extra)
            covered = sorted({sym for m in MODELS.values() for sym in m.species})
            raise ApiError(
                "No tight-binding parameterisation covers "
                f"{', '.join(sorted(present))}, so the Tersoff-Hamann STM model "
                "cannot be evaluated. Parameterised species: "
                f"{', '.join(covered)} (models: {', '.join(sorted(MODELS))}). "
                "Use an external solver adapter for other materials."
            )
        return TightBinding(name)

    def stm_scan(self, target, bias_volts: float = 1.0, current_nA: float = 0.5,
                 resolution: Sequence[int] = (256, 256), mode: str = "constant-current",
                 tip: str = "W", noise: str = "realistic", seed: int = 0,
                 material: Optional[MaterialHandle] = None, **kwargs):
        from ..microscopy.noise import NoiseModel
        from ..microscopy.stm import STMSettings, STMSimulator
        from ..microscopy.tip import Tip
        structure = self._structure(target)
        mat = (material.definition if isinstance(material, MaterialHandle)
               else getattr(target, "material", None))
        solver = self._tb_for(structure, mat)
        wf = 4.85
        if mat is not None and mat.property("work_function"):
            wf = float(mat.property("work_function").value)
        noise_model = {"realistic": NoiseModel.realistic, "quiet": NoiseModel.quiet,
                       "noisy": NoiseModel.noisy}.get(noise)
        if noise_model is None:
            raise ApiError(f"Unknown noise preset {noise!r}; use quiet, realistic or noisy.")
        settings = STMSettings(bias_V=bias_volts, setpoint_nA=current_nA,
                               mode=mode, resolution=tuple(int(v) for v in resolution),
                               noise=noise_model(seed), **kwargs)
        sim = STMSimulator(solver, Tip(tip), sample_work_function_eV=wf)
        return sim.scan(structure, settings)

    def stm_spectroscopy(self, target, x: float, y: float, height_A: float = 5.0,
                         bias_range_V: Sequence[float] = (-2.0, 2.0), **kwargs):
        from ..microscopy.stm import STMSimulator
        from ..microscopy.tip import Tip
        structure = self._structure(target)
        mat = getattr(target, "material", None)
        sim = STMSimulator(self._tb_for(structure, mat), Tip("W"))
        z = float(structure.positions[:, 2].max()) + height_A
        return sim.spectroscopy(structure, x, y, z,
                                bias_range_V=tuple(bias_range_V), **kwargs)

    def afm_scan(self, target, mode: str = "fm-afm", height_A: float = 4.0,
                 resolution: Sequence[int] = (256, 256), tip: str = "W",
                 noise: str = "realistic", seed: int = 0, **kwargs):
        from ..microscopy.afm import AFMSettings, AFMSimulator
        from ..microscopy.noise import NoiseModel
        from ..microscopy.tip import Tip
        structure = self._structure(target)
        noise_model = {"realistic": NoiseModel.realistic, "quiet": NoiseModel.quiet,
                       "noisy": NoiseModel.noisy}[noise]
        settings = AFMSettings(mode=mode, height_A=height_A,
                               resolution=tuple(int(v) for v in resolution),
                               noise=noise_model(seed), **kwargs)
        return AFMSimulator(Tip(tip)).scan(structure, settings)

    def force_curve(self, target, x: float, y: float, **kwargs):
        from ..microscopy.afm import AFMSimulator
        from ..microscopy.tip import Tip
        return AFMSimulator(Tip("W")).force_curve(self._structure(target), x, y, **kwargs)


class DFTRelaxationFailed(ApiError):
    """A relaxation that was cancelled, timed out or failed.  Nothing was stored."""

    def __init__(self, status: str, reason: str, audit: Optional[dict] = None) -> None:
        super().__init__(f"DFT relaxation {status}: {reason}")
        self.status = status
        self.reason = reason
        self.audit = dict(audit or {})


class DFTDOSFailed(ApiError):
    """A DOS calculation that was cancelled, timed out or failed.  Nothing was stored."""

    def __init__(self, status: str, reason: str, audit: Optional[dict] = None) -> None:
        super().__init__(f"DFT DOS {status}: {reason}")
        self.status = status
        self.reason = reason
        self.audit = dict(audit or {})


class DFTBandsFailed(ApiError):
    """A band structure that was cancelled, timed out, failed or did not verify.  Nothing was
    stored."""

    def __init__(self, status: str, reason: str, audit: Optional[dict] = None) -> None:
        super().__init__(f"DFT band structure {status}: {reason}")
        self.status = status
        self.reason = reason
        self.audit = dict(audit or {})


class DFTEOSFailed(ApiError):
    """An equation of state that was cancelled, failed or whose fit did not verify.  Nothing
    was stored."""

    def __init__(self, status: str, reason: str) -> None:
        super().__init__(f"DFT equation of state {status}: {reason}")
        self.status = status
        self.reason = reason


class DFTLDOSFailed(ApiError):
    """An LDOS calculation that was cancelled, timed out, failed or did not verify.  Nothing was
    stored."""

    def __init__(self, status: str, reason: str, audit: Optional[dict] = None) -> None:
        super().__init__(f"DFT LDOS {status}: {reason}")
        self.status = status
        self.reason = reason
        self.audit = dict(audit or {})


class DFTNamespace:
    """First-principles calculations through GPAW.

    GPAW is optional and runs in its own interpreter.  ``status()`` says
    whether it is usable and, when it is not, which of the two independent
    pieces is missing: the code or the PAW datasets.

    The ground-state experiment is the full interface: ``experiment()``
    builds an immutable, versioned specification of every variable,
    ``check()`` lists every refusal, ``ground_state()`` runs it and stores
    each observable with its provenance, and ``convergence_study()`` measures
    how much a result still depends on one numerical parameter.  ``energy()``
    is the earlier single-point adapter, kept for existing scripts; it
    refuses charged and spin-polarised systems, which the experiment handles.

    ``relax_spec()``, ``relax_check()`` and ``relax()`` are the relaxation
    experiment: the same ground-state variables plus the ionic search, fixed
    cell or variable cell.  ``relax_state()`` and ``relax_apply()`` give it
    the same current, stale and detached ownership as the other solvers, and
    applying it is one undoable change.

    ``dos_spec()``, ``dos_check()`` and ``dos()`` are the density of states
    and projected density of states, of a structure or of a stored converged
    ground state or relaxation; ``dos_array()`` returns the curves and
    ``dos_state()`` whether the DOS still describes its structure.

    ``bands_spec()``, ``bands_check()`` and ``bands()`` are the Kohn-Sham band
    structure along an explicit reciprocal-space path, of a structure or of a
    stored converged ground state or relaxation; ``bands_array()`` returns the
    eigenvalues, k-points and path distance and ``bands_state()`` whether the
    bands are current, stale, detached or corrupt.

    ``eos_spec()``, ``eos_check()`` and ``eos()`` fit a Birch-Murnaghan
    equation of state to ground states at several volumes, all with the same
    pinned k-point grid and cutoff; ``eos_apply()`` scales the structure to the
    fitted equilibrium volume as one undoable change.

    ``ldos_spec()``, ``ldos_check()`` and ``ldos()`` compute the spatially
    resolved local density of states in an energy window about the Fermi
    level; ``stm_image()`` takes a Tersoff-Hamann constant-height or
    constant-current image from a stored slab LDOS.

    Example::

        h2 = ...                                        # a molecule in a box
        spec = dft.experiment(h2, xc="PBE", grid_spacing_A=0.18)
        dft.check(spec)["blocking"]                     # [] when it may run
        run = dft.ground_state(spec=spec)
        run["energy"].value, run["charge_accounting"].value["balanced"]
        dft.array(name="density").data.shape
    """

    def __init__(self, lab: "Lab") -> None:
        self._lab = lab

    def _solver(self):
        from ..solvers.gpaw_driver import GPAWSolver

        return GPAWSolver()

    def status(self) -> dict:
        """What GPAW Materia can see, and what is missing if it cannot run."""
        from ..solvers.gpaw_driver import discover

        return discover().as_dict()

    def available(self) -> bool:
        return bool(self.status()["operational"])

    def presets(self) -> List[dict]:
        """Conservative starting configurations, each already validated."""
        from ..solvers.gpaw_driver import describe_presets

        return describe_presets()

    def settings(self, preset_name: str = "molecule", target=None,
                 **overrides) -> dict:
        """Build and validate a configuration without running anything."""
        from ..solvers.gpaw_driver import preset

        structure = target.structure if isinstance(target, SurfaceHandle) else target
        return preset(preset_name, structure, **overrides).as_dict()

    def energy(self, target, settings: Optional[dict] = None,
               preset_name: Optional[str] = None,
               progress: Optional[Callable[[dict], None]] = None,
               cancelled: Optional[Callable[[], bool]] = None,
               timeout_s: Optional[float] = None,
               apply_forces: bool = False, **overrides) -> Result:
        """One self-consistent total energy, with the forces that go with it.

        Returns the energy :class:`Result`.  A calculation that did not
        converge, was cancelled or failed comes back as an unsupported result
        carrying the reason, never as a number.  The structure is not touched
        unless ``apply_forces`` is set and the calculation converged, and then
        only through the project's own undo machinery.
        """
        from ..solvers.gpaw_driver import preset, validate

        structure = target.structure if isinstance(target, SurfaceHandle) else target
        if settings is None:
            settings = preset(preset_name or "molecule", structure, **overrides).as_dict()
        else:
            merged = dict(settings)
            merged.update(overrides)
            settings = validate(merged, structure).as_dict()

        import time
        import uuid

        from ..solvers.gpaw_driver.conversion import (
            from_worker_spec, input_digest, to_worker_spec,
        )

        project = self._lab.project
        spec = to_worker_spec(structure)
        submitted_digest = input_digest(spec, settings)
        run_id = f"{int(time.time() * 1000):013d}-{uuid.uuid4().hex[:6]}"
        structure_key = project.key_of(structure)

        solver = self._solver()
        out = solver.single_point(from_worker_spec(spec), settings=settings,
                                  progress=progress, cancelled=cancelled,
                                  timeout_s=timeout_s)
        for result in out.results.values():
            result.provenance.inputs_digest = submitted_digest
            result.extra["run_id"] = run_id
        for key, result in out.results.items():
            project.add_result(f"gpaw::{run_id}::{key}", result)
        energy = out.results["energy"]
        if apply_forces and energy.supported and "forces" in out.results:
            self._apply(structure, structure_key, out.results["forces"], settings,
                        spec, submitted_digest, run_id)
        return energy

    def _apply(self, structure: Structure, structure_key: Optional[str],
               forces: Result, settings: dict, spec: dict,
               submitted_digest: str, run_id: str) -> None:
        """Write converged forces onto the structure they were computed for.

        Refuses, loudly, if that structure has changed since the calculation
        started: forces belong to one geometry and attaching them to another
        would be a fabrication.
        """
        from ..core_model.selection import Selection
        from ..solvers.gpaw_driver.conversion import (
            apply_forces, describe_difference, input_digest, to_worker_spec,
        )

        project = self._lab.project
        now = to_worker_spec(structure)
        if input_digest(now, settings) != submitted_digest:
            difference = describe_difference(spec, now) or "the input changed"
            raise ApiError(
                f"The GPAW forces were not applied: {difference} while the "
                "calculation was running, so they were computed for a different "
                f"structure. The energy is kept as gpaw::{run_id}::energy. Re-run "
                "on the current geometry to get forces for it.")
        before = structure.copy()
        before_selection = Selection(list(project.selection.ids), project.selection.query)
        apply_forces(structure, forces.extra["by_atom_id"])
        project.record_change(
            structure, before, before_selection,
            "GPAW forces", "solver.gpaw.forces",
            {"xc": settings.get("xc"), "mode": settings.get("mode"),
             "run_id": run_id, "inputs_digest": submitted_digest},
            result_summary=f"max |F| = {forces.extra['max_force_eV_A']:.4f} eV/A")


    def _structure(self, target=None) -> Structure:
        if target is None:
            structure = self._lab.project.structure
            if structure is None:
                raise ApiError("No active structure.")
            return structure
        return target.structure if isinstance(target, SurfaceHandle) else target

    def experiment(self, target=None, **variables):
        """A ground-state specification for a structure, with safe defaults.

        Variables are the settable fields of
        :class:`~materia.experiments.dft.spec.GroundStateSpec`, for example
        ``xc``, ``representation``, ``grid_spacing_A``, ``cutoff_eV``,
        ``kpoints``, ``charge_e``, ``spin_polarized``,
        ``initial_magnetic_moments_muB``, ``occupations``, ``smearing_eV``,
        ``external_field_V_per_A`` and ``observables``.  The specification is
        not checked here; :meth:`check` says whether it may run.
        """
        from ..experiments.dft import spec as specs

        structure = self._structure(target)
        try:
            return specs.build(structure, structure_key=self._lab.project.key_of(structure),
                               **variables)
        except specs.SpecError as exc:
            raise ApiError(str(exc)) from None

    def change(self, spec, **variables):
        """A new specification with some variables changed."""
        from ..experiments.dft import spec as specs

        try:
            return specs.changed(spec, **variables)
        except specs.SpecError as exc:
            raise ApiError(str(exc)) from None

    def check(self, spec) -> dict:
        """Every refusal and warning for a specification, keyed by variable."""
        from ..experiments.dft import spec as specs

        return specs.check(spec).as_dict()

    def describe(self, spec) -> dict:
        """Every variable with its unit, explanation, value and any problem."""
        from ..experiments.dft import spec as specs

        return specs.describe(spec, specs.check(spec))

    def ground_state(self, target=None, spec=None, *,
                     progress: Optional[Callable[[dict], None]] = None,
                     cancelled: Optional[Callable[[], bool]] = None,
                     timeout_s: Optional[float] = None,
                     reuse_restart: bool = False, keep_restart: bool = False,
                     run_id: Optional[str] = None, project: Optional[Project] = None,
                     **variables) -> Dict[str, Result]:
        """Run one self-consistent ground-state calculation and store it.

        Returns the results keyed by quantity: ``run`` (the record, always
        present), and for a converged run ``energy``, ``forces``,
        ``charge_accounting``, ``scf_history`` and whichever other
        observables were requested.  A refused specification raises
        :class:`ApiError` and stores nothing.  A run that does not converge,
        fails or is cancelled stores only its record, with origin
        ``unsupported`` and the reason.  The structure is never changed and
        no undo point is recorded.
        """
        from ..experiments.dft import records
        from ..experiments.dft import spec as specs
        from ..experiments.dft.run import execute

        project = project or self._lab.project
        if spec is None:
            spec = self.experiment(target, **variables)
        elif variables:
            spec = self.change(spec, **variables)
        try:
            outcome = execute(spec, progress=progress, cancelled=cancelled,
                              timeout_s=timeout_s, reuse_restart=reuse_restart,
                              keep_restart=keep_restart, run_id=run_id)
        except specs.SpecRefused as exc:
            raise ApiError(str(exc)) from None
        records.store(project, outcome)
        return dict(outcome.results)

    def runs(self) -> List[dict]:
        """Every stored ground-state run, newest first, with its current state."""
        from ..experiments.dft import records

        project = self._lab.project
        return [records.summary(project, run_id) for run_id in records.run_ids(project)]

    def _run_id(self, run_id: Optional[str]) -> str:
        from ..experiments.dft import records

        ids = records.run_ids(self._lab.project)
        if not ids:
            raise ApiError("No DFT ground-state run is stored in this project.")
        if run_id is None:
            return ids[0]
        if run_id not in ids:
            raise ApiError(f"No DFT ground-state run {run_id}.")
        return run_id

    def result(self, run_id: Optional[str] = None, quantity: str = "energy") -> Result:
        """One stored quantity of a run (the newest when ``run_id`` is omitted)."""
        from ..experiments.dft import records

        run_id = self._run_id(run_id)
        item = self._lab.project.results.get(records.key(run_id, quantity))
        if item is None:
            raise ApiError(f"Run {run_id} has no {quantity!r}. It stored: " + ", ".join(
                sorted(k.split("::")[2] for k in self._lab.project.results
                       if k.startswith(records.key(run_id, "")))))
        return item

    def array(self, run_id: Optional[str] = None, name: str = "density"):
        """A stored array of a run: density, spin_density, electrostatic_potential,
        eigenvalues or occupations, as a
        :class:`~materia.project_format.arrays.StoredArray`."""
        from ..experiments.dft import records

        run_id = self._run_id(run_id)
        stored = self._lab.project.arrays.get(records.key(run_id, name))
        if stored is None:
            raise ApiError(f"Run {run_id} stored no {name!r} array.")
        return stored

    def spec_of(self, run_id: Optional[str] = None):
        from ..experiments.dft import records

        return records.spec_of(self._lab.project, self._run_id(run_id))

    def state(self, run_id: Optional[str] = None, target=None) -> dict:
        """Whether a run is current, stale or detached, and why."""
        from ..experiments.dft import records

        structure = None if target is None else self._structure(target)
        return records.status(self._lab.project, self._run_id(run_id), structure)

    def is_current(self, run_id: Optional[str] = None, target=None) -> bool:
        return bool(self.state(run_id, target).get("current"))

    def convergence_study(self, target=None, parameter: str = "grid_spacing_A",
                          values: Sequence[Any] = (), observable: str = "energy_per_atom_eV",
                          tolerance: float = 1.0e-3, spec=None, *,
                          progress: Optional[Callable[[float, str], None]] = None,
                          cancelled: Optional[Callable[[], bool]] = None,
                          project: Optional[Project] = None, **variables) -> Result:
        """Vary one numerical parameter and measure how much the observable moves.

        Every point is a full ground-state run and is stored as one.  Returns
        the study record, stored as ``dftstudy::<study_id>``, whose value
        holds each point, the successive changes, the residual between the
        two most accurate points and whether it is within ``tolerance``.
        """
        from ..experiments.dft import convergence, records

        project = project or self._lab.project
        if spec is None:
            spec = self.experiment(target, **variables)
        elif variables:
            spec = self.change(spec, **variables)
        study = convergence.StudySpec(base=spec, parameter=parameter,
                                      values=tuple(values), observable=observable,
                                      tolerance=float(tolerance))
        try:
            convergence.plan(study)
        except convergence.StudyError as exc:
            raise ApiError(str(exc)) from None

        def keep(run, point) -> None:
            records.store(project, run)

        outcome = convergence.run_study(study, progress=progress, cancelled=cancelled,
                                        on_run=keep)
        record = outcome.record()
        project.add_result(f"{records.STUDY_PREFIX}{outcome.study_id}", record)
        return record

    def studies(self) -> List[dict]:
        from ..experiments.dft import records

        out = []
        for key, item in sorted(self._lab.project.results.items(), reverse=True):
            if key.startswith(records.STUDY_PREFIX):
                value = item.value or {}
                out.append({"study_id": item.extra.get("study_id"),
                            "parameter": value.get("parameter"),
                            "observable": value.get("observable"),
                            "tolerance_met": value.get("tolerance_met"),
                            "residual": value.get("residual"),
                            "status": item.extra.get("status"),
                            "run_ids": item.extra.get("run_ids")})
        return out

    def evidence(self, run_id: Optional[str] = None) -> List[dict]:
        """Convergence studies that show a run converged, parameter by parameter."""
        from ..experiments.dft import convergence, records

        spec = records.spec_of(self._lab.project, self._run_id(run_id))
        studies = [v for k, v in self._lab.project.results.items()
                   if k.startswith(records.STUDY_PREFIX)]
        return convergence.evidence(spec, studies)

    def relax_spec(self, target=None, **variables):
        """A relaxation specification for a structure, with safe defaults.

        Relaxation variables are ``mode`` (``"fixed-cell"`` or
        ``"variable-cell"``), ``optimizer`` (``"BFGS"`` or ``"FIRE"``),
        ``fmax_eV_A``, ``max_steps``, ``maxstep_A``, ``symmetry``
        (``"preserve"`` or ``"off"``) and, for a variable cell,
        ``stress_tol_eV_A3``, ``cell_mask``, ``hydrostatic_strain`` and
        ``target_pressure_GPa``.  Every other variable is a ground-state
        variable of :meth:`experiment`.  Fixed atoms come from the structure.
        """
        from ..experiments.dft import relaxation
        from ..experiments.dft import spec as specs

        structure = self._structure(target)
        try:
            return relaxation.build(structure,
                                    structure_key=self._lab.project.key_of(structure),
                                    **variables)
        except specs.SpecError as exc:
            raise ApiError(str(exc)) from None

    def relax_change(self, spec, **variables):
        from ..experiments.dft import relaxation
        from ..experiments.dft import spec as specs

        try:
            return relaxation.changed(spec, **variables)
        except specs.SpecError as exc:
            raise ApiError(str(exc)) from None

    def relax_check(self, spec) -> dict:
        from ..experiments.dft import relaxation

        return relaxation.check(spec).as_dict()

    def relax_describe(self, spec) -> dict:
        from ..experiments.dft import relaxation

        return relaxation.describe(spec)

    def relax(self, target=None, spec=None, *,
              progress: Optional[Callable[[dict], None]] = None,
              cancelled: Optional[Callable[[], bool]] = None,
              timeout_s: Optional[float] = None, apply: bool = True,
              run_id: Optional[str] = None, project: Optional[Project] = None,
              **variables) -> Dict[str, Result]:
        """Relax a structure with GPAW and store the result.

        Returns the stored results keyed by quantity: ``relaxation`` (the
        summary: history, energies, displacement, cell change, convergence),
        ``run``, and the ground-state observables at the final geometry
        (``energy``, ``forces``, ``stress`` for a variable cell,
        ``charge_accounting``, ``scf_history`` and whichever others were
        requested).  A refused specification raises :class:`ApiError`; a
        cancelled, timed-out or failed run raises :class:`DFTRelaxationFailed`.
        Neither stores anything, records history or changes the structure.
        With ``apply`` set, a converged relaxation is written onto its
        structure as one undoable change if that structure is unchanged.
        """
        from ..experiments.dft import records, relaxation
        from ..experiments.dft import spec as specs

        project = project or self._lab.project
        if spec is None:
            spec = self.relax_spec(target, **variables)
        elif variables:
            spec = self.relax_change(spec, **variables)
        try:
            outcome = relaxation.execute(spec, progress=progress, cancelled=cancelled,
                                         timeout_s=timeout_s, run_id=run_id)
        except specs.SpecRefused as exc:
            raise ApiError(f"DFT relaxation refused: {exc}") from None
        if not outcome.ok:
            raise DFTRelaxationFailed(outcome.status, outcome.reason, outcome.audit)
        records.store_relaxation(project, outcome, apply=apply)
        return dict(outcome.results)

    def relax_runs(self) -> List[dict]:
        from ..experiments.dft import records

        project = self._lab.project
        return [records.relax_summary(project, r) for r in records.relax_run_ids(project)]

    def _relax_id(self, run_id: Optional[str]) -> str:
        from ..experiments.dft import records

        ids = records.relax_run_ids(self._lab.project)
        if not ids:
            raise ApiError("No DFT relaxation is stored in this project.")
        if run_id is None:
            return ids[0]
        if run_id not in ids:
            raise ApiError(f"No DFT relaxation {run_id}.")
        return run_id

    def relax_result(self, run_id: Optional[str] = None,
                     quantity: str = "relaxation") -> Result:
        from ..experiments.dft import records

        run_id = self._relax_id(run_id)
        item = self._lab.project.results.get(records.relax_key(run_id, quantity))
        if item is None:
            raise ApiError(f"Relaxation {run_id} has no {quantity!r}.")
        return item

    def relax_array(self, run_id: Optional[str] = None, name: str = "final_positions"):
        """A stored array: initial_positions, final_positions, initial_cell, final_cell,
        final_forces, step_positions or step_cells."""
        from ..experiments.dft import records

        run_id = self._relax_id(run_id)
        stored = self._lab.project.arrays.get(records.relax_key(run_id, name))
        if stored is None:
            raise ApiError(f"Relaxation {run_id} stored no {name!r} array.")
        return stored

    def relax_spec_of(self, run_id: Optional[str] = None):
        from ..experiments.dft import records

        return records.relax_spec_of(self._lab.project, self._relax_id(run_id))

    def relax_state(self, run_id: Optional[str] = None, target=None) -> dict:
        """Whether a relaxation is current, stale or detached, and whether it can be applied."""
        from ..experiments.dft import records

        structure = None if target is None else self._structure(target)
        return records.relax_status(self._lab.project, self._relax_id(run_id), structure)

    def relax_apply(self, run_id: Optional[str] = None) -> str:
        """Write a stored relaxation onto its structure as one undoable change."""
        from ..experiments.dft import records

        try:
            return records.apply_relaxation(self._lab.project, self._relax_id(run_id))
        except records.ApplyRefused as exc:
            raise ApiError(f"Not applied: {exc}") from None


    def dos_spec(self, target=None, *, source_run: Optional[str] = None,
                 source_kind: str = "ground-state", **variables):
        """A DOS specification: of a structure, or of a stored converged run.

        With ``source_run`` the geometry and every electronic setting come
        from that ground-state run or, with ``source_kind="relaxation"``, from
        that relaxation's final geometry, and only DOS variables may be given:
        ``energy_reference``, ``energy_min_eV``, ``energy_max_eV``,
        ``energy_step_eV``, ``broadening``, ``width_eV``, ``spin_channels``,
        ``dos_kpoints``, ``dos_kpoints_gamma_centered``, ``n_bands`` and
        ``projections`` (a list such as ``[{"element": "Si", "angular": "p"}]``).
        """
        from ..experiments.dft import dos
        from ..experiments.dft import spec as specs

        try:
            if source_run is not None:
                return dos.build_from_run(self._lab.project, source_kind, source_run,
                                          **variables)
            structure = self._structure(target)
            return dos.build(structure, structure_key=self._lab.project.key_of(structure),
                             **variables)
        except specs.SpecError as exc:
            raise ApiError(str(exc)) from None

    def dos_change(self, spec, **variables):
        from ..experiments.dft import dos
        from ..experiments.dft import spec as specs

        try:
            return dos.changed(spec, **variables)
        except specs.SpecError as exc:
            raise ApiError(str(exc)) from None

    def dos_check(self, spec) -> dict:
        from ..experiments.dft import dos

        return dos.check(spec).as_dict()

    def dos_describe(self, spec) -> dict:
        from ..experiments.dft import dos

        return dos.describe(spec)

    def dos(self, target=None, spec=None, *,
            progress: Optional[Callable[[dict], None]] = None,
            cancelled: Optional[Callable[[], bool]] = None,
            timeout_s: Optional[float] = None, run_id: Optional[str] = None,
            project: Optional[Project] = None, **variables) -> Dict[str, Result]:
        """Compute the DOS and PDOS with GPAW and store them.

        Returns the stored results keyed by quantity: ``dos`` (the summary:
        grid, reference, Fermi level, projections, normalisation and
        integration checks), ``fermi_level``, ``energy``, ``charge_accounting``,
        ``scf_history`` and ``run``.  The curves are arrays; see
        :meth:`dos_array`.  A refused specification raises :class:`ApiError`;
        a cancelled, timed-out or failed run raises :class:`DFTDOSFailed`.
        Neither stores anything or records history.  No structure is changed.
        """
        from ..experiments.dft import dos, records
        from ..experiments.dft import spec as specs

        project = project or self._lab.project
        if spec is None:
            spec = self.dos_spec(target, **variables)
        elif variables:
            spec = self.dos_change(spec, **variables)
        try:
            outcome = dos.execute(spec, progress=progress, cancelled=cancelled,
                                  timeout_s=timeout_s, run_id=run_id)
        except specs.SpecRefused as exc:
            raise ApiError(f"DFT DOS refused: {exc}") from None
        if not outcome.ok:
            raise DFTDOSFailed(outcome.status, outcome.reason, outcome.audit)
        records.store_dos(project, outcome)
        return dict(outcome.results)

    def dos_runs(self) -> List[dict]:
        from ..experiments.dft import records

        project = self._lab.project
        return [records.dos_summary(project, r) for r in records.dos_run_ids(project)]

    def _dos_id(self, run_id: Optional[str]) -> str:
        from ..experiments.dft import records

        ids = records.dos_run_ids(self._lab.project)
        if not ids:
            raise ApiError("No DOS is stored in this project.")
        if run_id is None:
            return ids[0]
        if run_id not in ids:
            raise ApiError(f"No DOS {run_id}.")
        return run_id

    def dos_result(self, run_id: Optional[str] = None, quantity: str = "dos") -> Result:
        from ..experiments.dft import records

        run_id = self._dos_id(run_id)
        item = self._lab.project.results.get(records.dos_key(run_id, quantity))
        if item is None:
            raise ApiError(f"DOS {run_id} has no {quantity!r}.")
        return item

    def dos_array(self, run_id: Optional[str] = None, name: str = "dos_total"):
        """A stored array: energies, dos_total, dos_spin, pdos, pdos_spin, eigenvalues or
        kpoint_weights."""
        from ..experiments.dft import records

        run_id = self._dos_id(run_id)
        stored = self._lab.project.arrays.get(records.dos_key(run_id, name))
        if stored is None:
            raise ApiError(f"DOS {run_id} stored no {name!r} array.")
        return stored

    def dos_spec_of(self, run_id: Optional[str] = None):
        from ..experiments.dft import records

        return records.dos_spec_of(self._lab.project, self._dos_id(run_id))

    def dos_state(self, run_id: Optional[str] = None, target=None) -> dict:
        """Whether a DOS is current, stale or detached, and why."""
        from ..experiments.dft import records

        structure = None if target is None else self._structure(target)
        return records.dos_status(self._lab.project, self._dos_id(run_id), structure)


    def bands_spec(self, target=None, *, source_run: Optional[str] = None,
                   source_kind: str = "ground-state", **variables):
        """A band-structure specification: of a structure, or of a stored converged run.

        The standard path of the lattice is generated once, here, and stored.
        Band-structure variables: ``energy_reference``, ``scf_symmetry``,
        ``path`` (such as ``"GXWKGL,UX"``, or ``"standard"`` to regenerate),
        ``special_points`` (label to fractional coordinates),
        ``sampling_density_per_invA``, ``segment_intervals``, ``n_bands`` and
        ``extra_bands``.  With ``source_run`` only these may be given.
        """
        from ..experiments.dft import bands
        from ..experiments.dft import spec as specs

        try:
            if source_run is not None:
                return bands.build_from_run(self._lab.project, source_kind, source_run,
                                            **variables)
            structure = self._structure(target)
            return bands.build(structure, structure_key=self._lab.project.key_of(structure),
                               **variables)
        except specs.SpecError as exc:
            raise ApiError(str(exc)) from None

    def bands_change(self, spec, **variables):
        from ..experiments.dft import bands
        from ..experiments.dft import spec as specs

        try:
            return bands.changed(spec, **variables)
        except specs.SpecError as exc:
            raise ApiError(str(exc)) from None

    def bands_check(self, spec) -> dict:
        from ..experiments.dft import bands

        return bands.check(spec).as_dict()

    def bands_describe(self, spec) -> dict:
        from ..experiments.dft import bands

        return bands.describe(spec)

    def bands(self, target=None, spec=None, *,
              progress: Optional[Callable[[dict], None]] = None,
              cancelled: Optional[Callable[[], bool]] = None,
              timeout_s: Optional[float] = None, run_id: Optional[str] = None,
              project: Optional[Project] = None, **variables) -> Dict[str, Result]:
        """Compute the band structure with GPAW and store it.

        Returns the stored results keyed by quantity: ``bands`` (the summary:
        path, labels, breaks, Fermi level, band edges along the path, checks
        and array checksums), ``fermi_level``, ``energy``,
        ``charge_accounting``, ``scf_history`` and ``run``.  The eigenvalues
        are arrays; see :meth:`bands_array`.  A refused specification raises
        :class:`ApiError`; a cancelled, timed-out, failed or unverifiable run
        raises :class:`DFTBandsFailed`.  Neither stores anything.
        """
        from ..experiments.dft import bands, records
        from ..experiments.dft import spec as specs

        project = project or self._lab.project
        if spec is None:
            spec = self.bands_spec(target, **variables)
        elif variables:
            spec = self.bands_change(spec, **variables)
        try:
            outcome = bands.execute(spec, progress=progress, cancelled=cancelled,
                                    timeout_s=timeout_s, run_id=run_id)
        except specs.SpecRefused as exc:
            raise ApiError(f"DFT band structure refused: {exc}") from None
        if not outcome.ok:
            raise DFTBandsFailed(outcome.status, outcome.reason, outcome.audit)
        records.store_bands(project, outcome)
        return dict(outcome.results)

    def bands_runs(self) -> List[dict]:
        from ..experiments.dft import records

        project = self._lab.project
        return [records.bands_summary(project, r) for r in records.bands_run_ids(project)]

    def _bands_id(self, run_id: Optional[str]) -> str:
        from ..experiments.dft import records

        ids = records.bands_run_ids(self._lab.project)
        if not ids:
            raise ApiError("No band structure is stored in this project.")
        if run_id is None:
            return ids[0]
        if run_id not in ids:
            raise ApiError(f"No band structure {run_id}.")
        return run_id

    def bands_result(self, run_id: Optional[str] = None, quantity: str = "bands") -> Result:
        from ..experiments.dft import records

        run_id = self._bands_id(run_id)
        item = self._lab.project.results.get(records.bands_key(run_id, quantity))
        if item is None:
            raise ApiError(f"Band structure {run_id} has no {quantity!r}.")
        return item

    def bands_array(self, run_id: Optional[str] = None, name: str = "eigenvalues"):
        """A stored array, verified against its recorded checksum: eigenvalues,
        kpoints_frac, kpoints_cartesian or distance."""
        from ..experiments.dft import records

        run_id = self._bands_id(run_id)
        damage = records.bands_integrity(self._lab.project, run_id)
        if damage:
            raise ApiError(f"Band structure {run_id} refused: {damage}")
        stored = self._lab.project.arrays.get(records.bands_key(run_id, name))
        if stored is None:
            raise ApiError(f"Band structure {run_id} stored no {name!r} array.")
        return stored

    def bands_spec_of(self, run_id: Optional[str] = None):
        from ..experiments.dft import records

        return records.bands_spec_of(self._lab.project, self._bands_id(run_id))

    def bands_state(self, run_id: Optional[str] = None, target=None) -> dict:
        """Whether a band structure is current, stale, detached or corrupt, and why."""
        from ..experiments.dft import records

        structure = None if target is None else self._structure(target)
        return records.bands_status(self._lab.project, self._bands_id(run_id), structure)


    def eos_spec(self, target=None, *, source_run: Optional[str] = None,
                 source_kind: str = "ground-state", **variables):
        """An equation-of-state specification: of a structure, or around a stored run.

        Variables: ``volume_min_scale``, ``volume_max_scale`` and ``n_points``
        (volumes as fractions of the reference volume), plus, for a structure,
        every ground-state variable. The k-point grid and cutoff of the
        reference are used at every volume.
        """
        from ..experiments.dft import eos
        from ..experiments.dft import spec as specs

        try:
            if source_run is not None:
                return eos.build_from_run(self._lab.project, source_kind, source_run,
                                          **variables)
            structure = self._structure(target)
            return eos.build(structure, structure_key=self._lab.project.key_of(structure),
                             **variables)
        except specs.SpecError as exc:
            raise ApiError(str(exc)) from None

    def eos_change(self, spec, **variables):
        from ..experiments.dft import eos
        from ..experiments.dft import spec as specs

        try:
            return eos.changed(spec, **variables)
        except specs.SpecError as exc:
            raise ApiError(str(exc)) from None

    def eos_check(self, spec) -> dict:
        from ..experiments.dft import eos

        return eos.check(spec).as_dict()

    def eos(self, target=None, spec=None, *,
            progress: Optional[Callable[[float, str], None]] = None,
            cancelled: Optional[Callable[[], bool]] = None,
            timeout_s: Optional[float] = None, run_id: Optional[str] = None,
            project: Optional[Project] = None, **variables) -> Result:
        """Compute and store an equation of state; returns its ``eos`` result.

        The value holds V0, B0, B', E0, every point and the checks. A refused
        specification raises :class:`ApiError`; a cancelled, failed or
        unverified run raises :class:`DFTEOSFailed`. Neither stores anything.
        """
        from ..experiments.dft import eos, records
        from ..experiments.dft import spec as specs

        project = project or self._lab.project
        if spec is None:
            spec = self.eos_spec(target, **variables)
        elif variables:
            spec = self.eos_change(spec, **variables)
        try:
            outcome = eos.execute(spec, progress=progress, cancelled=cancelled,
                                  timeout_s=timeout_s, run_id=run_id)
        except specs.SpecRefused as exc:
            raise ApiError(f"DFT equation of state refused: {exc}") from None
        if not outcome.ok:
            raise DFTEOSFailed(outcome.status, outcome.reason)
        records.store_eos(project, outcome)
        return outcome.results["eos"]

    def eos_runs(self) -> List[dict]:
        from ..experiments.dft import records

        project = self._lab.project
        return [records.eos_summary(project, r) for r in records.eos_run_ids(project)]

    def _eos_id(self, run_id: Optional[str]) -> str:
        from ..experiments.dft import records

        ids = records.eos_run_ids(self._lab.project)
        if not ids:
            raise ApiError("No equation of state is stored in this project.")
        if run_id is None:
            return ids[0]
        if run_id not in ids:
            raise ApiError(f"No equation of state {run_id}.")
        return run_id

    def eos_result(self, run_id: Optional[str] = None) -> Result:
        from ..experiments.dft import records

        return records.eos_record(self._lab.project, self._eos_id(run_id))

    def eos_array(self, run_id: Optional[str] = None, name: str = "energies"):
        """A stored array, verified against its checksum: scales, volumes, energies,
        fit_pressures, stress_pressures or max_forces."""
        from ..experiments.dft import records

        run_id = self._eos_id(run_id)
        damage = records.eos_integrity(self._lab.project, run_id)
        if damage:
            raise ApiError(f"Equation of state {run_id} refused: {damage}")
        stored = self._lab.project.arrays.get(records.eos_key(run_id, name))
        if stored is None:
            raise ApiError(f"Equation of state {run_id} stored no {name!r} array.")
        return stored

    def eos_spec_of(self, run_id: Optional[str] = None):
        from ..experiments.dft import records

        return records.eos_spec_of(self._lab.project, self._eos_id(run_id))

    def eos_state(self, run_id: Optional[str] = None, target=None) -> dict:
        """Current, applied, stale, detached or corrupt, and why."""
        from ..experiments.dft import records

        structure = None if target is None else self._structure(target)
        return records.eos_status(self._lab.project, self._eos_id(run_id), structure)

    def eos_apply(self, run_id: Optional[str] = None) -> str:
        """Scale the structure to the fitted equilibrium volume as one undoable change."""
        from ..experiments.dft import records

        try:
            return records.apply_eos(self._lab.project, self._eos_id(run_id))
        except records.ApplyRefused as exc:
            raise ApiError(f"Not applied: {exc}") from None


    def ldos_spec(self, target=None, *, source_run: Optional[str] = None,
                  source_kind: str = "ground-state", **variables):
        """An LDOS specification: of a structure, or of a stored converged run.

        Variables: ``energy_min_eV`` and ``energy_max_eV`` (the window about the
        Fermi level; a sample bias V images 0 to V for V > 0 and V to 0 for
        V < 0), ``spin_channels`` and ``n_bands``, plus, for a structure, every
        ground-state variable.
        """
        from ..experiments.dft import ldos
        from ..experiments.dft import spec as specs

        try:
            if source_run is not None:
                return ldos.build_from_run(self._lab.project, source_kind, source_run,
                                           **variables)
            structure = self._structure(target)
            return ldos.build(structure, structure_key=self._lab.project.key_of(structure),
                              **variables)
        except specs.SpecError as exc:
            raise ApiError(str(exc)) from None

    def ldos_change(self, spec, **variables):
        from ..experiments.dft import ldos
        from ..experiments.dft import spec as specs

        try:
            return ldos.changed(spec, **variables)
        except specs.SpecError as exc:
            raise ApiError(str(exc)) from None

    def ldos_check(self, spec) -> dict:
        from ..experiments.dft import ldos

        return ldos.check(spec).as_dict()

    def ldos(self, target=None, spec=None, *,
             progress: Optional[Callable[[dict], None]] = None,
             cancelled: Optional[Callable[[], bool]] = None,
             timeout_s: Optional[float] = None, run_id: Optional[str] = None,
             project: Optional[Project] = None, **variables) -> Dict[str, Result]:
        """Compute the spatial LDOS with GPAW and store it.

        Returns the stored results keyed by quantity; ``ldos`` holds the window,
        state count, map integral, augmentation radii and checks, and the map is
        the array ``ldos``. A refused specification raises :class:`ApiError`; a
        cancelled, failed or unverified run raises :class:`DFTLDOSFailed`.
        """
        from ..experiments.dft import ldos, records
        from ..experiments.dft import spec as specs

        project = project or self._lab.project
        if spec is None:
            spec = self.ldos_spec(target, **variables)
        elif variables:
            spec = self.ldos_change(spec, **variables)
        try:
            outcome = ldos.execute(spec, progress=progress, cancelled=cancelled,
                                   timeout_s=timeout_s, run_id=run_id)
        except specs.SpecRefused as exc:
            raise ApiError(f"DFT LDOS refused: {exc}") from None
        if not outcome.ok:
            raise DFTLDOSFailed(outcome.status, outcome.reason, outcome.audit)
        records.store_ldos(project, outcome)
        return dict(outcome.results)

    def ldos_runs(self) -> List[dict]:
        from ..experiments.dft import records

        project = self._lab.project
        return [records.ldos_summary(project, r) for r in records.ldos_run_ids(project)]

    def _ldos_id(self, run_id: Optional[str]) -> str:
        from ..experiments.dft import records

        ids = records.ldos_run_ids(self._lab.project)
        if not ids:
            raise ApiError("No LDOS is stored in this project.")
        if run_id is None:
            return ids[0]
        if run_id not in ids:
            raise ApiError(f"No LDOS {run_id}.")
        return run_id

    def ldos_array(self, run_id: Optional[str] = None, name: str = "ldos"):
        """A stored array, verified against its checksum: ldos, ldos_spin, eigenvalues or
        kpoint_weights."""
        from ..experiments.dft import records

        run_id = self._ldos_id(run_id)
        damage = records.ldos_integrity(self._lab.project, run_id)
        if damage:
            raise ApiError(f"LDOS {run_id} refused: {damage}")
        stored = self._lab.project.arrays.get(records.ldos_key(run_id, name))
        if stored is None:
            raise ApiError(f"LDOS {run_id} stored no {name!r} array.")
        return stored

    def ldos_result(self, run_id: Optional[str] = None) -> Result:
        from ..experiments.dft import records

        return records.ldos_record(self._lab.project, self._ldos_id(run_id))

    def ldos_spec_of(self, run_id: Optional[str] = None):
        from ..experiments.dft import records

        return records.ldos_spec_of(self._lab.project, self._ldos_id(run_id))

    def ldos_state(self, run_id: Optional[str] = None, target=None) -> dict:
        """Current, stale, detached or corrupt, and why."""
        from ..experiments.dft import records

        structure = None if target is None else self._structure(target)
        return records.ldos_status(self._lab.project, self._ldos_id(run_id), structure)

    def stm_image(self, run_id: Optional[str] = None, mode: str = "constant-height",
                  height_A: Optional[float] = None, isovalue: Optional[float] = None) -> dict:
        """A Tersoff-Hamann image from a stored slab LDOS.

        ``constant-height`` returns the LDOS at ``height_A`` above the topmost
        atom; ``constant-current`` returns the height above it where the LDOS
        equals ``isovalue``. Values are proportional to the current; no
        conversion to amperes is made.
        """
        from ..experiments.dft import ldos

        run_id = self._ldos_id(run_id)
        spec = self.ldos_spec_of(run_id)
        supported, reason = ldos.stm_supported(spec.ground_state)
        if not supported:
            raise ApiError(reason)
        stored = self.ldos_array(run_id, "ldos")
        record = self.ldos_result(run_id)
        try:
            return ldos.stm_image(stored.data, np.asarray(record.value["cell_A"]),
                                  np.asarray(spec.ground_state.positions_A), mode,
                                  height_A=height_A, isovalue=isovalue,
                                  augmentation_radius_A=float(
                                      record.value["max_augmentation_radius_A"]))
        except ValueError as exc:
            raise ApiError(str(exc)) from None


class ClaimsNamespace:
    """Classified statements and the evidence behind them.

    Classifications are observation, computational-prediction,
    candidate-relation, empirical-invariant, conjecture,
    independently-reproduced and experimentally-supported.  There is no
    theorem classification: a theorem needs a formal proof, which no
    computation provides.
    """

    def __init__(self, lab: "Lab") -> None:
        self._lab = lab

    def classifications(self) -> List[dict]:
        from ..provenance.classification import describe

        return describe()

    def evidence_from(self, key: str):
        """Evidence citing a stored result by its project key."""
        from ..provenance.classification import Evidence

        item = self._lab.project.results.get(key)
        if item is None:
            raise ApiError(f"No stored result {key}.")
        params = item.provenance.parameters or {}
        software = params.get("software_reported") or params.get("software_pinned") or {}
        return Evidence(kind="result", reference=key, model=item.provenance.model,
                        software=f"GPAW {software.get('gpaw', '')}".strip()
                        if software else item.provenance.software_version,
                        digest=item.provenance.inputs_digest,
                        converged=bool(item.supported and (
                            item.convergence is None or item.convergence.converged)))

    def record(self, statement: str, classification: str,
               evidence: Sequence[Any] = (), conditions: Optional[dict] = None,
               notes: str = ""):
        """Store a claim after checking its evidence supports the classification."""
        from ..provenance.classification import Claim, ClassificationError, Evidence, parse

        items = []
        for entry in evidence:
            if isinstance(entry, Evidence):
                items.append(entry)
            elif isinstance(entry, str):
                items.append(self.evidence_from(entry))
            elif isinstance(entry, dict):
                items.append(Evidence.from_dict(entry))
            else:
                raise ApiError(f"Cannot use {entry!r} as evidence.")
        try:
            claim = Claim(statement=statement, classification=parse(classification),
                          evidence=items, conditions=dict(conditions or {}), notes=notes)
            self._lab.project.add_claim(claim)
        except ClassificationError as exc:
            raise ApiError(str(exc)) from None
        return claim

    def list(self) -> List[dict]:
        return [c.as_dict() for c in self._lab.project.claims.values()]


class ElectrostaticsNamespace:
    """Long-range electrostatics of explicit point charges.

    Charges are never inferred.  ``assign`` records a charge model on a
    structure as an undoable change; ``compute`` evaluates the Coulomb energy,
    forces, site potentials and site fields with Ewald summation (crystals),
    slab-corrected Ewald (slabs) or a direct sum (clusters) and stores every
    result in the project with its provenance.  Configurations without a
    defined electrostatic energy, such as a charged slab or a charged crystal
    without an explicitly requested background, are refused.

    Example::

        electrostatics.assign(by_element={"Ga": 1, "As": -1},
                              source="rigid-ion test charges")
        run = electrostatics.compute()
        print(run["energy"].value, run["energy"].unit)
    """

    def __init__(self, lab: "Lab") -> None:
        self._lab = lab

    def _structure(self, target=None) -> Structure:
        if target is None:
            structure = self._lab.project.structure
            if structure is None:
                raise ApiError("No active structure.")
            return structure
        return target.structure if isinstance(target, SurfaceHandle) else target

    def _settings(self, settings: Optional[dict], overrides: dict):
        from ..physics.electrostatics import EwaldSettings

        merged = dict(settings or {})
        merged.update(overrides)
        try:
            return EwaldSettings.from_dict(merged)
        except ValueError as exc:
            raise ApiError(str(exc)) from None

    def charge_model(self, target=None) -> Optional[dict]:
        """The charge model recorded on the structure, or ``None``."""
        from ..physics.electrostatics import stored_charge_model

        model = stored_charge_model(self._structure(target))
        return None if model is None else model.as_dict()

    def geometry(self, target=None) -> dict:
        """How the Coulomb sum would be performed for this structure's cell."""
        from ..physics.electrostatics import classify

        return classify(self._structure(target).cell).as_dict()

    def assign(self, target=None, *, by_element: Optional[Dict[str, float]] = None,
               by_atom_id: Optional[Dict[int, float]] = None, formal: bool = False,
               source: str = "") -> dict:
        """Record a point-charge model on a structure, as one undoable change.

        Exactly one of ``by_element``, ``by_atom_id`` or ``formal=True`` must be
        given.  The model is checked against the structure before anything is
        changed, so an incomplete table is refused and leaves no trace.
        """
        from ..core_model.selection import Selection
        from ..physics.electrostatics import (
            ChargeModel, ChargeModelError, charge_accounting, store_charge_model,
        )

        chosen = [by_element is not None, by_atom_id is not None, bool(formal)]
        if sum(chosen) != 1:
            raise ApiError("Give exactly one of by_element, by_atom_id or formal=True.")
        structure = self._structure(target)
        try:
            if formal:
                model = ChargeModel.formal_point_ion(source)
            elif by_element is not None:
                model = ChargeModel.per_element(by_element, source)
            else:
                model = ChargeModel.per_atom(by_atom_id, source)
            charges = model.charges(structure)
        except ChargeModelError as exc:
            raise ApiError(str(exc)) from None
        project = self._lab.project
        before = structure.copy()
        before_selection = Selection(list(project.selection.ids), project.selection.query)
        store_charge_model(structure, model)
        accounting = charge_accounting(structure, charges)
        project.record_change(
            structure, before, before_selection,
            f"Assign point charges ({model.kind})", "electrostatics.assign",
            {"charge_model": model.as_dict()},
            result_summary=f"net point charge {accounting['total_charge_e']:+.6g} e")
        return {"charge_model": model.as_dict(), "charge_accounting": accounting}

    def clear(self, target=None) -> bool:
        """Remove the charge model, as one undoable change.  ``False`` if none."""
        from ..core_model.selection import Selection
        from ..physics.electrostatics import store_charge_model, stored_charge_model

        structure = self._structure(target)
        if stored_charge_model(structure) is None:
            return False
        project = self._lab.project
        before = structure.copy()
        before_selection = Selection(list(project.selection.ids), project.selection.query)
        store_charge_model(structure, None)
        project.record_change(structure, before, before_selection,
                              "Remove point charges", "electrostatics.clear", {})
        return True

    def status(self, target=None, settings: Optional[dict] = None, **overrides) -> dict:
        """Charge model, geometry, charge accounting and whether a run can proceed."""
        from ..physics.electrostatics import (
            ChargeModelError, charge_accounting, classify, stored_charge_model,
        )
        from ..solvers.electrostatics import ElectrostaticsSolver

        structure = self._structure(target)
        parsed = self._settings(settings, overrides)
        out = {"geometry": classify(structure.cell).as_dict(),
               "settings": parsed.as_dict(), "n_atoms": len(structure),
               "elements": sorted({_pt.symbol(int(z)) for z in structure.numbers}),
               "formal_total_e": float(structure.total_charge()),
               "charge_model": None, "charge_accounting": None}
        try:
            model = stored_charge_model(structure)
        except ChargeModelError as exc:
            out["ready"] = False
            out["blocking"] = [str(exc)]
            out["warnings"] = []
            return out
        if model is not None:
            out["charge_model"] = model.as_dict()
            try:
                out["charge_accounting"] = charge_accounting(structure,
                                                             model.charges(structure))
            except ChargeModelError:
                pass
        report = ElectrostaticsSolver().supports(structure, model, parsed)
        out["ready"] = report.ok
        out["blocking"] = list(report.blocking)
        out["warnings"] = list(report.warnings)
        return out

    def compute(self, target=None, settings: Optional[dict] = None,
                progress: Optional[Callable[[float, str], Optional[bool]]] = None,
                cancelled: Optional[Callable[[], bool]] = None,
                run_id: Optional[str] = None, project: Optional[Project] = None,
                structure_key: Optional[str] = None, **overrides):
        """Run one electrostatics calculation and store its results.

        Returns the :class:`~materia.solvers.SolverResult`.  A refused or
        cancelled calculation comes back as an unsupported ``energy`` result
        carrying the reason; nothing numeric is stored for it.  Results are
        kept in the project as ``electrostatics::<run_id>::<quantity>``.

        The calculation runs on a copy, so editing the structure while it runs
        cannot change what is computed.  ``project`` and ``structure_key`` let
        a caller that froze its inputs earlier name where the results belong.
        """
        from ..project_format.history import LogEntry
        from ..solvers.electrostatics import ElectrostaticsSolver

        structure = self._structure(target)
        parsed = self._settings(settings, overrides)
        project = project or self._lab.project
        run_id = run_id or f"{int(time.time() * 1000):013d}-{os.urandom(3).hex()}"
        if structure_key is None:
            structure_key = project.key_of(structure)
        out = ElectrostaticsSolver().single_point(structure.copy(), settings=parsed,
                                                  progress=progress, cancelled=cancelled)
        for result in out.results.values():
            result.extra["run_id"] = run_id
            result.extra["structure_key"] = structure_key
        for key, result in out.results.items():
            project.add_result(f"electrostatics::{run_id}::{key}", result)
        energy = out.results["energy"]
        summary = (f"E = {energy.value:.6f} eV, {energy.provenance.origin.value}"
                   if energy.supported else
                   f"{energy.extra.get('status', 'refused')}: {energy.unsupported_reason}")
        project.history.log.append(LogEntry(
            operation="electrostatics.compute",
            label=f"Electrostatics, {len(structure)} atoms",
            parameters={"run_id": run_id, "structure_key": structure_key,
                        "settings": parsed.as_dict()},
            result_summary=summary[:300], undoable=False))
        project.touch()
        return out

    def runs(self) -> List[dict]:
        """Every electrostatics run held by the project, newest first."""
        runs: Dict[str, dict] = {}
        for key, result in self._lab.project.results.items():
            parts = key.split("::")
            if len(parts) != 3 or parts[0] != "electrostatics":
                continue
            run = runs.setdefault(parts[1], {"run_id": parts[1], "keys": []})
            run["keys"].append(key)
            if parts[2] == "energy":
                run["supported"] = bool(result.supported)
                run["energy_eV"] = result.value
                run["status"] = result.extra.get("status", "")
                run["origin"] = result.provenance.origin.value
                run["structure_key"] = result.extra.get("structure_key")
                run["inputs_digest"] = result.provenance.inputs_digest
                run["reason"] = result.unsupported_reason
        return [runs[k] for k in sorted(runs, reverse=True)]

    def result(self, run_id: Optional[str] = None, quantity: str = "energy") -> Result:
        """One stored quantity of a run; the newest run when ``run_id`` is omitted."""
        runs = self.runs()
        if not runs:
            raise ApiError("No electrostatics run is stored in this project.")
        run_id = run_id or runs[0]["run_id"]
        key = f"electrostatics::{run_id}::{quantity}"
        if key not in self._lab.project.results:
            raise ApiError(f"No stored result {key}.")
        return self._lab.project.results[key]


EAM_TASKS = ("energy", "relax", "md")
EAM_SETTINGS = {
    "energy": {},
    "relax": {"fmax_eV_A": 0.01, "max_steps": 1000},
    "md": {"steps": 200, "dt_fs": 2.0, "temperature_K": 300.0, "thermostat": "langevin",
           "friction_per_fs": 0.01, "seed": 0, "sample_every": 1},
}


def structure_state_digest(structure: Structure) -> str:
    """Fingerprint of what an energy depends on: atoms, elements, positions, cell."""
    from ..provenance import digest

    return digest({"ids": structure.ids.tolist(), "numbers": structure.numbers.tolist(),
                   "positions": np.round(structure.positions, 10).tolist(),
                   "cell": structure.cell.as_dict()})


class EAMNamespace:
    """Embedded-atom potentials for metals.

    Shipped potentials are read from checksummed setfl files with a recorded
    source, licence and citation.  ``potential`` may be a shipped identifier
    such as ``"Cu-Zhou04"``, the path of a user-supplied setfl file, or
    ``None`` to choose the shipped potential that covers the structure's
    elements.  A structure containing an element the potential does not
    describe is refused.

    Example::

        cu = materials.load("copper").bulk(repeat=(3, 3, 3))
        run = eam.run(cu, "relax", fmax_eV_A=0.005)
        print(run["energy"].value, run["energy"].convergence.message)
    """

    def __init__(self, lab: "Lab") -> None:
        self._lab = lab

    def _structure(self, target=None) -> Structure:
        if target is None:
            structure = self._lab.project.structure
            if structure is None:
                raise ApiError("No active structure.")
            return structure
        return target.structure if isinstance(target, SurfaceHandle) else target

    def catalog(self) -> List[dict]:
        """Shipped potentials with their identity, source, licence and citations."""
        from ..physics import eam as eam_module

        out = []
        for entry in eam_module.catalog():
            potential = eam_module.load_shipped(entry["id"])
            info = potential.identity.as_dict()
            info["cutoff_A"] = potential.cutoff_A
            info["cutoff_residuals"] = potential.cutoff_residuals()
            out.append(info)
        return out

    def user_files(self) -> List[str]:
        from ..physics import eam as eam_module

        return [str(p) for p in eam_module.user_files()]

    def potential(self, potential: Optional[str] = None, target=None):
        """The :class:`~materia.physics.eam.EAMPotential` a run would use."""
        from ..physics import eam as eam_module
        from ..physics.potentials import PotentialError

        try:
            if potential is None:
                structure = self._structure(target)
                present = sorted({_pt.symbol(int(z)) for z in structure.numbers})
                chosen, reason = eam_module.select_shipped(present)
                if chosen is None:
                    raise ApiError(reason)
                return eam_module.load_shipped(chosen)
            if potential in eam_module.shipped_ids():
                return eam_module.load_shipped(potential)
            if os.path.exists(os.path.expanduser(potential)):
                return eam_module.load_file(potential)
        except PotentialError as exc:
            raise ApiError(str(exc)) from None
        raise ApiError(f"Unknown EAM potential {potential!r}. Shipped: "
                       f"{', '.join(eam_module.shipped_ids())}; or give the path of a "
                       "setfl file.")

    def settings(self, task: str, settings: Optional[dict] = None, **overrides) -> dict:
        if task not in EAM_TASKS:
            raise ApiError(f"Unknown task {task!r}. Available: {', '.join(EAM_TASKS)}.")
        merged = dict(EAM_SETTINGS[task])
        given = dict(settings or {})
        given.update(overrides)
        unknown = sorted(set(given) - set(merged))
        if unknown:
            raise ApiError(f"Unknown setting(s) for {task}: {', '.join(unknown)}. Known: "
                           f"{', '.join(sorted(merged)) or 'none'}.")
        merged.update(given)
        if task == "relax":
            merged["fmax_eV_A"] = float(merged["fmax_eV_A"])
            merged["max_steps"] = int(merged["max_steps"])
            if merged["fmax_eV_A"] <= 0 or merged["max_steps"] < 1:
                raise ApiError("fmax_eV_A must be positive and max_steps at least 1.")
        if task == "md":
            merged["steps"] = int(merged["steps"])
            merged["dt_fs"] = float(merged["dt_fs"])
            merged["temperature_K"] = float(merged["temperature_K"])
            merged["friction_per_fs"] = float(merged["friction_per_fs"])
            merged["seed"] = int(merged["seed"])
            merged["sample_every"] = max(1, int(merged["sample_every"]))
            if not np.isfinite([merged["dt_fs"], merged["temperature_K"],
                                merged["friction_per_fs"]]).all():
                raise ApiError("MD time step, temperature and friction must be finite.")
            if merged["thermostat"] not in ("none", "langevin"):
                raise ApiError("thermostat must be 'none' (NVE) or 'langevin'.")
            if merged["steps"] < 1 or not (0 < merged["dt_fs"] <= 10):
                raise ApiError("steps must be at least 1 and dt_fs between 0 and 10 fs.")
            if merged["temperature_K"] < 0:
                raise ApiError("temperature_K cannot be negative.")
            if merged["friction_per_fs"] < 0:
                raise ApiError("friction_per_fs cannot be negative.")
            if merged["thermostat"] == "langevin" and merged["temperature_K"] <= 0:
                raise ApiError("A Langevin thermostat needs a temperature above 0 K. Use "
                               "thermostat 'none' for constant-energy dynamics.")
        return merged

    def run(self, target=None, task: str = "energy", potential: Optional[str] = None,
            settings: Optional[dict] = None,
            progress: Optional[Callable[[float, str], None]] = None,
            cancelled: Optional[Callable[[], bool]] = None,
            run_id: Optional[str] = None, project: Optional[Project] = None,
            structure_key: Optional[str] = None, apply: bool = True,
            input_digest: Optional[str] = None, **overrides):
        """Run one EAM calculation on a copy and store it in the project.

        Returns a dict of :class:`~materia.provenance.Result` keyed by quantity;
        ``"energy"`` is always present and is unsupported when the run was
        refused or cancelled.

        ``task`` is ``"energy"`` (energy and forces), ``"relax"`` (FIRE at fixed
        cell) or ``"md"`` (velocity Verlet, optionally with a Langevin
        thermostat).  Results are stored as ``eam::<run_id>::<quantity>``.

        With ``apply`` set, the outcome is written back to the structure it was
        computed for as one undoable change: forces for an energy run, relaxed
        positions and forces for a converged relaxation, final positions,
        velocities and forces for dynamics.  It is not written if that
        structure changed while the run was in progress, or if a relaxation
        did not converge.  A cancelled run stores an unsupported record and
        changes nothing.
        """
        from ..physics.neighbors import CoincidentAtoms
        from ..physics.potentials import PotentialError
        from ..solvers.classical import ClassicalSolver

        structure = self._structure(target)
        project = project or self._lab.project
        if structure_key is None:
            structure_key = project.key_of(structure)
        chosen = self.potential(potential, structure)
        parsed = self.settings(task, settings, **overrides)
        run_id = run_id or f"{int(time.time() * 1000):013d}-{os.urandom(3).hex()}"
        before_digest = input_digest or structure_state_digest(structure)
        snapshot = structure.copy()
        solver = ClassicalSolver(chosen)
        report = solver.supports(snapshot)
        stopped = {"flag": False}

        def is_cancelled() -> bool:
            return bool(cancelled is not None and cancelled())

        def tick(fraction: float, message: str) -> bool:
            if progress is not None:
                progress(min(0.99, fraction), message)
            if is_cancelled():
                stopped["flag"] = True
                return False
            return True

        if not report.ok:
            raise ApiError("; ".join(report.blocking))
        try:
            out = self._execute(solver, snapshot, task, parsed, chosen, tick, is_cancelled,
                                stopped)
        except (PotentialError, CoincidentAtoms) as exc:
            refusal = unsupported("eam_energy", chosen.name,
                                  f"The {task} calculation was refused part way: {exc} No "
                                  "value is kept and the structure was not changed.",
                                  unit="eV")
            refusal.extra.update({"status": "refused", "task": task,
                                  "structure_key": structure_key})
            project.add_result(f"eam::{run_id}::energy", refusal)
            return {"energy": refusal}
        if stopped["flag"] or is_cancelled():
            refusal = unsupported("eam_energy", chosen.name,
                                  f"The {task} calculation was cancelled. No value is kept "
                                  "and the structure was not changed.", unit="eV")
            refusal.extra.update({"status": "cancelled", "task": task})
            project.add_result(f"eam::{run_id}::energy", refusal)
            self._log(project, run_id, task, chosen, structure_key, refusal)
            return {"energy": refusal}
        return self._finish(project, structure, structure_key, snapshot, task, parsed,
                            chosen, run_id, before_digest, apply, out)

    def _execute(self, solver, snapshot: Structure, task: str, parsed: dict, chosen, tick,
                 is_cancelled, stopped):
        if is_cancelled():
            stopped["flag"] = True
            return None
        if task == "energy":
            tick(0.1, f"{chosen.name}: energy and forces, {len(snapshot)} atoms")
            out = solver.single_point(snapshot)
            if not out.results["energy"].supported:
                raise PotentialError(out.results["energy"].unsupported_reason)
            return out
        if task == "relax":
            return solver.relax(
                snapshot, fmax_eV_A=parsed["fmax_eV_A"], max_steps=parsed["max_steps"],
                in_place=True,
                callback=lambda step, e, f: tick(
                    step / parsed["max_steps"],
                    f"step {step}: E = {e:.6f} eV, max |F| = {f:.2e} eV/A"))
        return solver.dynamics(
            snapshot, steps=parsed["steps"], dt_fs=parsed["dt_fs"],
            temperature_K=parsed["temperature_K"] or None,
            thermostat=parsed["thermostat"], friction_per_fs=parsed["friction_per_fs"],
            seed=parsed["seed"], sample_every=parsed["sample_every"],
            initialise_velocities=parsed["temperature_K"] > 0, in_place=True,
            callback=lambda step, e, t: tick(
                step / parsed["steps"], f"step {step}: T = {t:.1f} K"))

    def _finish(self, project: Project, structure: Structure, structure_key, snapshot,
                task: str, parsed: dict, chosen, run_id: str, before_digest: str,
                apply: bool, out):
        from ..provenance import Origin
        from ..project_format.arrays import StoredArray

        converged = out.convergence.converged if (task == "relax" and out.convergence) else True
        after_digest = structure_state_digest(snapshot)
        applied, reason = False, "Not applied: apply was switched off."
        if apply:
            applied, reason = self._apply(project, structure_key, before_digest, snapshot,
                                          task, chosen, run_id, converged, out)
        kept = {}
        for key in ("energy", "forces", "relaxation_history", "trajectory", "mean_temperature"):
            if key in out.results:
                kept[key] = out.results[key]
        trajectory = kept.get("trajectory")
        if trajectory is not None and trajectory.value:
            values = trajectory.value
            positions = np.asarray(values.pop("positions_A", []), dtype=np.float64)
            velocities = np.asarray(values.pop("velocities_A_fs", []), dtype=np.float64)
            if positions.ndim == 3 and positions.shape[0]:
                common = {
                    "run_id": run_id,
                    "atom_ids": [int(i) for i in snapshot.ids],
                    "numbers": [int(z) for z in snapshot.numbers],
                    "roles": [str(role) for role in snapshot.roles],
                    "cell": snapshot.cell.matrix.tolist(),
                    "pbc": list(snapshot.cell.pbc),
                    "frame_steps": [int(step) for step in values.get("step", [])],
                    "frame_times_fs": [float(value) for value in values.get("time_fs", [])],
                    "collision": dict(snapshot.info.get("collision", {})),
                }
                positions_key = f"eam::{run_id}::positions"
                velocities_key = f"eam::{run_id}::velocities"
                project.add_array(positions_key, StoredArray(
                    positions, "A", "Sampled atom positions from EAM dynamics",
                    "trajectory", dict(common)))
                project.add_array(velocities_key, StoredArray(
                    velocities, "A/fs", "Sampled atom velocities from EAM dynamics",
                    "trajectory", dict(common)))
                values["positions_array"] = positions_key
                values["velocities_array"] = velocities_key
                values["frames"] = int(positions.shape[0])
        for result in kept.values():
            if not converged:
                result.provenance.origin = Origin.ESTIMATED
            result.provenance.inputs_digest = before_digest
            result.provenance.parameters["task_settings"] = dict(parsed)
            result.extra.update({
                "run_id": run_id, "task": task, "structure_key": structure_key,
                "potential_id": chosen.identity.id, "potential_sha256": chosen.identity.sha256,
                "input_state_digest": before_digest, "output_state_digest": after_digest,
                "applied": applied, "apply_reason": reason,
                "status": ("converged" if converged else "not-converged")
                if task == "relax" else "complete",
                "atom_ids": [int(i) for i in snapshot.ids],
                "n_atoms": len(snapshot), "wall_time_s": out.wall_time_s,
                "neighbour_list_builds": chosen.last_evaluation.get("neighbour_list_builds"),
                "rho_extrapolated": chosen.last_evaluation.get("rho_extrapolated", 0),
                "log": list(out.log)[:20]})
        for key, result in kept.items():
            project.add_result(f"eam::{run_id}::{key}", result)
        self._log(project, run_id, task, chosen, structure_key, kept["energy"])
        return kept

    def _apply(self, project: Project, structure_key: Optional[str], before_digest: str,
               snapshot: Structure, task: str, chosen, run_id: str, converged: bool, out):
        from ..core_model.selection import Selection

        if structure_key is None or structure_key not in project.structures:
            return False, "Not applied: the structure is not held by this project."
        current = project.structures[structure_key]
        if structure_state_digest(current) != before_digest:
            return False, ("Not applied: the structure changed while the calculation was "
                           "running, so the result describes a different geometry.")
        if task == "relax" and not converged:
            return False, ("Not applied: the relaxation did not reach its force tolerance, "
                           "so the geometry is not an energy minimum.")
        before = current.copy()
        before_selection = Selection(list(project.selection.ids), project.selection.query)
        if task in ("relax", "md"):
            current.positions = snapshot.positions
        if task == "md":
            current.velocities = snapshot.velocities
        current.forces = snapshot.forces
        energy = out.results["energy"].value
        label = {"energy": "EAM forces", "relax": "EAM relaxation",
                 "md": "EAM dynamics"}[task]
        project.record_change(current, before, before_selection, label,
                              f"solver.eam.{task}",
                              {"potential": chosen.identity.id,
                               "sha256": chosen.identity.sha256, "run_id": run_id},
                              result_summary=f"E = {energy:.6f} eV")
        current.invalidate_bonds()
        return True, "Applied to the structure as one undoable change."

    def _log(self, project: Project, run_id: str, task: str, chosen, structure_key,
             energy) -> None:
        from ..project_format.history import LogEntry

        summary = (f"E = {energy.value:.6f} eV" if energy.supported else
                   f"{energy.extra.get('status', 'refused')}: {energy.unsupported_reason}")
        project.history.log.append(LogEntry(
            operation=f"eam.{task}", label=f"EAM {task} with {chosen.name}",
            parameters={"run_id": run_id, "structure_key": structure_key,
                        "potential": chosen.identity.id,
                        "sha256": chosen.identity.sha256},
            result_summary=summary[:300], undoable=False))
        project.touch()

    def runs(self) -> List[dict]:
        """Every EAM run held by the project, newest first."""
        runs: Dict[str, dict] = {}
        for key, result in self._lab.project.results.items():
            parts = key.split("::")
            if len(parts) != 3 or parts[0] != "eam":
                continue
            run = runs.setdefault(parts[1], {"run_id": parts[1], "keys": []})
            run["keys"].append(key)
            if parts[2] == "energy":
                run.update({"supported": bool(result.supported), "energy_eV": result.value,
                            "task": result.extra.get("task"),
                            "status": result.extra.get("status", ""),
                            "origin": result.provenance.origin.value,
                            "potential_id": result.extra.get("potential_id"),
                            "applied": result.extra.get("applied"),
                            "structure_key": result.extra.get("structure_key")})
        return [runs[k] for k in sorted(runs, reverse=True)]

    def result(self, run_id: Optional[str] = None, quantity: str = "energy") -> Result:
        runs = self.runs()
        if not runs:
            raise ApiError("No EAM run is stored in this project.")
        run_id = run_id or runs[0]["run_id"]
        key = f"eam::{run_id}::{quantity}"
        if key not in self._lab.project.results:
            raise ApiError(f"No stored result {key}.")
        return self._lab.project.results[key]

    def is_current(self, run_id: Optional[str] = None) -> bool:
        """Whether a stored run still describes its structure as it is now."""
        energy = self.result(run_id, "energy")
        key = energy.extra.get("structure_key")
        structure = self._lab.project.structures.get(key or "")
        if structure is None or not energy.supported:
            return False
        return structure_state_digest(structure) == energy.extra.get("output_state_digest")


class CollisionNamespace:
    """Classical atomistic impact experiments with explicit model boundaries."""

    def __init__(self, lab: "Lab") -> None:
        self._lab = lab

    @staticmethod
    def _structure(value) -> Structure:
        structure = value.structure if isinstance(value, SurfaceHandle) else value
        if not isinstance(structure, Structure):
            raise ApiError("projectile and target must be Materia structures or handles.")
        return structure

    def prepare(self, projectile, target, relative_speed_A_fs: float,
                direction: Sequence[float] = (1.0, 0.0, 0.0),
                impact_parameter_A: float = 0.0, gap_A: float = 4.0,
                padding_A: float = 8.0, activate: bool = True) -> Structure:
        """Build a non-periodic collision system and preserve its initial velocities."""
        from ..experiments.collisions import CollisionError, prepare

        try:
            structure = prepare(
                self._structure(projectile), self._structure(target),
                relative_speed_A_fs=relative_speed_A_fs, direction=direction,
                impact_parameter_A=impact_parameter_A, gap_A=gap_A,
                padding_A=padding_A)
        except CollisionError as exc:
            raise ApiError(str(exc)) from None
        if activate:
            self._lab.project.add_structure(structure, activate=True, log=True)
            self._lab.project.history.log[-1].label = "Prepare atomistic collision"
        return structure

    def run(self, target=None, potential: Optional[str] = None, steps: int = 1000,
            dt_fs: float = 0.2, sample_every: int = 5, apply: bool = True,
            **settings):
        """Run NVE dynamics using the velocities assigned by :meth:`prepare`."""
        structure = (self._lab.project.structure if target is None
                     else self._structure(target))
        if structure is None:
            raise ApiError("No collision structure is active.")
        if "collision" not in structure.info:
            raise ApiError("This structure has no collision setup. Call collisions.prepare first.")
        variables = {
            "steps": steps, "dt_fs": dt_fs, "temperature_K": 0.0,
            "thermostat": "none", "sample_every": sample_every,
        }
        variables.update(settings)
        return self._lab.eam.run(structure, "md", potential=potential,
                                 settings=variables, apply=apply)

    def fragments(self, target=None, positions=None, scale: float = 1.35) -> dict:
        """Return contact-connected fragments for a structure or trajectory frame."""
        from ..experiments.collisions import CollisionError, fragments

        structure = (self._lab.project.structure if target is None
                     else self._structure(target))
        if structure is None:
            raise ApiError("No active structure.")
        try:
            labels, summary = fragments(
                structure, structure.positions if positions is None else positions, scale)
        except CollisionError as exc:
            raise ApiError(str(exc)) from None
        return {"labels": labels, "fragments": summary}


class LAMMPSRunFailed(ApiError):
    """A LAMMPS run that was cancelled or failed.  Nothing was stored."""

    def __init__(self, status: str, reason: str, audit: Optional[dict] = None) -> None:
        super().__init__(f"LAMMPS run {status}: {reason}")
        self.status = status
        self.reason = reason
        self.audit = dict(audit or {})


class LAMMPSNamespace:
    """LAMMPS, driven out of process, with a frozen and auditable specification.

    LAMMPS is optional and never bundled.  ``status()`` reports the exact
    executable or Python module found, its version and packages, or how to
    install it.  ``spec()`` freezes every variable of a run, ``check()`` lists
    every refusal, ``native_input()`` shows the exact files LAMMPS will read,
    and ``run()`` executes, validates and stores the result.  A refused,
    failed or cancelled run raises and leaves the project unchanged.

    Tasks: ``"energy"`` (energy, forces and, for a fully periodic cell, the
    stress), ``"relax"`` (fixed-cell minimisation) and ``"md"`` (``nve``,
    ``langevin`` or ``nvt``).  Potentials are EAM setfl files, the shipped
    ones or a user file, exactly as for :attr:`eam`.  Nothing ever falls back
    to Materia's own EAM solver.

    Example::

        cu = materials.load("copper").bulk(repeat=(3, 3, 3))
        spec = lammps.spec(cu, "md", steps=1000, ensemble="langevin",
                           temperature_K=300, timestep_fs=2.0, sample_every=10)
        lammps.check(spec)["blocking"]
        run = lammps.run(spec=spec)
        run["energy"].value, lammps.state()["state"]
    """

    def __init__(self, lab: "Lab") -> None:
        self._lab = lab

    def _structure(self, target=None) -> Structure:
        if target is None:
            structure = self._lab.project.structure
            if structure is None:
                raise ApiError("No active structure.")
            return structure
        return target.structure if isinstance(target, SurfaceHandle) else target

    def status(self, refresh: bool = False) -> dict:
        """The LAMMPS Materia can see, or what is missing and how to install it."""
        from ..solvers.lammps import discover

        return discover(refresh=refresh).as_dict()

    def available(self) -> bool:
        return bool(self.status()["available"])

    def spec(self, target=None, task: str = "energy", potential: Optional[str] = None,
             settings: Optional[dict] = None, project: Optional[Project] = None,
             **overrides):
        """Freeze a run specification.  It is not checked here; see :meth:`check`."""
        from ..solvers.lammps import discover
        from ..solvers.lammps import spec as specs

        structure = self._structure(target)
        project = project or self._lab.project
        chosen = self._lab.eam.potential(potential, structure)
        try:
            return specs.build(structure, task, chosen, discover(),
                               structure_key=project.key_of(structure),
                               settings=settings, **overrides)
        except specs.SpecError as exc:
            raise ApiError(str(exc)) from None

    def change(self, spec, **values):
        from ..solvers.lammps import spec as specs

        try:
            return specs.changed(spec, **values)
        except specs.SpecError as exc:
            raise ApiError(str(exc)) from None

    def check(self, spec) -> dict:
        from ..solvers.lammps import spec as specs

        return specs.check(spec).as_dict()

    def describe(self, spec) -> dict:
        from ..solvers.lammps import spec as specs

        return specs.describe(spec)

    def native_input(self, spec) -> dict:
        """The exact input script and data file LAMMPS would read, with their SHA-256."""
        from ..solvers.lammps import native

        script = native.render_input(spec)
        data = native.render_data(spec)
        return {"in.lammps": script, "structure.data": data,
                "sha256": {"in.lammps": native.sha256_text(script),
                           "structure.data": native.sha256_text(data),
                           native.potential_file_name(spec): spec.potential.get("sha256")}}

    def run(self, target=None, task: str = "energy", potential: Optional[str] = None,
            settings: Optional[dict] = None, *, spec=None,
            progress: Optional[Callable[[float, str], None]] = None,
            cancelled: Optional[Callable[[], bool]] = None,
            timeout_s: Optional[float] = None, apply: bool = True,
            run_id: Optional[str] = None, project: Optional[Project] = None,
            keep_files: bool = False, **overrides) -> Dict[str, Result]:
        """Run LAMMPS and store the validated result.

        Returns the stored results keyed by quantity: ``run`` (the audit record
        with the command, version, packages and input hashes), ``energy``,
        ``forces``, and as requested ``stress``, ``thermo``, ``relaxation``,
        ``trajectory`` and ``mean_temperature``.  Raises :class:`ApiError` for a
        refused specification and :class:`LAMMPSRunFailed` for a failed or
        cancelled run; neither stores anything or touches the history.

        With ``apply`` set, the final state is written onto the structure as
        one undoable change if that structure has not changed since the run
        started and, for a relaxation, if it converged.
        """
        from ..solvers.lammps import records
        from ..solvers.lammps import spec as specs
        from ..solvers.lammps.run import execute

        project = project or self._lab.project
        if spec is None:
            spec = self.spec(target, task, potential, settings, project=project, **overrides)
        elif overrides or settings:
            spec = self.change(spec, **dict(settings or {}), **overrides)
        try:
            outcome = execute(spec, progress=progress, cancelled=cancelled,
                              timeout_s=timeout_s, run_id=run_id, keep_files=keep_files)
        except specs.SpecRefused as exc:
            raise ApiError(f"LAMMPS run refused: {exc}") from None
        if not outcome.ok:
            raise LAMMPSRunFailed(outcome.status, outcome.reason, outcome.audit)
        records.store(project, outcome, apply=apply)
        return dict(outcome.results)

    def runs(self) -> List[dict]:
        from ..solvers.lammps import records

        project = self._lab.project
        return [records.summary(project, run_id) for run_id in records.run_ids(project)]

    def _run_id(self, run_id: Optional[str]) -> str:
        from ..solvers.lammps import records

        ids = records.run_ids(self._lab.project)
        if not ids:
            raise ApiError("No LAMMPS run is stored in this project.")
        if run_id is None:
            return ids[0]
        if run_id not in ids:
            raise ApiError(f"No LAMMPS run {run_id}.")
        return run_id

    def result(self, run_id: Optional[str] = None, quantity: str = "energy") -> Result:
        from ..solvers.lammps.run import key

        run_id = self._run_id(run_id)
        item = self._lab.project.results.get(key(run_id, quantity))
        if item is None:
            raise ApiError(f"LAMMPS run {run_id} has no {quantity!r}.")
        array_key = item.extra.get("array")
        if item.value is None and array_key in self._lab.project.arrays:
            item.value = np.array(self._lab.project.arrays[array_key].data)
        return item

    def array(self, run_id: Optional[str] = None, name: str = "positions"):
        """A stored array: positions, velocities, forces, final_positions or final_velocities."""
        from ..solvers.lammps.run import key

        run_id = self._run_id(run_id)
        stored = self._lab.project.arrays.get(key(run_id, name))
        if stored is None:
            raise ApiError(f"LAMMPS run {run_id} stored no {name!r} array.")
        return stored

    def spec_of(self, run_id: Optional[str] = None):
        from ..solvers.lammps import records

        return records.spec_of(self._lab.project, self._run_id(run_id))

    def state(self, run_id: Optional[str] = None, target=None) -> dict:
        """Whether a run is current, stale or detached, and whether it can be applied."""
        from ..solvers.lammps import records

        structure = None if target is None else self._structure(target)
        return records.state(self._lab.project, self._run_id(run_id), structure)

    def is_current(self, run_id: Optional[str] = None, target=None) -> bool:
        return bool(self.state(run_id, target).get("current"))

    def apply(self, run_id: Optional[str] = None) -> str:
        """Write a stored run onto its structure as one undoable change."""
        from ..solvers.lammps import records

        try:
            return records.apply_run(self._lab.project, self._run_id(run_id))
        except records.ApplyRefused as exc:
            raise ApiError(f"Not applied: {exc}") from None


class MeasureNamespace:
    def __init__(self, lab: "Lab") -> None:
        self._lab = lab

    def _s(self, target=None) -> Structure:
        if target is None:
            s = self._lab.project.structure
            if s is None:
                raise ApiError("No active structure")
            return s
        return target.structure if isinstance(target, SurfaceHandle) else target

    def distance(self, a: int, b: int, target=None) -> float:
        return self._s(target).distance(a, b)

    def angle(self, a: int, b: int, c: int, target=None) -> float:
        return self._s(target).angle(a, b, c)

    def dihedral(self, i: int, j: int, k: int, l: int, target=None) -> float:
        return self._s(target).dihedral(i, j, k, l)

    def coordination(self, atom_id: int, target=None) -> int:
        s = self._s(target)
        if s.bonds is None:
            attach_bonds(s)
        return len(s.neighbors_of(atom_id))

    def neighbour_list(self, cutoff_A: float, target=None):
        s = self._s(target)
        return neighbor_list(s.positions, s.cell, cutoff_A)

    def strain(self, reference, target=None, **kwargs) -> Result:
        ref = reference.structure if isinstance(reference, SurfaceHandle) else reference
        return local_strain(self._s(target), ref, **kwargs)

    def radial_distribution(self, r_max_A: float = 8.0, bins: int = 200, target=None):
        from ..physics.neighbors import radial_distribution
        s = self._s(target)
        nl = neighbor_list(s.positions, s.cell, r_max_A)
        return radial_distribution(nl, len(s), s.cell.volume or 1.0, r_max_A, bins)


class NEBNamespace:
    def __init__(self, lab: "Lab") -> None:
        self._lab = lab

    @staticmethod
    def _structure(target) -> Structure:
        if isinstance(target, SurfaceHandle):
            return target.structure
        if isinstance(target, Structure):
            return target
        raise ApiError("NEB endpoints must be structures or surface handles.")

    def run(self, initial, final, model: str = "recommended", settings=None, **kwargs):
        from ..physics import neb as neb_module
        from ..solvers.classical import ClassicalSolver

        first = self._structure(initial)
        last = self._structure(final)
        if settings is not None and kwargs:
            raise ApiError("Pass either settings or individual NEB options, not both.")
        if settings is None:
            settings = neb_module.NEBSettings(**kwargs)
        elif not isinstance(settings, neb_module.NEBSettings):
            raise ApiError("settings must be a NEBSettings instance.")
        material = initial.material if isinstance(initial, SurfaceHandle) else None
        solver = self._lab._resolve_solver(model, first, material, "energy")
        if not isinstance(solver, ClassicalSolver):
            raise ApiError(
                f"{solver.name} is not an in-process force model for NEB. "
                "Use a classical Materia potential until scheduled DFT NEB is available.")
        path = neb_module.run(first, last, solver.potential, settings)
        run_id = f"{int(time.time() * 1000):013d}-{os.urandom(3).hex()}"
        for key, result in path.results().items():
            result.extra["run_id"] = run_id
            self._lab.project.add_result(f"neb::{run_id}::{key}", result)
        for key, stored in path.stored_arrays().items():
            stored.meta["run_id"] = run_id
            self._lab.project.add_array(f"neb::{run_id}::{key}", stored)
        return path

    def runs(self) -> List[str]:
        prefix = "neb::"
        ids = {key[len(prefix):].split("::", 1)[0]
               for key in list(self._lab.project.results) + list(self._lab.project.arrays)
               if key.startswith(prefix) and "::" in key[len(prefix):]}
        return sorted(ids, reverse=True)

    def result(self, run_id: Optional[str] = None, quantity: str = "barrier") -> Result:
        ids = self.runs()
        chosen = run_id or (ids[0] if ids else None)
        if chosen is None:
            raise ApiError("No stored NEB runs.")
        key = f"neb::{chosen}::{quantity}"
        if key not in self._lab.project.results:
            available = sorted(
                stored_key.rsplit("::", 1)[-1]
                for stored_key in self._lab.project.results
                if stored_key.startswith(f"neb::{chosen}::"))
            raise ApiError(
                f"NEB run {chosen!r} has no result {quantity!r}. Available: "
                f"{available}")
        return self._lab.project.results[key]

    def array(self, run_id: Optional[str] = None, name: str = "positions"):
        ids = self.runs()
        chosen = run_id or (ids[0] if ids else None)
        if chosen is None:
            raise ApiError("No stored NEB runs.")
        key = f"neb::{chosen}::{name}"
        if key not in self._lab.project.arrays:
            raise ApiError(f"NEB run {chosen!r} has no stored array {name!r}.")
        return self._lab.project.arrays[key]


class AlloyNamespace:
    def __init__(self, lab: "Lab") -> None:
        self._lab = lab

    def random(self, target, composition: Mapping[str, float], *, atom_ids=None,
               seed: int = 0, activate: bool = True) -> SurfaceHandle:
        from ..structure_builder.alloys import random_substitutional

        if isinstance(target, SurfaceHandle):
            structure = target.structure
            material = target.material
        elif isinstance(target, Structure):
            structure = target
            material_id = structure.info.get("material_id")
            if not material_id:
                raise ApiError(
                    "A raw structure needs info['material_id'] so the alloy keeps its material.")
            material = self._lab.library.get(material_id)
        else:
            raise ApiError("The alloy target must be a structure or surface handle.")
        output, record = random_substitutional(
            structure, composition, atom_ids=atom_ids, seed=seed)
        if activate:
            key = self._lab.project.add_structure(output, activate=True, log=True)
            self._lab.project.history.log[-1].label = (
                f"Build random alloy {key}, seed {int(seed)}")
            self._lab.project.history.log[-1].parameters.update({
                "composition": dict(composition), "seed": int(seed),
                "site_count": record["site_count"]})
        return SurfaceHandle(output, material, self._lab)

    def sample_occupancy(self, material, repeat: Sequence[int] = (2, 2, 2), *,
                         seed: int = 0, activate: bool = True) -> SurfaceHandle:
        from ..structure_builder.alloys import sample_partial_occupancy

        definition = material.definition if isinstance(material, MaterialHandle) else (
            self._lab.library.get(material) if isinstance(material, str) else material)
        if not isinstance(definition, MaterialDefinition):
            raise ApiError("material must be a material id, definition or material handle.")
        output, record = sample_partial_occupancy(definition, repeat, seed=seed)
        if activate:
            key = self._lab.project.add_structure(output, activate=True, log=True)
            self._lab.project.history.log[-1].label = (
                f"Sample partial occupancy for {definition.id}, seed {int(seed)}")
            self._lab.project.history.log[-1].parameters.update({
                "material": definition.id, "seed": int(seed),
                "repeat": list(repeat), "remaining_atoms": record["remaining_atoms"]})
        return SurfaceHandle(output, definition, self._lab)


class MechanicsNamespace:
    def __init__(self, lab: "Lab") -> None:
        self._lab = lab

    @staticmethod
    def _structure(target) -> Structure:
        if isinstance(target, SurfaceHandle):
            return target.structure
        if isinstance(target, Structure):
            return target
        raise ApiError("The mechanics target must be a structure or surface handle.")

    def apply_force(self, target, atom_id: int, force_eV_A: Sequence[float], *,
                    reference_position_A: Optional[Sequence[float]] = None) -> dict:
        from ..physics.external_bias import set_constant_force

        structure = self._structure(target)
        before = structure.copy()
        before_selection = Selection(list(self._lab.project.selection.ids),
                                     self._lab.project.selection.query)
        record = set_constant_force(
            structure, atom_id, force_eV_A,
            reference_position_A=reference_position_A)
        self._lab.project.record_change(
            structure, before, before_selection,
            f"Apply force to atom {int(atom_id)}", "mechanics.constant_force",
            record, result_summary=f"force {record['force_eV_A']} eV/A")
        return record

    def restrain(self, target, atom_id: int, spring_eV_A2: float, *,
                 target_position_A: Optional[Sequence[float]] = None) -> dict:
        from ..physics.external_bias import set_harmonic_restraint

        structure = self._structure(target)
        before = structure.copy()
        before_selection = Selection(list(self._lab.project.selection.ids),
                                     self._lab.project.selection.query)
        record = set_harmonic_restraint(
            structure, atom_id, spring_eV_A2,
            target_position_A=target_position_A)
        self._lab.project.record_change(
            structure, before, before_selection,
            f"Restrain atom {int(atom_id)}", "mechanics.harmonic_restraint",
            record, result_summary=f"spring {record['spring_eV_A2']} eV/A^2")
        return record

    def clear(self, target, atom_ids=None, kind: Optional[str] = None) -> int:
        from ..physics.external_bias import clear_external_biases

        structure = self._structure(target)
        before = structure.copy()
        before_selection = Selection(list(self._lab.project.selection.ids),
                                     self._lab.project.selection.query)
        selected = None if atom_ids is None else [int(value) for value in atom_ids]
        removed = clear_external_biases(structure, atom_ids=selected, kind=kind)
        if removed:
            self._lab.project.record_change(
                structure, before, before_selection,
                "Clear external biases", "mechanics.clear_biases",
                {"atom_ids": selected,
                 "kind": kind}, result_summary=f"removed {removed} biases")
        return removed

    def list(self, target) -> dict:
        from ..physics.external_bias import external_biases

        return external_biases(self._structure(target))


class SearchNamespace:
    def __init__(self, lab: "Lab") -> None:
        self._lab = lab

    def basin_hopping(self, target, model: str = "recommended", settings=None,
                      activate_best: bool = False, **kwargs):
        from ..physics import structure_search
        from ..solvers.classical import ClassicalSolver

        if isinstance(target, SurfaceHandle):
            structure = target.structure
            material = target.material
        elif isinstance(target, Structure):
            structure = target
            material = None
        else:
            raise ApiError("The structure-search target must be a structure or surface handle.")
        if settings is not None and kwargs:
            raise ApiError("Pass either settings or individual structure-search options, not both.")
        if settings is None:
            settings = structure_search.BasinHoppingSettings(**kwargs)
        elif not isinstance(settings, structure_search.BasinHoppingSettings):
            raise ApiError("settings must be a BasinHoppingSettings instance.")
        solver = self._lab._resolve_solver(model, structure, material, "energy")
        if not isinstance(solver, ClassicalSolver):
            raise ApiError(
                f"{solver.name} is not an in-process classical model for structure search.")
        search = structure_search.basin_hopping(structure, solver.potential, settings)
        run_id = f"{int(time.time() * 1000):013d}-{os.urandom(3).hex()}"
        search.diagnostics["run_id"] = run_id
        for key, result in search.results().items():
            result.extra["run_id"] = run_id
            self._lab.project.add_result(f"search::{run_id}::{key}", result)
        for key, stored in search.stored_arrays().items():
            stored.meta["run_id"] = run_id
            self._lab.project.add_array(f"search::{run_id}::{key}", stored)
        if activate_best:
            key = self._lab.project.add_structure(
                search.best.structure.copy(), activate=True, log=True)
            self._lab.project.history.log[-1].label = (
                f"Activate structure-search minimum {key}")
            self._lab.project.history.log[-1].parameters.update({
                "run_id": run_id,
                "energy_eV": search.best.energy_eV,
                "trial": search.best.trial,
            })
        return search

    def runs(self) -> List[str]:
        prefix = "search::"
        ids = {key[len(prefix):].split("::", 1)[0]
               for key in list(self._lab.project.results) + list(self._lab.project.arrays)
               if key.startswith(prefix) and "::" in key[len(prefix):]}
        return sorted(ids, reverse=True)

    def result(self, run_id: Optional[str] = None, quantity: str = "best_energy") -> Result:
        ids = self.runs()
        chosen = run_id or (ids[0] if ids else None)
        if chosen is None:
            raise ApiError("No stored structure-search runs.")
        key = f"search::{chosen}::{quantity}"
        if key not in self._lab.project.results:
            raise ApiError(f"Structure-search run {chosen!r} has no result {quantity!r}.")
        return self._lab.project.results[key]

    def array(self, run_id: Optional[str] = None, name: str = "candidate_positions"):
        ids = self.runs()
        chosen = run_id or (ids[0] if ids else None)
        if chosen is None:
            raise ApiError("No stored structure-search runs.")
        key = f"search::{chosen}::{name}"
        if key not in self._lab.project.arrays:
            raise ApiError(f"Structure-search run {chosen!r} has no stored array {name!r}.")
        return self._lab.project.arrays[key]


class CompareNamespace:
    def __init__(self, lab: "Lab") -> None:
        self._lab = lab

    @staticmethod
    def _structure(target) -> Structure:
        if isinstance(target, SurfaceHandle):
            return target.structure
        if isinstance(target, Structure):
            return target
        raise ApiError("Comparison inputs must be structures or surface handles.")

    def structures(self, before, after, **options):
        from ..physics.structure_comparison import compare_structures

        comparison = compare_structures(
            self._structure(before), self._structure(after), **options)
        run_id = f"{int(time.time() * 1000):013d}-{os.urandom(3).hex()}"
        comparison.metrics["run_id"] = run_id
        for key, result in comparison.results().items():
            result.extra["run_id"] = run_id
            self._lab.project.add_result(f"comparison::{run_id}::{key}", result)
        for key, stored in comparison.stored_arrays().items():
            stored.meta["run_id"] = run_id
            self._lab.project.add_array(f"comparison::{run_id}::{key}", stored)
        return comparison

    def runs(self) -> List[str]:
        prefix = "comparison::"
        ids = {key[len(prefix):].split("::", 1)[0]
               for key in list(self._lab.project.results) + list(self._lab.project.arrays)
               if key.startswith(prefix) and "::" in key[len(prefix):]}
        return sorted(ids, reverse=True)

    def result(self, run_id: Optional[str] = None,
               quantity: str = "rms_displacement") -> Result:
        ids = self.runs()
        chosen = run_id or (ids[0] if ids else None)
        if chosen is None:
            raise ApiError("No stored structure comparisons.")
        key = f"comparison::{chosen}::{quantity}"
        if key not in self._lab.project.results:
            raise ApiError(f"Comparison {chosen!r} has no result {quantity!r}.")
        return self._lab.project.results[key]

    def array(self, run_id: Optional[str] = None, name: str = "displacements"):
        ids = self.runs()
        chosen = run_id or (ids[0] if ids else None)
        if chosen is None:
            raise ApiError("No stored structure comparisons.")
        key = f"comparison::{chosen}::{name}"
        if key not in self._lab.project.arrays:
            raise ApiError(f"Comparison {chosen!r} has no stored array {name!r}.")
        return self._lab.project.arrays[key]


class EnsembleNamespace:
    def __init__(self, lab: "Lab") -> None:
        self._lab = lab

    @staticmethod
    def _structure(target) -> Structure:
        if isinstance(target, SurfaceHandle):
            return target.structure
        if isinstance(target, Structure):
            return target
        raise ApiError("Ensemble inputs must be structures or surface handles.")

    def _models(self, target, models, labels=None):
        from ..solvers.classical import ClassicalSolver

        structure = self._structure(target)
        material = target.material if isinstance(target, SurfaceHandle) else None
        if labels is not None and len(labels) != len(models):
            raise ApiError("labels must contain one name for every ensemble model.")
        chosen_labels = []
        potentials = []
        for index, value in enumerate(models):
            if isinstance(value, ClassicalSolver):
                solver = value
                label = value.name
            elif isinstance(value, str):
                solver = self._lab._resolve_solver(value, structure, material, "energy")
                label = value
            else:
                raise ApiError(
                    "Ensemble models must be registered model names or ClassicalSolver objects.")
            if not isinstance(solver, ClassicalSolver):
                raise ApiError(f"{solver.name} is not an in-process classical model.")
            chosen_labels.append(str(labels[index]) if labels is not None else label)
            potentials.append(solver.potential)
        return chosen_labels, potentials

    def _store(self, kind: str, analysis) -> str:
        run_id = f"{int(time.time() * 1000):013d}-{os.urandom(3).hex()}"
        analysis.metrics["run_id"] = run_id
        analysis.metrics["ensemble_kind"] = kind
        for key, result in analysis.results().items():
            result.extra["run_id"] = run_id
            result.extra["ensemble_kind"] = kind
            self._lab.project.add_result(f"ensemble::{run_id}::{key}", result)
        for key, stored in analysis.stored_arrays().items():
            stored.meta["run_id"] = run_id
            stored.meta["ensemble_kind"] = kind
            self._lab.project.add_array(f"ensemble::{run_id}::{key}", stored)
        return run_id

    def forces(self, target, models, *, labels=None):
        from ..physics.model_ensemble import force_ensemble

        labels, potentials = self._models(target, models, labels)
        analysis = force_ensemble(
            self._structure(target), potentials, labels=labels)
        self._store("forces", analysis)
        return analysis

    def energy_change(self, before, after, models, *, labels=None):
        from ..physics.model_ensemble import energy_change_ensemble

        labels, potentials = self._models(before, models, labels)
        analysis = energy_change_ensemble(
            self._structure(before), self._structure(after), potentials, labels=labels)
        self._store("energy_change", analysis)
        return analysis

    def runs(self) -> List[str]:
        prefix = "ensemble::"
        ids = {key[len(prefix):].split("::", 1)[0]
               for key in list(self._lab.project.results) + list(self._lab.project.arrays)
               if key.startswith(prefix) and "::" in key[len(prefix):]}
        return sorted(ids, reverse=True)

    def result(self, run_id: Optional[str] = None,
               quantity: str = "force_disagreement") -> Result:
        ids = self.runs()
        chosen = run_id or (ids[0] if ids else None)
        if chosen is None:
            raise ApiError("No stored model ensembles.")
        key = f"ensemble::{chosen}::{quantity}"
        if key not in self._lab.project.results:
            raise ApiError(f"Model ensemble {chosen!r} has no result {quantity!r}.")
        return self._lab.project.results[key]

    def array(self, run_id: Optional[str] = None, name: str = "model_forces"):
        ids = self.runs()
        chosen = run_id or (ids[0] if ids else None)
        if chosen is None:
            raise ApiError("No stored model ensembles.")
        key = f"ensemble::{chosen}::{name}"
        if key not in self._lab.project.arrays:
            raise ApiError(f"Model ensemble {chosen!r} has no stored array {name!r}.")
        return self._lab.project.arrays[key]


class IoNamespace:
    def __init__(self, lab: "Lab") -> None:
        self._lab = lab

    def read(self, path: str, fmt: Optional[str] = None) -> Structure:
        return _io.read_structure(path, fmt)

    def write(self, path: str, target, fmt: Optional[str] = None) -> str:
        st = target.structure if isinstance(target, SurfaceHandle) else target
        return _io.write_structure(path, st, fmt)

    def write_image(self, path: str, scan, channel: Optional[str] = None,
                    palette: str = "silver", scale_bar: bool = True, **kwargs) -> str:
        from ..visualization.image import add_scale_bar, write_png, write_tiff16
        from ..visualization.palette import apply_palette
        data = scan.channel(channel)
        if path.lower().endswith((".tif", ".tiff")):
            return write_tiff16(path, data, description=scan.provenance.summary())
        rgb, meta = apply_palette(data, palette, **kwargs)
        if scale_bar:
            rgb, _ = add_scale_bar(rgb, scan.extent_A)
        return write_png(path, rgb, text={
            "software": "Materia",
            "provenance": scan.provenance.summary()[:800],
            "channel": channel or scan.primary_channel,
            "extent_A": str(list(scan.extent_A)),
            "palette_limits": f"{meta['vmin']:.6g}..{meta['vmax']:.6g}",
        })

    def write_csv(self, path: str, columns: Dict[str, Sequence[Any]], comment: str = "") -> str:
        return _io.write_csv(path, columns, comment)

    def write_npz(self, path: str, parts: Sequence[str] = ("structure", "arrays",
                                                            "results", "scans")) -> dict:
        """Write the project's numerical data to a verified ``.npz`` archive."""
        from ..dataio import npz
        return npz.export(path, self._lab.project, tuple(parts))

    def write_notebook(self, path: str, project_path: Optional[str] = None,
                       **options) -> dict:
        """Write a deterministic notebook that reloads and verifies the project."""
        from ..dataio import notebook

        if project_path is None:
            return notebook.export(path, project=self._lab.project, **options)
        return notebook.export(path, project_path=project_path, **options)

    def write_cube(self, path: str, target, data, **kwargs) -> str:
        st = target.structure if isinstance(target, SurfaceHandle) else target
        return _io.write_cube(path, st, data, **kwargs)

    def formats(self) -> dict:
        return {"readable": list(_io.READABLE), "writable": list(_io.WRITABLE)}


class Lab:
    """The object graph a script sees."""

    def __init__(self, project: Optional[Project] = None,
                 library: Optional[MaterialLibrary] = None) -> None:
        self.library = library or default_library()
        self.project = project or Project(library=self.library)
        self.materials = MaterialsNamespace(self)
        self.view = ViewNamespace(self)
        self.microscope = MicroscopeNamespace(self)
        self.measure = MeasureNamespace(self)
        self.alloys = AlloyNamespace(self)
        self.mechanics = MechanicsNamespace(self)
        self.search = SearchNamespace(self)
        self.compare = CompareNamespace(self)
        self.ensembles = EnsembleNamespace(self)
        self.neb = NEBNamespace(self)
        self.dft = DFTNamespace(self)
        self.claims = ClaimsNamespace(self)
        self.eam = EAMNamespace(self)
        self.collisions = CollisionNamespace(self)
        self.lammps = LAMMPSNamespace(self)
        self.electrostatics = ElectrostaticsNamespace(self)
        self.io = IoNamespace(self)
        self.np = np

    def _checkpoint(self, label: str, operation: str, parameters: dict) -> None:
        try:
            self.project.begin(label, operation, parameters)
        except Exception:
            pass

    def create_wafer(self, material: str = "silicon", **kwargs) -> Wafer:
        spec = WaferSpec(material_id=material, **kwargs)
        return self.project.create_wafer(spec)

    def extract_region(self, **kwargs) -> SurfaceHandle:
        region = self.project.extract_region(RegionSpec(**kwargs))
        return SurfaceHandle(region.structure, self.library.get(self.project.wafer.spec.material_id),
                             self)

    @property
    def active_region(self) -> Optional[SurfaceHandle]:
        r = self.project.active_region
        if r is None:
            st = self.project.structure
            if st is None:
                return None
            mat_id = st.info.get("material_id", "silicon")
            return SurfaceHandle(st, self.library.get(mat_id), self)
        return SurfaceHandle(r.structure,
                             self.library.get(self.project.wafer.spec.material_id), self)

    def _resolve_solver(self, model: str, structure: Structure,
                        material: Optional[MaterialDefinition], task: str):
        from ..solvers.classical import lennard_jones, stillinger_weber
        from ..solvers.tight_binding import MODELS, TightBinding, model_for_material
        from ..elements import periodic_table as pt

        if model not in ("recommended", "auto"):
            if model in _registry.available():
                return _registry.create(model)
            if model in MODELS:
                return TightBinding(model)
            raise ApiError(
                f"Unknown model {model!r}. Registered solvers: "
                f"{_registry.available()}; tight-binding models: {sorted(MODELS)}."
            )

        present = sorted({pt.symbol(int(z)) for z in structure.numbers})
        if task in ("relax", "dynamics", "energy", "phonons"):
            from ..physics import eam as eam_module
            from ..physics.potentials import SW_PARAMETERS, StillingerWeber
            from ..physics import rigid_ion
            from ..solvers.classical import ClassicalSolver
            ionic = rigid_ion.select_shipped(present)
            if ionic is not None:
                return ClassicalSolver(rigid_ion.RigidIon(rigid_ion.SHIPPED[ionic]))
            hosts = [e for e in present if e in SW_PARAMETERS]
            others = [e for e in present if e not in SW_PARAMETERS]
            if len(hosts) == 1 and not others:
                return stillinger_weber(hosts[0])
            if len(hosts) == 1 and others:
                try:
                    return ClassicalSolver(
                        StillingerWeber(hosts[0], impurities=others))
                except Exception:
                    pass
            eam_id, _ = eam_module.select_shipped(present)
            if eam_id is not None:
                return ClassicalSolver(eam_module.load_shipped(eam_id))
            if material is not None:
                try:
                    return lennard_jones(material=material)
                except Exception:
                    pass
            raise ApiError(
                f"No recommended {task} model for a system containing "
                f"{', '.join(present)}. Stillinger-Weber covers "
                f"{sorted(SW_PARAMETERS)} as hosts (with radius-scaled impurities); "
                f"rigid-ion models cover {[e['elements'] for e in rigid_ion.catalog()]}; "
                "a Lennard-Jones fit needs a tabulated cohesive energy. Name a model "
                "explicitly or install an external solver."
            )
        from ..solvers.tight_binding import HARRISON_TERM_VALUES
        name = model_for_material(material.id) if material else None
        if name is not None and not (set(present) <= set(MODELS[name].species)):
            extra = sorted(set(present) - set(MODELS[name].species))
            if all(e in HARRISON_TERM_VALUES for e in extra):
                return TightBinding(name, impurities=extra)
            name = None
        if name is None:
            for key, m in MODELS.items():
                if set(present) <= set(m.species):
                    name = key
                    break
        if name is None:
            for key, m in MODELS.items():
                hosts = set(m.species)
                extra = [e for e in present if e not in hosts]
                if (set(present) & hosts) and all(e in HARRISON_TERM_VALUES for e in extra):
                    return TightBinding(key, impurities=extra)
            raise ApiError(
                f"No electronic-structure model covers {', '.join(present)}. "
                f"Available tight-binding models: {sorted(MODELS)}; elements that "
                f"can be added as substitutional impurities: "
                f"{sorted(HARRISON_TERM_VALUES)}."
            )
        return TightBinding(name)

    def energy(self, target, model: str = "recommended",
               material: Optional[MaterialDefinition] = None) -> Result:
        st = target.structure if isinstance(target, SurfaceHandle) else target
        solver = self._resolve_solver(model, st, material, "energy")
        return solver.single_point(st)["energy"]

    def relax(self, target, model: str = "recommended",
              material: Optional[MaterialDefinition] = None,
              fmax: float = 0.02, steps: int = 400, **kwargs) -> Result:
        st = target.structure if isinstance(target, SurfaceHandle) else target
        solver = self._resolve_solver(model, st, material, "relax")
        self._checkpoint(f"Relax with {solver.name}", "solver.relax",
                         {"model": solver.name, "fmax": fmax, "steps": steps})
        out = solver.relax(st, fmax_eV_A=fmax, max_steps=steps, in_place=True, **kwargs)
        for key, res in out.results.items():
            self.project.add_result(f"relax::{key}", res)
        return out.results.get("relaxed_structure", out.results.get("energy"))

    def dynamics(self, target, model: str = "recommended",
                 material: Optional[MaterialDefinition] = None, **kwargs) -> Result:
        st = target.structure if isinstance(target, SurfaceHandle) else target
        solver = self._resolve_solver(model, st, material, "dynamics")
        self._checkpoint(f"Dynamics with {solver.name}", "solver.dynamics",
                         {"model": solver.name, **{k: v for k, v in kwargs.items()}})
        out = solver.dynamics(st, in_place=True, **kwargs)
        for key, res in out.results.items():
            self.project.add_result(f"md::{key}", res)
        return out.results["trajectory"]

    def phonons(self, target, model: str = "recommended",
                material: Optional[MaterialDefinition] = None,
                settings=None, temperatures_K=None, **kwargs):
        from ..physics.phonons import PhononSettings

        st = target.structure if isinstance(target, SurfaceHandle) else target
        if settings is not None and kwargs:
            raise ApiError("Pass either settings or individual phonon options, not both.")
        if settings is None:
            settings = PhononSettings(**kwargs)
        elif not isinstance(settings, PhononSettings):
            raise ApiError("settings must be a PhononSettings instance.")
        solver = self._resolve_solver(model, st, material, "phonons")
        out = solver.phonons(
            st, settings=settings, temperatures_K=temperatures_K)
        for key, res in out.results.items():
            self.project.add_result(f"phonons::{key}", res)
        from ..project_format.arrays import StoredArray

        array_specs = {
            "frequencies": ("Gamma-point frequencies, negative means imaginary", "table"),
            "force_constants": ("harmonic force constants", "table"),
            "eigenvectors": ("mass-weighted eigenvectors, columns are modes", "table"),
            "displacements": ("normal-mode Cartesian displacements", "trajectory"),
            "helmholtz_free_energy_eV": ("vibrational Helmholtz free energy", "table"),
            "internal_energy_eV": ("vibrational internal energy", "table"),
            "entropy_eV_K": ("vibrational entropy", "table"),
            "heat_capacity_eV_K": ("vibrational constant-volume heat capacity", "table"),
        }
        for key, (description, kind) in array_specs.items():
            result = out.results.get(key)
            if result is None or not result.supported or not isinstance(result.value, np.ndarray):
                continue
            self.project.add_array(
                f"phonons::{key}",
                StoredArray(
                    result.value, result.unit, description, kind,
                    {"model": result.provenance.model,
                     "input_digest": result.provenance.inputs_digest,
                     "free_atom_ids": list(result.extra.get("free_atom_ids", []))}))
        return out

    def phonon_dispersion(self, target, model: str = "recommended",
                          material: Optional[MaterialDefinition] = None,
                          settings=None, **kwargs):
        from ..physics.phonon_dispersion import (
            PeriodicPhononSettings,
            periodic_phonon_analysis,
        )
        from ..solvers.classical import ClassicalSolver

        st = target.structure if isinstance(target, SurfaceHandle) else target
        if settings is not None and kwargs:
            raise ApiError(
                "Pass either settings or individual periodic-phonon options, not both.")
        if settings is None:
            settings = PeriodicPhononSettings(**kwargs)
        elif not isinstance(settings, PeriodicPhononSettings):
            raise ApiError("settings must be a PeriodicPhononSettings instance.")
        solver = self._resolve_solver(model, st, material, "phonons")
        if not isinstance(solver, ClassicalSolver):
            raise ApiError(
                f"{solver.name} is not an in-process force model for periodic phonons.")
        analysis = periodic_phonon_analysis(
            st, solver.potential, settings, name=solver.name)
        run_id = f"{int(time.time() * 1000):013d}-{os.urandom(3).hex()}"
        for key, result in analysis.results().items():
            result.extra["run_id"] = run_id
            self.project.add_result(f"phonon-dispersion::{run_id}::{key}", result)
        for key, stored in analysis.stored_arrays().items():
            stored.meta["run_id"] = run_id
            self.project.add_array(f"phonon-dispersion::{run_id}::{key}", stored)
        return analysis

    def solve(self, target, model: str = "recommended",
              material: Optional[MaterialDefinition] = None,
              self_consistent: bool = False, **kwargs):
        st = target.structure if isinstance(target, SurfaceHandle) else target
        solver = self._resolve_solver(model, st, material, "electronic")
        out = solver.eigenstates(st, self_consistent=self_consistent, **kwargs)
        for key, res in out.results.items():
            self.project.add_result(f"electronic::{key}", res)
        return out

    def band_structure(self, target, model: str = "recommended", **kwargs):
        st = target.structure if isinstance(target, SurfaceHandle) else target
        material = getattr(target, "material", None)
        solver = self._resolve_solver(model, st, material, "electronic")
        return solver.band_structure(st, **kwargs)

    def solvers(self) -> List[dict]:
        return _registry.describe_all()

    def select(self, selection: Union[Selection, Sequence[int]]) -> Selection:
        sel = selection if isinstance(selection, Selection) else Selection(list(selection))
        self.project.selection = sel
        return sel

    @property
    def selection(self) -> Selection:
        return self.project.selection

    def save_checkpoint(self, name: str, note: str = ""):
        return self.project.save_checkpoint(name, note)

    def restore_checkpoint(self, name: str) -> None:
        self.project.restore_checkpoint(name)

    def save(self, path: str) -> str:
        return self.project.save(path)

    def undo(self) -> str:
        return self.project.undo()

    def redo(self) -> str:
        return self.project.redo()


def build_namespace(lab: Lab) -> Dict[str, Any]:
    """The globals dict handed to a user script."""
    from ..structure_builder import defects as defects_module
    from ..structure_builder.reconstruction import (
        apply_reconstruction,
        available_reconstructions,
    )
    return {
        "lab": lab,
        "project": lab.project,
        "materials": lab.materials,
        "view": lab.view,
        "microscope": lab.microscope,
        "measure": lab.measure,
        "alloys": lab.alloys,
        "mechanics": lab.mechanics,
        "search": lab.search,
        "compare": lab.compare,
        "ensembles": lab.ensembles,
        "phonons": lab.phonons,
        "phonon_dispersion": lab.phonon_dispersion,
        "neb": lab.neb,
        "dft": lab.dft,
        "claims": lab.claims,
        "eam": lab.eam,
        "collisions": lab.collisions,
        "lammps": lab.lammps,
        "LAMMPSRunFailed": LAMMPSRunFailed,
        "DFTRelaxationFailed": DFTRelaxationFailed,
        "DFTDOSFailed": DFTDOSFailed,
        "DFTBandsFailed": DFTBandsFailed,
        "DFTEOSFailed": DFTEOSFailed,
        "DFTLDOSFailed": DFTLDOSFailed,
        "electrostatics": lab.electrostatics,
        "io": lab.io,
        "solvers": _registry,
        "defects": defects_module,
        "build": {
            "bulk": bulk, "surface": make_surface, "passivate": passivate,
            "strain": apply_strain, "interplanar_spacing": lattice_planes,
            "reconstruct": apply_reconstruction,
            "reconstructions": available_reconstructions,
        },
        "Selection": Selection,
        "Capability": Capability,
        "ApiError": ApiError,
        "UnsupportedRequest": UnsupportedRequest,
        "np": np,
        "__materia_version__": __import__("materia").__version__,
    }
