"""Point-charge electrostatics as a solver.

Wraps :mod:`materia.physics.electrostatics` in the solver interface so that
its results carry the same provenance, convergence and refusal records as
every other model.  The solver computes energy, forces, site potentials and
site fields for charges taken from an explicit charge model.  It does not
relax or run dynamics: point charges without a short-range repulsion have no
stable ionic arrangement.  Relaxation and dynamics of ionic solids use a
rigid-ion model, :mod:`materia.physics.rigid_ion`, which adds that repulsion.
"""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional

import numpy as np

from ..core_model.structure import Structure
from ..physics import electrostatics as es
from ..provenance import Convergence, Fidelity, Origin, Provenance, Result, digest, unsupported
from .base import Capability, Solver, SolverResult, SupportReport

SOLVER_NAME = "electrostatics/ewald"

BOUNDARY_TEXT = {
    "bulk-3d": "three-dimensional periodic",
    "slab-2d": "two-dimensional periodic slab, open along the normal",
    "isolated": "isolated cluster, open boundaries",
    "wire-1d": "one-dimensional periodic wire",
}


def input_digest(structure: Structure, charges: np.ndarray,
                 settings: es.EwaldSettings) -> str:
    """Fingerprint of everything an electrostatics result depends on."""
    return digest({
        "ids": structure.ids.tolist(),
        "numbers": structure.numbers.tolist(),
        "positions": np.round(structure.positions, 10).tolist(),
        "cell": structure.cell.as_dict(),
        "charges": np.round(np.asarray(charges, dtype=float), 12).tolist(),
        "settings": settings.as_dict(),
    })


def structure_digest(structure: Structure, settings: Optional[es.EwaldSettings] = None
                     ) -> Optional[str]:
    """The input digest for a structure's own charge model, or ``None``.

    ``None`` means the structure has no usable charge model, so no stored
    result can belong to it.
    """
    model = es.stored_charge_model(structure)
    if model is None:
        return None
    try:
        charges = model.charges(structure)
    except es.ChargeModelError:
        return None
    return input_digest(structure, charges, settings or es.EwaldSettings())


class ElectrostaticsSolver(Solver):
    """Ewald, slab-corrected Ewald or direct Coulomb sums of point charges."""

    name = SOLVER_NAME
    fidelity = Fidelity.TIER1_CLASSICAL
    capabilities = {Capability.ENERGY, Capability.FORCES, Capability.SITE_POTENTIAL,
                    Capability.CHARGED_SYSTEM}
    description = ("Long-range Coulomb energy, forces, site potentials and site fields "
                   "of explicit point charges. Ewald summation for crystals, "
                   "Yeh-Berkowitz corrected Ewald for slabs, direct sums for clusters.")

    def supports(self, structure: Structure,
                 model: Optional[es.ChargeModel] = None,
                 settings: Optional[es.EwaldSettings] = None) -> SupportReport:
        try:
            model = model or es.stored_charge_model(structure)
        except es.ChargeModelError as exc:
            return SupportReport(False, [str(exc)], [])
        if model is None:
            return SupportReport(False, [
                "No point-charge model is assigned to this structure. Electrostatics "
                "needs explicit charges: assign per-element or per-atom charges with "
                "a stated source, or choose the formal-point-ion model."], [])
        try:
            settings = (settings or es.EwaldSettings()).validate()
            charges = model.charges(structure)
        except (es.ChargeModelError, ValueError) as exc:
            return SupportReport(False, [str(exc)], [])
        geometry = es.classify(structure.cell)
        reason = es.refusal(structure, charges, geometry, settings)
        if reason:
            return SupportReport(False, [reason], [])
        warnings: List[str] = []
        if len(structure) and not np.any(charges):
            warnings.append("Every point charge is zero, so every electrostatic "
                            "quantity is exactly zero.")
        return SupportReport(True, [], warnings)

    def run(self, structure: Structure, task: str = "energy", **kwargs) -> SolverResult:
        if task == "energy":
            return self.single_point(structure, **kwargs)
        raise ValueError(f"Unknown task {task!r} for {self.name}. Supported: energy.")

    def relax(self, structure: Structure, **kwargs) -> SolverResult:
        out = SolverResult(solver=self.name, structure=structure)
        out.results["relaxed_structure"] = self.require(
            Capability.RELAX,
            "point charges alone have no short-range repulsion, so an ionic "
            "arrangement collapses under them. Relax with a rigid-ion model "
            "instead, such as rigid-ion/bks-silica for SiO2")
        return out

    def dynamics(self, structure: Structure, **kwargs) -> SolverResult:
        out = SolverResult(solver=self.name, structure=structure)
        out.results["trajectory"] = self.require(
            Capability.DYNAMICS,
            "point charges alone have no short-range repulsion. Run dynamics with "
            "a rigid-ion model instead, such as rigid-ion/bks-silica for SiO2")
        return out

    def _provenance(self, model: es.ChargeModel, settings: es.EwaldSettings,
                    output: es.ElectrostaticsOutput, inputs_digest: str) -> Provenance:
        geometry = output.geometry
        approximations = [
            model.describe(),
            "Point charges in vacuum: no charge penetration, no electronic "
            "polarisation and no dielectric screening. Charges are fixed and do "
            "not respond to the field they create.",
        ]
        if geometry.kind == "bulk-3d":
            approximations.append(
                "Tin-foil (conducting) surrounding at infinity."
                if settings.surrounding == "tinfoil" else
                "Vacuum surrounding of a spherical macroscopic crystal; the surface "
                "term depends on which periodic image of each atom is stored.")
            if settings.background == "uniform":
                approximations.append(
                    "Uniform neutralising background included. The energy of a charged "
                    "periodic cell depends on cell size (Makov and Payne 1995) and no "
                    "finite-size correction is applied.")
        elif geometry.kind == "slab-2d":
            approximations.append(
                "Slab evaluated as a three-dimensional Ewald sum in an internally "
                "enlarged cell with the Yeh-Berkowitz dipole correction; the stored "
                "cell height is not used.")
        boundary = BOUNDARY_TEXT.get(geometry.kind, geometry.kind)
        if geometry.kind == "bulk-3d":
            boundary += f", {settings.surrounding} surrounding, background {settings.background}"
        references = list(es.REFERENCES)
        if model.source:
            references.append(f"Point charges: {model.source}")
        parameters = {k: v for k, v in output.parameters.items()}
        parameters["geometry"] = geometry.kind
        parameters["charge_model"] = model.as_dict()
        parameters["settings"] = settings.as_dict()
        return Provenance(
            model=self.name,
            fidelity=self.fidelity,
            origin=Origin.CALCULATED if output.converged else Origin.ESTIMATED,
            approximations=approximations,
            tolerances={"accuracy": settings.accuracy,
                        "energy_tolerance_eV_per_atom": settings.energy_tolerance_eV_per_atom,
                        "force_tolerance_eV_A": settings.force_tolerance_eV_A},
            boundary_conditions=boundary,
            parameters=parameters,
            references=references,
            dataset=model.kind,
            inputs_digest=inputs_digest,
            notes=output.convergence_message,
        )

    def single_point(self, structure: Structure,
                     charge_model: Optional[es.ChargeModel] = None,
                     settings: Optional[es.EwaldSettings] = None,
                     progress: Optional[Callable[[float, str], Optional[bool]]] = None,
                     cancelled: Optional[Callable[[], bool]] = None) -> SolverResult:
        t0 = time.perf_counter()
        out = SolverResult(solver=self.name, structure=structure)
        try:
            settings = (settings or es.EwaldSettings()).validate()
        except ValueError as exc:
            out.results["energy"] = self._refused(str(exc), "invalid-settings")
            out.log.append(str(exc))
            return out
        report = self.supports(structure, charge_model, settings)
        if not report.ok:
            out.results["energy"] = self._refused("; ".join(report.blocking), "refused")
            out.log.extend(report.blocking)
            return out
        model = charge_model or es.stored_charge_model(structure)
        try:
            output = es.compute(structure, model, settings, progress=progress,
                                cancelled=cancelled)
        except es.ElectrostaticsCancelled as exc:
            out.results["energy"] = self._refused(str(exc), "cancelled")
            out.log.append(str(exc))
            out.wall_time_s = time.perf_counter() - t0
            return out
        except (es.UndefinedElectrostatics, es.ChargeModelError) as exc:
            out.results["energy"] = self._refused(str(exc), "refused")
            out.log.append(str(exc))
            return out

        inputs = input_digest(structure, output.charges_e, settings)
        prov = self._provenance(model, settings, output, inputs)
        check = output.check or {}
        conv = Convergence(
            converged=output.converged,
            iterations=0,
            residual=check.get("energy_difference_eV"),
            residual_metric=("|E(alpha) - E(0.8 alpha)| in eV" if check else
                             "direct sum, no truncation" if output.geometry.kind == "isolated"
                             else "not checked"),
            tolerance=check.get("energy_tolerance_eV"),
            message=output.convergence_message,
        )
        n = len(structure)
        ids = structure.ids.tolist()
        accounting = es.charge_accounting(structure, output.charges_e)
        uncertainty = check.get("energy_difference_eV") if check else None
        status = "converged" if output.converged else "not-converged"
        common = {"atom_ids": ids, "geometry": output.geometry.as_dict(), "status": status,
                  "inputs_digest": inputs}
        out.results["energy"] = Result(
            "electrostatic_energy", float(output.energy_eV), "eV", prov,
            uncertainty=uncertainty,
            uncertainty_kind="model-spread" if uncertainty is not None else "",
            convergence=conv,
            extra={**common,
                   "components_eV": dict(output.components_eV),
                   "energy_per_atom_eV": float(output.energy_eV) / max(1, n),
                   "charge_accounting": accounting,
                   "check": check,
                   "log": list(output.log)})
        forces = output.forces_eV_A
        out.results["forces"] = Result(
            "electrostatic_forces", forces, "eV/A", prov, convergence=conv,
            extra={**common,
                   "max_force_eV_A": float(np.linalg.norm(forces, axis=1).max()) if n else 0.0,
                   "net_force_eV_A": forces.sum(axis=0).tolist() if n else [0.0, 0.0, 0.0]})
        out.results["site_potential"] = Result(
            "site_potential", output.site_potential_V, "V", prov, convergence=conv,
            extra={**common, "note": "Potential at each nucleus from every other charge "
                                     "and all periodic images; the atom's own charge "
                                     "is excluded."})
        field = output.site_field_V_A
        out.results["site_field"] = Result(
            "site_field", field, "V/A", prov, convergence=conv,
            extra={**common,
                   "max_field_V_A": float(np.linalg.norm(field, axis=1).max()) if n else 0.0,
                   "note": "Field at each nucleus excluding the atom's own charge."})
        out.results["point_charges"] = Result(
            "point_charges", output.charges_e, "e", prov, convergence=conv,
            extra={**common, "charge_model": model.as_dict()})
        out.convergence = conv
        out.log.extend(report.warnings + list(output.log))
        out.wall_time_s = time.perf_counter() - t0
        return out

    def _refused(self, reason: str, status: str) -> Result:
        result = unsupported("electrostatic_energy", self.name, reason, unit="eV",
                             fidelity=self.fidelity)
        result.extra["status"] = status
        return result
