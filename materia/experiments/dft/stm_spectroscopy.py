"""STM spectroscopy and tip models from GPAW wavefunctions.

Built on the LDOS experiment (:mod:`.ldos`), whose frozen specification is
reused unchanged; this module adds its own versioned specification for two
quantities computed in the same non-self-consistent GPAW step:

p-wave tip (Chen's derivative rule)
    ``M(r) = sum_states w |d psi/dx|^2 + |d psi/dy|^2`` over the LDOS window:
    the Tersoff-Hamann current for a tip whose apex state has p_x and p_y
    character (C. J. Chen, Phys. Rev. B 42 (1990) 8841).  It is the usual
    first model of a CO-terminated tip (L. Gross et al., Phys. Rev. Lett. 107
    (2011) 086101).  Derivatives are central differences of the
    pseudo-wavefunction on the coarse grid, converted to Cartesian axes.

Energy-resolved LDOS for dI/dV
    ``rho(E, r) = sum_states w g_sigma(E - e) |psi|^2`` with a normalised
    Gaussian of width ``sigma`` at each requested energy about the Fermi
    level.  In the Tersoff-Hamann picture ``dI/dV(V, r)`` at low bias is
    proportional to ``rho(E_F + eV, r)``: slices give dI/dV maps and columns
    give point spectra.

Every slice of the stack is checked against the broadened state count
computed independently from the returned eigenvalues, and the p-tip map is
checked to be finite and non-negative.  No current in amperes is computed.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ...project_format.arrays import StoredArray
from ...provenance import Fidelity, Origin, Provenance, Result
from ...solvers.gpaw_driver import runner
from . import run as ground
from . import ldos as L
from . import spec as specs

SCHEMA = "materia.dft.stm-spectroscopy"
VERSION = "1.0"
TIPS = ("s", "p")
REFERENCES = ["C. J. Chen, Phys. Rev. B 42 (1990) 8841",
              "J. Tersoff and D. R. Hamann, Phys. Rev. B 31 (1985) 805",
              "L. Gross et al., Phys. Rev. Lett. 107 (2011) 086101"]


class SpectroscopyError(ValueError):
    pass


@dataclass(frozen=True)
class SpectroscopySpec:
    ldos: L.LDOSSpec
    tip: str = "p"
    energies_eV: tuple = ()
    broadening_eV: float = 0.1
    schema: str = SCHEMA
    version: str = VERSION

    def as_dict(self) -> dict:
        return {"schema": self.schema, "version": self.version, "ldos": self.ldos.as_dict(),
                "tip": self.tip, "energies_eV": list(self.energies_eV),
                "broadening_eV": self.broadening_eV}


def build(ldos_spec: L.LDOSSpec, tip: str = "p", energies_eV: Sequence[float] = (),
          broadening_eV: float = 0.1) -> SpectroscopySpec:
    if tip not in TIPS:
        raise SpectroscopyError(f"tip must be one of {TIPS}.")
    energies = tuple(float(e) for e in energies_eV)
    if energies and not 0.005 <= broadening_eV <= 1.0:
        raise SpectroscopyError("broadening_eV must lie between 0.005 and 1 eV.")
    if len(energies) > 200:
        raise SpectroscopyError("At most 200 energies may be requested.")
    if tip == "s" and not energies:
        raise SpectroscopyError("Nothing to compute: an s-wave tip without energies is the "
                                "plain LDOS experiment.")
    if energies:
        _, high = ldos_spec.energy_min_eV, ldos_spec.energy_max_eV
        top = max(energies) + 5 * broadening_eV
        if top > 20:
            raise SpectroscopyError("Energies more than 20 eV from the Fermi level are refused.")
    return SpectroscopySpec(ldos_spec, tip, energies, float(broadening_eV))


def worker_job(spec: SpectroscopySpec) -> Dict[str, Any]:
    job = L.worker_job(spec.ldos)
    if spec.tip == "p":
        job["ldos"]["tip"] = "p"
    if spec.energies_eV:
        job["ldos"]["stack_energies"] = list(spec.energies_eV)
        job["ldos"]["broadening"] = spec.broadening_eV
    return job


def broadened_count(eigenvalues: np.ndarray, weights: np.ndarray, fermi: float,
                    energies: Sequence[float], sigma: float) -> np.ndarray:
    """States per cell per eV at each energy, both spins, from eigenvalues [s, k, n]."""
    relative = np.asarray(eigenvalues, dtype=float) - fermi
    degeneracy = 2.0 / relative.shape[0]
    out = []
    for energy in energies:
        gauss = np.exp(-0.5 * ((energy - relative) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))
        gauss = np.where(np.abs(energy - relative) < 5 * sigma, gauss, 0.0)
        out.append(degeneracy * float(np.einsum("k,skn->", np.asarray(weights), gauss)))
    return np.array(out)


@dataclass
class SpectroscopyOutcome:
    spec: SpectroscopySpec
    status: str
    reason: str = ""
    results: Dict[str, Result] = field(default_factory=dict)
    arrays: Dict[str, StoredArray] = field(default_factory=dict)
    wall_time_s: float = 0.0


def execute(spec: SpectroscopySpec, environment=None, *, timeout_s: Optional[float] = None,
            cancelled=None) -> SpectroscopyOutcome:
    environment = specs._environment(environment)
    L.require(spec.ldos, environment)
    started = time.perf_counter()
    job = worker_job(spec)
    out = SpectroscopyOutcome(spec, L.STATUS_FAILED)
    run = runner.run_job(job, environment, worker=ground.WORKER, cancelled=cancelled,
                         timeout_s=timeout_s)
    out.wall_time_s = time.perf_counter() - started
    if run.status != runner.STATUS_CONVERGED:
        out.reason = f"The calculation did not complete: {run.error or run.status}."
        return out
    arrays = {k: np.asarray(v) for k, v in run.arrays.items()}
    try:
        data = L._validate(spec.ldos, run, job, arrays)
    except L._Invalid as exc:
        out.reason = f"{exc} Nothing was kept."
        return out
    shape = data["shape"]
    meta = {"grid_shape": list(shape), "cell_A": data["cell"].tolist(),
            "window_eV": [spec.ldos.energy_min_eV, spec.ldos.energy_max_eV],
            "reference": "self-consistent Fermi level"}
    prov = Provenance(
        model="gpaw/stm-spectroscopy", fidelity=Fidelity.TIER3_EXTERNAL, origin=Origin.CALCULATED,
        approximations=L._approximations(spec.ldos)[:3] + [
            "p-wave tip by Chen's derivative rule from finite differences of the "
            "pseudo-wavefunction on the coarse grid." if spec.tip == "p" else "s-wave tip.",
            f"Energy-resolved LDOS with Gaussian broadening {spec.broadening_eV:g} eV; dI/dV "
            "taken proportional to it (Tersoff-Hamann, low bias, constant tip DOS)."
            if spec.energies_eV else "No energy-resolved stack requested."],
        parameters={"tip": spec.tip, "energies_eV": list(spec.energies_eV),
                    "broadening_eV": spec.broadening_eV, "ldos_spec_digest": spec.ldos.digest},
        references=list(REFERENCES))
    out.arrays["ldos"] = StoredArray(np.ascontiguousarray(data["total"]), "states/A^3",
                                     "s-wave LDOS in the window", "volumetric", dict(meta))
    if spec.tip == "p":
        p_map = np.asarray(arrays.get("ldos_ptip"), dtype=float)
        if p_map.shape != shape or not np.all(np.isfinite(p_map)):
            out.reason = "The p-tip map is missing, misshapen or not finite. Nothing was kept."
            return out
        if float(p_map.min()) < -1e-12 * max(1e-300, float(np.abs(p_map).max())):
            out.reason = "The p-tip map is negative somewhere. Nothing was kept."
            return out
        out.arrays["ptip"] = StoredArray(np.ascontiguousarray(p_map), "states/A^5",
                                         "Chen p-wave tip map: sum |dpsi/dx|^2 + |dpsi/dy|^2",
                                         "volumetric", dict(meta))
    if spec.energies_eV:
        stack = np.asarray(arrays.get("ldos_stack"), dtype=float)
        if stack.shape != (len(spec.energies_eV),) + tuple(shape) or not np.all(np.isfinite(stack)):
            out.reason = "The energy-resolved stack is missing or misshapen. Nothing was kept."
            return out
        highest = data["highest_above_EF"]
        if max(spec.energies_eV) + 5 * spec.broadening_eV >= highest:
            out.reason = (f"The highest computed band reaches {highest:.3f} eV above the Fermi "
                          "level, below the top of the requested energies plus five widths; "
                          "raise n_bands. Nothing was kept.")
            return out
        volume = abs(float(np.linalg.det(data["cell"])))
        integrals = stack.reshape(len(spec.energies_eV), -1).sum(axis=1) * volume / stack[0].size
        expected = broadened_count(data["eigen"], data["weights"], data["fermi_level_eV"],
                                   spec.energies_eV, spec.broadening_eV)
        for e, got, want in zip(spec.energies_eV, integrals, expected):
            if want > 1e-6 and not 0.5 < got / want < 1.5:
                out.reason = (f"At {e:g} eV the stack integrates to {got:.4g} states/eV against "
                              f"{want:.4g} from the eigenvalues. Nothing was kept.")
                return out
        out.arrays["stack"] = StoredArray(np.ascontiguousarray(stack), "states/(eV A^3)",
                                          "Energy-resolved LDOS [energy, a, b, c]", "volumetric",
                                          dict(meta, energies_eV=list(spec.energies_eV)))
        out.results["dos"] = Result("broadened_dos", expected.tolist(), "states/eV/cell", prov,
                                    extra={"energies_eV": list(spec.energies_eV),
                                           "stack_integrals": integrals.tolist()})
    out.results["record"] = Result("stm_spectroscopy", {"spec": spec.as_dict(),
                                                        "fermi_level_eV": data["fermi_level_eV"],
                                                        "states_in_window": data["count"],
                                                        "radii_A": data["radii"]},
                                   "mixed", prov)
    out.status = L.STATUS_COMPLETE
    return out


def point_spectrum(stack: np.ndarray, cell: np.ndarray, energies: Sequence[float],
                   x_A: float, y_A: float, z_A: float) -> Dict[str, List[float]]:
    """dI/dV-proportional spectrum at one tip position by trilinear interpolation."""
    stack = np.asarray(stack, dtype=float)
    cell = np.asarray(cell, dtype=float)
    fractional = np.linalg.solve(cell.T, np.array([x_A, y_A, z_A]))
    shape = np.array(stack.shape[1:])
    index = fractional * shape
    base = np.floor(index).astype(int)
    frac = index - base
    values = np.zeros(stack.shape[0])
    for corner in range(8):
        offset = np.array([(corner >> b) & 1 for b in range(3)])
        weight = np.prod(np.where(offset, frac, 1 - frac))
        i, j, k = (base + offset) % shape
        values += weight * stack[:, i, j, k]
    return {"energies_eV": [float(e) for e in energies], "ldos_per_eV": values.tolist()}
