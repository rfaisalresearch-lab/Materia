"""Tier-2 semi-empirical tight binding.

Models
------
``sp3s*``
    Orthogonal nearest-neighbour sp3s* model of Vogl, Hjalmarson and Dow,
    J. Phys. Chem. Solids 44 (1983) 365.  Parameterised here for Si, Ge and
    GaAs.  The published ``V_ss``, ``V_xx``, ``V_xy``, ``V_sapc``, ``V_pasc``
    parameters are the four-neighbour sums of the underlying Slater-Koster
    two-centre integrals; this module converts them back to
    ``(ss-sigma), (sp-sigma), (pp-sigma), (pp-pi)`` so the Hamiltonian can be
    built for *arbitrary* geometries -- surfaces, defects and relaxed
    structures -- rather than only the ideal crystal.

    The conversion uses the tetrahedral direction cosines:

        (ss-sigma) = V_ss / 4
        (sp-sigma) = sqrt(3) * V_sapc / 4
        (pp-sigma) = (V_xx + 2 V_xy) / 4
        (pp-pi)    = (V_xx -   V_xy) / 4

``pz``
    Single-orbital nearest-neighbour model for graphene and other sp2 carbon,
    with the standard ``t = -2.7 eV``.

Approximations, stated plainly
------------------------------
* Orthogonal basis: overlap is assumed to be the identity.
* Two-centre, nearest-neighbour only.
* Distance dependence of the hoppings follows Harrison's ``d^-2`` scaling.
  This is a rule of thumb: the parameters were fitted at the equilibrium bond
  length only.
* The Hamiltonian is *not* self-consistent.  Charge transfer, band bending,
  surface dipoles and the response to a net charge are therefore not
  described.  Asking for a self-consistent result returns an explicit
  unsupported response.
* Surface dangling-bond states appear, but their absolute energies are not
  quantitative: the parameterisation was fitted to bulk bands.
* No spin-orbit coupling, no spin polarisation, no excited states.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core_model.structure import Structure
from ..core_model.units import BOLTZMANN_EV_K
from ..elements import periodic_table as pt
from ..physics.neighbors import neighbor_list
from ..provenance import Convergence, Fidelity, Origin, Provenance, Result, unsupported
from .base import Capability, Solver, SolverResult, SupportReport, check_electron_bookkeeping

ORBITALS_SP3S = ("s", "px", "py", "pz", "s*")
ORBITALS_PZ = ("pz",)


@dataclass(frozen=True)
class SKIntegrals:
    """Slater-Koster two-centre integrals at the reference bond length."""

    ss_sigma: float
    sp_sigma: float
    ps_sigma: float
    pp_sigma: float
    pp_pi: float
    sstar_p_sigma: float = 0.0
    p_sstar_sigma: float = 0.0
    d0_A: float = 2.35


@dataclass(frozen=True)
class TBSpecies:
    symbol: str
    onsite: Dict[str, float]
    valence_electrons: int


@dataclass(frozen=True)
class TBModel:
    """A complete tight-binding parameterisation."""

    name: str
    orbitals: Tuple[str, ...]
    species: Dict[str, TBSpecies]
    bonds: Dict[Tuple[str, str], SKIntegrals]
    cutoff_A: float
    reference: str
    scaling: str = "harrison-d2"
    notes: str = ""


def _vogl_to_sk(V_ss, V_xx, V_xy, V_sapc, V_pasc, V_sstar_a_pc, V_pa_sstar_c, d0):
    root3 = np.sqrt(3.0)
    return dict(
        ss_sigma=V_ss / 4.0,
        sp_sigma=root3 * V_sapc / 4.0,
        ps_sigma=root3 * V_pasc / 4.0,
        pp_sigma=(V_xx + 2.0 * V_xy) / 4.0,
        pp_pi=(V_xx - V_xy) / 4.0,
        sstar_p_sigma=root3 * V_sstar_a_pc / 4.0,
        p_sstar_sigma=root3 * V_pa_sstar_c / 4.0,
        d0_A=d0,
    )


VOGL_REFERENCE = "P. Vogl, H. P. Hjalmarson and J. D. Dow, J. Phys. Chem. Solids 44 (1983) 365"


def _homopolar(symbol: str, Es, Ep, Estar, V_ss, V_xx, V_xy, V_sp, V_sstar_p, d0, name):
    sk = SKIntegrals(**_vogl_to_sk(V_ss, V_xx, V_xy, V_sp, V_sp, V_sstar_p, V_sstar_p, d0))
    sp = TBSpecies(symbol, {"s": Es, "px": Ep, "py": Ep, "pz": Ep, "s*": Estar}, 4)
    return TBModel(
        name=name, orbitals=ORBITALS_SP3S, species={symbol: sp},
        bonds={(symbol, symbol): sk}, cutoff_A=d0 * 1.25, reference=VOGL_REFERENCE,
    )


MODELS: Dict[str, TBModel] = {}

MODELS["sp3s*-Si"] = _homopolar(
    "Si", Es=-4.2000, Ep=1.7150, Estar=6.6850,
    V_ss=-8.3000, V_xx=1.7150, V_xy=4.5750, V_sp=5.7292, V_sstar_p=5.3749,
    d0=5.4310205 * math.sqrt(3.0) / 4.0, name="sp3s*-Si",
)
MODELS["sp3s*-Ge"] = _homopolar(
    "Ge", Es=-5.8800, Ep=1.6100, Estar=6.3900,
    V_ss=-6.7800, V_xx=1.6100, V_xy=4.9000, V_sp=5.4649, V_sstar_p=5.2191,
    d0=5.65785 * math.sqrt(3.0) / 4.0, name="sp3s*-Ge",
)

_gaas_sk = SKIntegrals(**_vogl_to_sk(
    V_ss=-6.4513, V_xx=1.9546, V_xy=5.0779, V_sapc=4.4800, V_pasc=5.7839,
    V_sstar_a_pc=4.8422, V_pa_sstar_c=4.8077, d0=5.65325 * math.sqrt(3.0) / 4.0,
))
MODELS["sp3s*-GaAs"] = TBModel(
    name="sp3s*-GaAs",
    orbitals=ORBITALS_SP3S,
    species={
        "As": TBSpecies("As", {"s": -8.3431, "px": 1.0414, "py": 1.0414, "pz": 1.0414,
                               "s*": 8.5914}, 5),
        "Ga": TBSpecies("Ga", {"s": -2.6569, "px": 3.6686, "py": 3.6686, "pz": 3.6686,
                               "s*": 6.7386}, 3),
    },
    bonds={("As", "Ga"): _gaas_sk},
    cutoff_A=5.65325 * math.sqrt(3.0) / 4.0 * 1.25,
    reference=VOGL_REFERENCE,
    notes="Anion (As) and cation (Ga) on-site energies follow Vogl's convention.",
)

MODELS["pz-graphene"] = TBModel(
    name="pz-graphene",
    orbitals=ORBITALS_PZ,
    species={"C": TBSpecies("C", {"pz": 0.0}, 1)},
    bonds={("C", "C"): SKIntegrals(0.0, 0.0, 0.0, 0.0, -2.7, d0_A=2.4612 / math.sqrt(3.0))},
    cutoff_A=1.75,
    reference="A. H. Castro Neto et al., Rev. Mod. Phys. 81 (2009) 109",
    notes="Nearest-neighbour pi model: t = -2.7 eV, electron-hole symmetric.",
)


HARRISON_TERM_VALUES: Dict[str, Dict[str, float]] = {
    "B":  {"s": -13.46, "p": -5.75},
    "C":  {"s": -17.52, "p": -8.97},
    "N":  {"s": -23.04, "p": -11.47},
    "O":  {"s": -29.14, "p": -14.13},
    "Al": {"s": -10.11, "p": -4.86},
    "Si": {"s": -14.79, "p": -7.59},
    "P":  {"s": -19.22, "p": -9.54},
    "S":  {"s": -24.02, "p": -11.60},
    "Ga": {"s": -11.37, "p": -4.90},
    "Ge": {"s": -15.16, "p": -7.33},
    "As": {"s": -19.37, "p": -9.17},
    "Se": {"s": -23.65, "p": -10.97},
    "In": {"s": -10.12, "p": -4.69},
    "Sn": {"s": -13.04, "p": -6.76},
    "Sb": {"s": -16.03, "p": -8.14},
}


def impurity_onsite_shift(host: str, impurity: str) -> Dict[str, float]:
    """Estimated on-site energy shift of a substitutional impurity, in eV.

    The shift is the difference of the free-atom term values of the impurity
    and the host, applied to the s and p (and s*) on-site energies.  This is
    the standard first approximation for a substitutional impurity potential
    in an empirical tight-binding model.

    What it does and does not give
    ------------------------------
    It reproduces the *direction* and rough size of the impurity potential, so
    a donor appears as an attractive site and an acceptor as a repulsive one,
    and the perturbation is visible in the local density of states.  It does
    **not** reproduce shallow-donor binding energies: the 45 meV binding energy
    of P in Si comes from the long-range Coulomb tail of the ionised donor and
    a central-cell correction, neither of which a short-range on-site shift in
    a finite non-self-consistent supercell contains.  Any donor level this
    model produces is therefore reported with that caveat attached.
    """
    h = HARRISON_TERM_VALUES.get(pt.symbol(host))
    i = HARRISON_TERM_VALUES.get(pt.symbol(impurity))
    if h is None or i is None:
        missing = [x for x, v in ((host, h), (impurity, i)) if v is None]
        raise KeyError(
            f"No Harrison free-atom term values tabulated for {', '.join(missing)}. "
            f"Tabulated: {sorted(HARRISON_TERM_VALUES)}."
        )
    ds = i["s"] - h["s"]
    dp = i["p"] - h["p"]
    return {"s": ds, "px": dp, "py": dp, "pz": dp, "s*": dp}


def model_for_material(material_id: str) -> Optional[str]:
    return {
        "silicon": "sp3s*-Si",
        "germanium": "sp3s*-Ge",
        "gallium_arsenide": "sp3s*-GaAs",
        "graphene": "pz-graphene",
    }.get(material_id)


def _slater_koster_block(model: TBModel, sym_i: str, sym_j: str, d_vec: np.ndarray,
                         d: float) -> np.ndarray:
    """Hopping block between all orbitals of atom i and atom j."""
    key = (sym_i, sym_j)
    if key in model.bonds:
        sk = model.bonds[key]
        flip = False
    elif (sym_j, sym_i) in model.bonds:
        sk = model.bonds[(sym_j, sym_i)]
        flip = True
    else:
        raise KeyError(f"No tight-binding hopping parameters for {sym_i}-{sym_j}")

    scale = (sk.d0_A / d) ** 2 if model.scaling == "harrison-d2" else 1.0
    l, m, n = d_vec / d
    orbs = model.orbitals
    B = np.zeros((len(orbs), len(orbs)))
    idx = {o: k for k, o in enumerate(orbs)}

    ss = sk.ss_sigma * scale
    ppS = sk.pp_sigma * scale
    ppP = sk.pp_pi * scale
    sp = (sk.ps_sigma if flip else sk.sp_sigma) * scale
    ps = (sk.sp_sigma if flip else sk.ps_sigma) * scale
    s_p = (sk.p_sstar_sigma if flip else sk.sstar_p_sigma) * scale
    p_s = (sk.sstar_p_sigma if flip else sk.p_sstar_sigma) * scale

    dirs = (l, m, n)
    if "s" in idx:
        B[idx["s"], idx["s"]] = ss
        for a, comp in zip(("px", "py", "pz"), dirs):
            if a in idx:
                B[idx["s"], idx[a]] = comp * sp
                B[idx[a], idx["s"]] = -comp * ps
    if "s*" in idx:
        for a, comp in zip(("px", "py", "pz"), dirs):
            if a in idx:
                B[idx["s*"], idx[a]] = comp * s_p
                B[idx[a], idx["s*"]] = -comp * p_s
    p_names = [o for o in ("px", "py", "pz") if o in idx]
    for a, ca in zip(p_names, [dirs[("px", "py", "pz").index(o)] for o in p_names]):
        for b, cb in zip(p_names, [dirs[("px", "py", "pz").index(o)] for o in p_names]):
            if a == b:
                B[idx[a], idx[b]] = ca * ca * ppS + (1.0 - ca * ca) * ppP
            else:
                B[idx[a], idx[b]] = ca * cb * (ppS - ppP)
    if len(orbs) == 1 and orbs[0] == "pz":
        B[0, 0] = ppP * 1.0
    return B



def robust_eigh(matrix: np.ndarray, eigenvectors: bool = True):
    """Symmetric/Hermitian eigendecomposition with driver fallbacks.

    LAPACK's divide-and-conquer driver occasionally reports that eigenvalues
    did not converge for large matrices. The relatively-robust
    representations driver succeeds in those cases, so it is tried next, and
    the classic QR driver last. A failure of all three is reported as such
    rather than returning a partial spectrum.
    """
    try:
        if eigenvectors:
            return np.linalg.eigh(matrix)
        return np.linalg.eigvalsh(matrix), None
    except np.linalg.LinAlgError:
        pass
    try:
        from scipy.linalg import eigh as scipy_eigh
    except ImportError as exc:
        raise np.linalg.LinAlgError(
            "Eigendecomposition failed and SciPy is unavailable for a fallback."
        ) from exc
    for driver in ("evr", "ev"):
        try:
            if eigenvectors:
                values, vectors = scipy_eigh(matrix, driver=driver)
                return values, vectors
            return scipy_eigh(matrix, eigvals_only=True, driver=driver), None
        except Exception:
            continue
    raise np.linalg.LinAlgError(
        f"Could not diagonalise a {matrix.shape[0]}x{matrix.shape[0]} Hamiltonian "
        "with any available LAPACK driver. Reduce the region size, or lower the "
        "electronic slab depth in the instrument settings."
    )


def eigh_in_window(matrix: np.ndarray, lo: float, hi: float):
    """Eigenpairs whose eigenvalues lie in ``[lo, hi]``.

    A scanning-probe image needs only the states inside the bias window, which
    is a small fraction of the spectrum. LAPACK's relatively-robust
    representations driver extracts exactly that subset, turning a full
    diagonalisation into a partial one. If the driver is unavailable the full
    spectrum is computed and sliced, which gives the same answer more slowly.
    """
    try:
        from scipy.linalg import eigh as scipy_eigh
    except ImportError:
        values, vectors = robust_eigh(matrix)
        sel = (values >= lo) & (values <= hi)
        return values[sel], vectors[:, sel]
    try:
        values, vectors = scipy_eigh(matrix, subset_by_value=(lo, hi), driver="evr")
        return values, vectors
    except Exception:
        values, vectors = robust_eigh(matrix)
        sel = (values >= lo) & (values <= hi)
        return values[sel], vectors[:, sel]


class TightBinding(Solver):
    """Orthogonal semi-empirical tight binding."""

    fidelity = Fidelity.TIER2_SEMI_EMPIRICAL
    capabilities = {
        Capability.BAND_STRUCTURE, Capability.TOTAL_DOS, Capability.LOCAL_DOS,
        Capability.STM_LDOS, Capability.ORBITALS, Capability.PARTIAL_CHARGES,
        Capability.CHARGE_DENSITY,
    }

    def __init__(self, model: str = "sp3s*-Si",
                 impurities: Sequence[str] = ()) -> None:
        if model not in MODELS:
            raise KeyError(
                f"Unknown tight-binding model {model!r}. Available: {sorted(MODELS)}."
            )
        self.model = MODELS[model]
        self.host = sorted(self.model.species)[0]
        self.impurities: Dict[str, Dict[str, float]] = {}
        for imp in impurities:
            sym = pt.symbol(imp)
            if sym in self.model.species:
                continue
            self.impurities[sym] = impurity_onsite_shift(self.host, sym)
        self.name = (f"tight-binding/{self.model.name}" if not self.impurities
                     else f"tight-binding/{self.model.name}+"
                          f"{'/'.join(sorted(self.impurities))}")
        self.description = (
            f"Orthogonal {self.model.name} tight binding. Not self-consistent; "
            "no charge transfer, no spin, no excited states."
            + ("" if not self.impurities else
               " Substitutional impurities are represented by an on-site energy shift "
               "from free-atom term values; hoppings are those of the host."))

    def supports(self, structure: Structure) -> SupportReport:
        present = {pt.symbol(int(z)) for z in structure.numbers}
        known = set(self.model.species) | set(self.impurities)
        missing = sorted(present - known)
        blocking, warnings = [], []
        if missing:
            available = sorted(set(HARRISON_TERM_VALUES) - known)
            blocking.append(
                f"{self.name} has no parameters for {', '.join(missing)}. "
                f"Parameterised species: {sorted(known)}. Elements that can be added "
                f"as substitutional impurities (on-site shift only): {available}."
            )
        if self.impurities:
            warnings.append(
                "Substitutional impurities are modelled by an on-site energy shift "
                "from free-atom term values, with the host's hoppings. The "
                "impurity's own bonding and any shallow-donor binding energy are "
                "not reproduced."
            )
        from ..physics.potentials import overlap_problem

        overlap = overlap_problem(structure)
        if overlap is not None:
            blocking.append(str(overlap))
        book = check_electron_bookkeeping(structure, allow_charged=False)
        blocking.extend(book.blocking)
        warnings.extend(book.warnings)
        if abs(structure.total_magnetic_moment()) > 1e-9:
            warnings.append(
                "A magnetic moment is set, but this model is spin-restricted; "
                "the moment is ignored in the Hamiltonian."
            )
        roles = set(str(r) for r in structure.roles)
        if "surface" in roles:
            warnings.append(
                "Surface dangling-bond states will appear in the gap. Their energies "
                "are not quantitative: the parameters were fitted to bulk bands."
            )
        return SupportReport(not blocking, blocking, warnings)

    def orbital_index(self, structure: Structure) -> Tuple[np.ndarray, List[str], List[int]]:
        """Return (atom_of_orbital, orbital_names, offsets)."""
        n_orb = len(self.model.orbitals)
        atom_of = np.repeat(np.arange(len(structure)), n_orb)
        names = [f"{pt.symbol(int(structure.numbers[a]))}{int(structure.ids[a])}:{o}"
                 for a in range(len(structure)) for o in self.model.orbitals]
        offsets = [a * n_orb for a in range(len(structure))]
        return atom_of, names, offsets

    def build_hamiltonian(self, structure: Structure, k_frac: Optional[Sequence[float]] = None
                          ) -> np.ndarray:
        """Assemble H(k).  ``k_frac`` is in reciprocal-lattice (fractional) units."""
        n_orb = len(self.model.orbitals)
        n = len(structure)
        dim = n * n_orb
        syms = [pt.symbol(int(z)) for z in structure.numbers]
        complex_h = k_frac is not None and structure.cell.is_periodic
        H = np.zeros((dim, dim), dtype=complex if complex_h else float)

        host = self.model.species[self.host]
        for a in range(n):
            sym = syms[a]
            if sym in self.model.species:
                sp = self.model.species[sym]
                shift = None
            else:
                sp = host
                shift = self.impurities[sym]
            for o, orb in enumerate(self.model.orbitals):
                value = sp.onsite[orb]
                if shift is not None:
                    value += shift.get(orb, 0.0)
                H[a * n_orb + o, a * n_orb + o] = value

        nl = neighbor_list(structure.positions, structure.cell, self.model.cutoff_A)
        if len(nl) == 0:
            return H
        if complex_h:
            kcart = np.asarray(k_frac, dtype=float) @ structure.cell.reciprocal
            phases = np.exp(1j * (nl.shifts.astype(float) @ structure.cell.matrix) @ kcart)
        else:
            phases = np.ones(len(nl))

        for p in range(len(nl)):
            i, j = int(nl.i[p]), int(nl.j[p])
            si = syms[i] if syms[i] in self.model.species else self.host
            sj = syms[j] if syms[j] in self.model.species else self.host
            block = _slater_koster_block(self.model, si, sj, nl.D[p], float(nl.d[p]))
            H[i * n_orb:(i + 1) * n_orb, j * n_orb:(j + 1) * n_orb] += block * phases[p]
        H = 0.5 * (H + H.conj().T)
        return H

    def _valence_of(self, z: int) -> int:
        from ..elements.configuration import valence_electrons
        sym = pt.symbol(z)
        if sym in self.model.species:
            return self.model.species[sym].valence_electrons
        return valence_electrons(z)

    def _n_electrons(self, structure: Structure) -> int:
        return int(sum(self._valence_of(int(z)) for z in structure.numbers))

    def fermi_level(self, eigenvalues: np.ndarray, n_electrons: int,
                    temperature_K: float = 300.0) -> Tuple[float, str]:
        """Fermi level from the eigenvalue spectrum (2 electrons per state)."""
        ev = np.sort(np.asarray(eigenvalues).ravel())
        n_states = n_electrons / 2.0
        idx = int(np.floor(n_states))
        if idx <= 0:
            return float(ev[0]), "below the lowest state"
        if idx >= len(ev):
            return float(ev[-1]), "all states occupied"
        homo = ev[idx - 1]
        lumo = ev[idx]
        if abs(n_states - idx) > 1e-9:
            return float(lumo), "partially filled state (metallic)"
        return float(0.5 * (homo + lumo)), "mid-gap between HOMO and LUMO"

    def run(self, structure: Structure, task: str = "eigenstates", **kwargs) -> SolverResult:
        if task in ("eigenstates", "electronic"):
            return self.eigenstates(structure, **kwargs)
        if task == "band_structure":
            return self.band_structure(structure, **kwargs)
        raise ValueError(
            f"Unknown task {task!r} for {self.name}. Supported: eigenstates, band_structure."
        )

    def _provenance(self, structure: Structure, **extra) -> Provenance:
        params = {"model": self.model.name, "cutoff_A": self.model.cutoff_A,
                  "orbitals": list(self.model.orbitals)}
        params.update(extra.pop("parameters", {}))
        return Provenance(
            model=self.name,
            fidelity=self.fidelity,
            origin=Origin.CALCULATED,
            parameters=params,
            approximations=[
                "Orthogonal two-centre nearest-neighbour tight binding.",
                f"Hopping distance dependence: {self.model.scaling} (rule of thumb).",
                "NOT self-consistent: no charge transfer, no band bending, no response "
                "to an applied field or net charge.",
                "No spin-orbit coupling and no spin polarisation.",
                "Single-particle eigenvalues; no excitonic or quasiparticle corrections.",
            ] + ([self.model.notes] if self.model.notes else [])
              + ([] if not self.impurities else [
                "Substitutional impurities (" + ", ".join(sorted(self.impurities))
                + ") carry an on-site shift from Harrison free-atom term values and "
                  "the host's hopping integrals. Shallow-donor binding energies are "
                  "NOT reproduced: they require the long-range Coulomb tail of the "
                  "ionised impurity and a central-cell correction."]),
            boundary_conditions=str(structure.cell.pbc),
            references=[self.model.reference],
            **extra,
        )

    def eigenstates(
        self,
        structure: Structure,
        temperature_K: float = 300.0,
        broadening_eV: float = 0.1,
        energy_window: Optional[Tuple[float, float]] = None,
        n_energy: int = 601,
        self_consistent: bool = False,
    ) -> SolverResult:
        """Diagonalise H at the supercell Gamma point and derive DOS and LDOS."""
        t0 = time.perf_counter()
        out = SolverResult(solver=self.name, structure=structure)
        report = self.supports(structure)
        if not report.ok:
            for key in ("eigenvalues", "total_dos", "local_dos"):
                out.results[key] = unsupported(
                    key, self.name, "; ".join(report.blocking),
                    suggested_models=["external:gpaw", "external:quantum-espresso"])
            out.log.extend(report.blocking)
            return out
        if self_consistent:
            out.results["self_consistency"] = unsupported(
                "self_consistent_solution", self.name,
                "This tight-binding model is non-self-consistent by construction: the "
                "on-site energies are fixed empirical parameters, so there is no charge "
                "density to iterate. A self-consistent treatment requires a "
                "self-consistent-charge method (DFTB) or a DFT solver.",
                suggested_models=["external:gpaw", "external:cp2k", "external:quantum-espresso"],
            )
            out.log.append("Self-consistency requested but not available in this model.")

        H = self.build_hamiltonian(structure)
        evals, evecs = robust_eigh(H)
        n_el = self._n_electrons(structure)
        e_f, ef_note = self.fermi_level(evals, n_el, temperature_K)

        lo, hi = energy_window or (float(evals.min()) - 2.0, float(evals.max()) + 2.0)
        grid = np.linspace(lo, hi, n_energy)
        sigma = max(broadening_eV, 1e-6)
        gauss = np.exp(-0.5 * ((grid[:, None] - evals[None, :]) / sigma) ** 2) / (
            sigma * np.sqrt(2.0 * np.pi))
        total_dos = 2.0 * gauss.sum(axis=1)

        n_orb = len(self.model.orbitals)
        weights = (np.abs(evecs) ** 2).reshape(len(structure), n_orb, -1).sum(axis=1)
        local_dos = 2.0 * (gauss @ weights.T)

        occ = np.zeros_like(evals)
        kt = BOLTZMANN_EV_K * max(temperature_K, 1e-6)
        occ = 1.0 / (1.0 + np.exp((evals - e_f) / kt))
        electrons_per_atom = 2.0 * (weights * occ[None, :]).sum(axis=1)
        z_val = np.array([self._valence_of(int(z)) for z in structure.numbers],
                         dtype=float)
        partial_charges = z_val - electrons_per_atom

        prov = self._provenance(
            structure,
            tolerances={"gaussian_broadening_eV": broadening_eV,
                        "energy_points": n_energy},
            parameters={"model": self.model.name, "cutoff_A": self.model.cutoff_A,
                        "orbitals": list(self.model.orbitals),
                        "k_sampling": "Gamma point of the supercell only",
                        "temperature_K": temperature_K},
        )
        prov.approximations.append(
            "Gamma-point sampling of the supercell: dispersion within the supercell "
            "Brillouin zone is not resolved. Use a larger cell or band_structure()."
        )

        conv = Convergence(True, 1, 0.0, "direct diagonalisation", None,
                           "Direct dense diagonalisation; no iterative convergence involved.")
        out.convergence = conv
        out.results["eigenvalues"] = Result("eigenvalues", evals, "eV", prov, convergence=conv)
        out.results["eigenvectors"] = Result(
            "eigenvectors", evecs, "dimensionless", prov, convergence=conv,
            extra={"layout": "rows = orbitals (atom-major), columns = states"})
        out.results["fermi_level"] = Result(
            "fermi_level", e_f, "eV", prov, convergence=conv,
            extra={"determination": ef_note, "valence_electrons": n_el})
        out.results["total_dos"] = Result(
            "total_dos", {"energy_eV": grid, "dos": total_dos}, "states/eV", prov,
            convergence=conv)
        out.results["local_dos"] = Result(
            "local_dos",
            {"energy_eV": grid, "ldos": local_dos, "atom_ids": structure.ids.tolist()},
            "states/eV/atom", prov, convergence=conv)
        out.results["partial_charges"] = Result(
            "partial_charges", partial_charges, "e", prov, convergence=conv,
            extra={"definition": "Mulliken-like population on an orthogonal basis "
                                 "(equal to the squared amplitude sum)."})

        gap = self._gap(evals, n_el)
        if gap is not None:
            out.results["hl_gap"] = Result(
                "homo_lumo_gap", gap["gap"], "eV", prov, convergence=conv,
                extra={"homo_eV": gap["homo"], "lumo_eV": gap["lumo"],
                       "note": "Supercell HOMO-LUMO gap at Gamma, not the bulk band gap. "
                               "For the band gap use band_structure() on a bulk cell."})
        structure.partial_charges[:] = partial_charges
        out.log.extend(report.warnings)
        out.wall_time_s = time.perf_counter() - t0
        return out

    @staticmethod
    def _gap(evals: np.ndarray, n_electrons: int) -> Optional[dict]:
        idx = n_electrons // 2
        if n_electrons % 2 or idx <= 0 or idx >= len(evals):
            return None
        ev = np.sort(evals)
        return {"homo": float(ev[idx - 1]), "lumo": float(ev[idx]),
                "gap": float(ev[idx] - ev[idx - 1])}

    def band_structure(
        self,
        structure: Structure,
        path: Optional[Sequence[Sequence[float]]] = None,
        labels: Optional[Sequence[str]] = None,
        n_per_segment: int = 40,
    ) -> SolverResult:
        """Eigenvalues along a k-path in fractional reciprocal coordinates."""
        t0 = time.perf_counter()
        out = SolverResult(solver=self.name, structure=structure)
        report = self.supports(structure)
        if not report.ok:
            out.results["band_structure"] = unsupported(
                "band_structure", self.name, "; ".join(report.blocking),
                suggested_models=["external:quantum-espresso", "external:gpaw"])
            return out
        if not structure.cell.is_periodic:
            out.results["band_structure"] = unsupported(
                "band_structure", self.name,
                "Band structure requires a periodic cell; this structure is a finite cluster.",
                suggested_models=["eigenstates (discrete molecular levels)"])
            return out

        if path is None:
            path = [(0.5, 0.5, 0.5), (0.0, 0.0, 0.0), (0.5, 0.0, 0.5)]
            labels = ["L", "G", "X"]
        pts = [np.asarray(p, dtype=float) for p in path]
        kpts, ticks, dist = [], [0.0], 0.0
        rec = structure.cell.reciprocal
        for a, b in zip(pts[:-1], pts[1:]):
            seg = [a + (b - a) * t for t in np.linspace(0, 1, n_per_segment, endpoint=False)]
            kpts.extend(seg)
            dist += float(np.linalg.norm((b - a) @ rec))
            ticks.append(dist)
        kpts.append(pts[-1])

        bands = []
        for k in kpts:
            H = self.build_hamiltonian(structure, k_frac=k)
            bands.append(robust_eigh(H, eigenvectors=False)[0])
        bands = np.array(bands)

        n_el = self._n_electrons(structure)
        n_occ = n_el // 2
        vbm = float(bands[:, :n_occ].max()) if n_occ else float("nan")
        cbm = float(bands[:, n_occ:].min()) if n_occ < bands.shape[1] else float("nan")
        gap = cbm - vbm
        k_vbm = int(np.argmax(bands[:, n_occ - 1])) if n_occ else 0
        k_cbm = int(np.argmin(bands[:, n_occ])) if n_occ < bands.shape[1] else 0
        direct = bool(k_vbm == k_cbm)

        coords = [0.0]
        for a, b in zip(kpts[:-1], kpts[1:]):
            coords.append(coords[-1] + float(np.linalg.norm((np.asarray(b) - np.asarray(a)) @ rec)))

        prov = self._provenance(structure, parameters={
            "model": self.model.name, "n_kpoints": len(kpts),
            "path": [list(map(float, p)) for p in pts], "labels": list(labels or []),
        })
        conv = Convergence(True, len(kpts), 0.0, "direct diagonalisation per k-point")
        out.convergence = conv
        out.results["band_structure"] = Result(
            "band_structure",
            {"k_coord": coords, "bands_eV": bands, "ticks": ticks,
             "labels": list(labels or []), "kpoints_frac": [list(map(float, k)) for k in kpts]},
            "eV", prov, convergence=conv)
        out.results["band_gap"] = Result(
            "band_gap", float(gap), "eV", prov, convergence=conv,
            extra={"vbm_eV": vbm, "cbm_eV": cbm, "direct": direct,
                   "note": ("Single-particle gap of the empirical model along the sampled "
                            "path only. Semi-empirical tight binding is fitted to reproduce "
                            "experimental gaps; it is not a predictive quasiparticle "
                            "calculation.")})
        out.results["fermi_level"] = Result(
            "fermi_level", float(0.5 * (vbm + cbm)), "eV", prov, convergence=conv,
            extra={"determination": "midgap"})
        out.log.extend(report.warnings)
        out.wall_time_s = time.perf_counter() - t0
        return out
