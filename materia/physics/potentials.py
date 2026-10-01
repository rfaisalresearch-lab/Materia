"""Tier-1 interatomic potentials with analytic forces.

Every potential exposes :meth:`Potential.energy_and_forces` returning total
potential energy in eV and forces in eV/A, plus :meth:`describe` giving the
approximations, parameters and references that go into the provenance record.

Implemented
-----------
``StillingerWeber``
    The two- plus three-body potential of Stillinger and Weber for
    tetrahedral semiconductors.  Parameterised for Si (original 1985 fit) and
    Ge (Ding and Andersen 1986).  By construction the ideal diamond structure
    at the fitted lattice constant has exactly ``2*epsilon`` cohesive energy
    per atom, which is used as an analytic regression test.
``LennardJones``
    12-6 pair potential with Lorentz-Berthelot mixing.  Parameters for metals
    are *derived* from the tabulated cohesive energy and nearest-neighbour
    distance via the fcc lattice sums, and are explicitly flagged as an
    interactive approximation: a pair potential cannot represent metallic
    many-body bonding.  Quantitative metal energetics need an EAM potential,
    available through the optional LAMMPS adapter.
``Harmonic``
    Einstein-solid springs to fixed reference sites.  Exists for testing
    minimisers and thermostats against analytic results.

Not implemented
---------------
REBO, MEAM and machine-learned potentials.  Tersoff bond-order potentials
live in :mod:`materia.physics.tersoff`.
Requests for them return an explicit unsupported result rather than a
substitute.  EAM lives in :mod:`materia.physics.eam`, long-range point-charge
electrostatics in :mod:`materia.physics.electrostatics` and rigid-ion
potentials (point charges plus a Buckingham term) in
:mod:`materia.physics.rigid_ion`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core_model.structure import Structure
from ..elements import periodic_table as pt
from ..provenance import Fidelity
from .neighbors import CoincidentAtoms, NeighborList, find_coincidence, neighbor_list


class PotentialError(Exception):
    pass


class UnsupportedSystem(PotentialError):
    """The potential has no parameters for this combination of elements."""


class OverlappingAtoms(UnsupportedSystem):
    """Two distinct atoms coincide, so the configuration has no defined energy."""

    def __init__(self, coincidence: CoincidentAtoms) -> None:
        super().__init__(str(coincidence))
        self.coincidence = coincidence


def refuse_overlap(structure: Structure, coincidence: CoincidentAtoms) -> OverlappingAtoms:
    """The refusal for a coincidence, naming the atoms by id and element."""
    label = lambda k: (f"#{int(structure.ids[k])} ({pt.symbol(int(structure.numbers[k]))})")
    return OverlappingAtoms(coincidence.relabel((label(coincidence.i), label(coincidence.j))))


def overlap_problem(structure: Structure) -> Optional[OverlappingAtoms]:
    """The first coincidence of distinct atoms in a structure, or ``None``."""
    found = find_coincidence(structure.positions, structure.cell)
    return None if found is None else refuse_overlap(structure, found)


class Potential:
    """Base class for Tier-1 classical potentials."""

    name = "abstract"
    fidelity = Fidelity.TIER1_CLASSICAL
    cutoff_A = 0.0

    def composition(self, structure: Structure) -> Tuple[bool, str]:
        """Whether the potential has parameters for every element present."""
        return True, ""

    def supports(self, structure: Structure) -> Tuple[bool, str]:
        """Whether the potential can evaluate this structure, and why not.

        Covers the elements and coincident atoms. Evaluation itself repeats the
        coincidence check on every call through the neighbour search, so
        :meth:`check` only needs the composition.
        """
        ok, why = self.composition(structure)
        if not ok:
            return ok, why
        overlap = overlap_problem(structure)
        if overlap is not None:
            return False, str(overlap)
        return True, ""

    def check(self, structure: Structure) -> None:
        ok, why = self.composition(structure)
        if not ok:
            raise UnsupportedSystem(why)

    def neighbours(self, structure: Structure, cutoff: float) -> NeighborList:
        """The neighbour list, refusing coincident atoms by id and element."""
        try:
            return neighbor_list(structure.positions, structure.cell, cutoff)
        except CoincidentAtoms as exc:
            raise refuse_overlap(structure, exc) from None

    def energy_and_forces(self, structure: Structure) -> Tuple[float, np.ndarray]:
        raise NotImplementedError

    def energy(self, structure: Structure) -> float:
        return self.energy_and_forces(structure)[0]

    def model_label(self) -> str:
        """The name results carry; external engines override it to name themselves."""
        return f"classical/{self.name}"

    def diagnostics(self, structure: Structure) -> dict:
        """Checks of a finished evaluation that belong in its record; empty by default."""
        return {}

    def describe(self) -> dict:
        return {"model": self.name, "fidelity": self.fidelity.value,
                "cutoff_A": self.cutoff_A, "approximations": [], "references": []}


@dataclass(frozen=True)
class SWParams:
    epsilon: float
    sigma: float
    a: float
    lam: float
    gamma: float
    A: float
    B: float
    p: int
    q: int
    cos_theta0: float = -1.0 / 3.0
    source: str = ""


SW_PARAMETERS: Dict[str, SWParams] = {
    "Si": SWParams(
        epsilon=2.1683, sigma=2.0951, a=1.80, lam=21.0, gamma=1.20,
        A=7.049556277, B=0.6022245584, p=4, q=0,
        source="F. H. Stillinger and T. A. Weber, Phys. Rev. B 31 (1985) 5262",
    ),
    "Ge": SWParams(
        epsilon=1.93, sigma=2.181, a=1.80, lam=31.0, gamma=1.20,
        A=7.049556277, B=0.6022245584, p=4, q=0,
        source="K. Ding and H. C. Andersen, Phys. Rev. B 34 (1986) 6987",
    ),
}


SW_VARIANTS: Dict[Tuple[str, str], SWParams] = {
    ("Si", "vink2001"): SWParams(
        epsilon=1.64833, sigma=2.0951, a=1.80, lam=31.5, gamma=1.20,
        A=7.049556277, B=0.6022245584, p=4, q=0,
        source="R. L. C. Vink, G. T. Barkema, W. F. van der Weg and N. Mousseau, "
               "J. Non-Cryst. Solids 282 (2001) 248 (refitted to amorphous silicon)",
    ),
}


class StillingerWeber(Potential):
    """Stillinger-Weber potential for a tetrahedral host, optionally with
    substitutional impurities.

    Impurity handling
    -----------------
    The published parameterisations cover one species.  With
    ``impurities=("P", "B", ...)`` an impurity atom is treated with the host's
    functional form and the host's ``epsilon``, but with its pair ``sigma``
    rescaled by the ratio of covalent radii:

        sigma_ij = sigma_host * (r_cov(i) + r_cov(j)) / (2 r_cov(host))

    This is a **geometric approximation**.  It reproduces the sign and rough
    size of the local bond-length change around a substitutional impurity and
    nothing else: it carries no information about the impurity's valence, its
    electronic level or its charge state.  Use it to obtain a relaxed geometry
    quickly, then compute the electronic consequences with a Tier-2 or Tier-3
    electronic-structure model.  Every result produced this way records the
    approximation explicitly.
    """

    fidelity = Fidelity.TIER1_CLASSICAL

    def __init__(self, element: str = "Si", impurities: Sequence[str] = (),
                 variant: Optional[str] = None) -> None:
        sym = pt.symbol(element)
        if sym not in SW_PARAMETERS:
            raise UnsupportedSystem(
                f"No Stillinger-Weber parameterisation for {sym}. "
                f"Available: {sorted(SW_PARAMETERS)}."
            )
        if variant is not None and (sym, variant) not in SW_VARIANTS:
            raise UnsupportedSystem(
                f"No Stillinger-Weber variant {variant!r} for {sym}. Available: "
                f"{sorted(v for e, v in SW_VARIANTS if e == sym)}.")
        self.element = sym
        self.variant = variant
        self.p = SW_VARIANTS[(sym, variant)] if variant else SW_PARAMETERS[sym]
        self.impurities = tuple(pt.symbol(i) for i in impurities)
        tag = f"{sym},{variant}" if variant else sym
        self.name = (f"stillinger-weber({tag})" if not self.impurities
                     else f"stillinger-weber({tag}+{'/'.join(self.impurities)})")
        host_r = pt.covalent_radius(sym) or 1.11
        self._sigma_scale = {sym: 1.0}
        for imp in self.impurities:
            r = pt.covalent_radius(imp)
            if r is None:
                raise UnsupportedSystem(
                    f"No covalent radius tabulated for {imp}; cannot scale the "
                    "Stillinger-Weber sigma for it."
                )
            self._sigma_scale[imp] = r / host_r
        self.cutoff_A = self.p.a * self.p.sigma * max(self._sigma_scale.values())

    def composition(self, structure: Structure) -> Tuple[bool, str]:
        known = {self.element} | set(self.impurities)
        others = sorted({pt.symbol(int(v)) for v in structure.numbers} - known)
        if others:
            return False, (
                f"{self.name} covers {sorted(known)}; this structure also contains "
                f"{', '.join(others)}. Declare them as impurities "
                f"(StillingerWeber('{self.element}', impurities={others!r})), accepting "
                "the geometric approximation that implies, or use an external solver."
            )
        return True, ""

    def _phi2(self, r: np.ndarray, sigma: Optional[np.ndarray] = None
              ) -> Tuple[np.ndarray, np.ndarray]:
        p = self.p
        sig = np.full_like(r, p.sigma) if sigma is None else np.asarray(sigma, dtype=float)
        rc = p.a * sig
        inside = r < rc - 1e-12
        e = np.zeros_like(r)
        de = np.zeros_like(r)
        rr = r[inside]
        if rr.size:
            p_sigma = sig[inside]
            sr = p_sigma / rr
            expo = np.exp(1.0 / (rr / p_sigma - p.a))
            radial = p.B * sr**p.p - sr**p.q
            e[inside] = p.A * p.epsilon * radial * expo
            dradial = (-p.p * p.B * sr**p.p + p.q * sr**p.q) / rr
            dexp = -expo / (p_sigma * (rr / p_sigma - p.a) ** 2)
            de[inside] = p.A * p.epsilon * (dradial * expo + radial * dexp)
        return e, de

    def energy_and_forces(self, structure: Structure) -> Tuple[float, np.ndarray]:
        self.check(structure)
        pos = structure.positions
        n = len(structure)
        forces = np.zeros((n, 3))
        if n == 0:
            return 0.0, forces

        nl = self.neighbours(structure, self.cutoff_A)
        if len(nl) == 0:
            return 0.0, forces

        scale = np.array([self._sigma_scale[pt.symbol(int(z))] for z in structure.numbers])
        pair_sigma = self.p.sigma * 0.5 * (scale[nl.i] + scale[nl.j])

        e2, de2 = self._phi2(nl.d, pair_sigma)
        energy = 0.5 * float(e2.sum())
        unit = nl.D / nl.d[:, None]
        np.add.at(forces, nl.i, de2[:, None] * unit)

        e3 = self._three_body(nl, forces, pair_sigma)
        energy += e3
        return float(energy), forces

    def _three_body(self, nl: NeighborList, forces: np.ndarray,
                    pair_sigma: np.ndarray) -> float:
        p = self.p
        order = np.argsort(nl.i, kind="stable")
        i_s, d_s, D_s = nl.i[order], nl.d[order], nl.D[order]
        sig_s = pair_sigma[order]
        counts = np.bincount(i_s, minlength=nl.n_atoms)
        starts = np.concatenate([[0], np.cumsum(counts)])

        total = 0.0
        for c in np.unique(counts):
            if c < 2:
                continue
            atoms = np.nonzero(counts == c)[0]
            if atoms.size == 0:
                continue
            idx = starts[atoms][:, None] + np.arange(c)[None, :]
            rj = d_s[idx]
            Dj = D_s[idx]
            sj = sig_s[idx]
            a_pairs, b_pairs = np.triu_indices(c, k=1)
            rij = rj[:, a_pairs]
            rik = rj[:, b_pairs]
            sij = sj[:, a_pairs]
            sik = sj[:, b_pairs]
            vij = Dj[:, a_pairs, :]
            vik = Dj[:, b_pairs, :]

            active = (rij < p.a * sij - 1e-12) & (rik < p.a * sik - 1e-12)
            if not active.any():
                continue

            cos = np.einsum("ijk,ijk->ij", vij, vik) / (rij * rik)
            dcos = cos - p.cos_theta0
            gij = p.gamma * sij / (rij - p.a * sij)
            gik = p.gamma * sik / (rik - p.a * sik)
            expo = np.exp(np.where(active, gij + gik, -700.0))
            h = p.lam * p.epsilon * dcos**2 * expo
            h = np.where(active, h, 0.0)
            total += float(h.sum())

            pref = np.where(active, p.lam * p.epsilon * expo, 0.0)
            dh_dcos = pref * 2.0 * dcos
            dgij_dr = -p.gamma * sij / (rij - p.a * sij) ** 2
            dgik_dr = -p.gamma * sik / (rik - p.a * sik) ** 2
            dh_drij = np.where(active, pref * dcos**2 * dgij_dr, 0.0)
            dh_drik = np.where(active, pref * dcos**2 * dgik_dr, 0.0)

            uij = vij / rij[..., None]
            uik = vik / rik[..., None]
            dcos_dvij = uik / rij[..., None] - cos[..., None] * uij / rij[..., None]
            dcos_dvik = uij / rik[..., None] - cos[..., None] * uik / rik[..., None]

            gvij = dh_dcos[..., None] * dcos_dvij + dh_drij[..., None] * uij
            gvik = dh_dcos[..., None] * dcos_dvik + dh_drik[..., None] * uik

            j_glob = nl.j[order][idx][:, a_pairs]
            k_glob = nl.j[order][idx][:, b_pairs]
            np.add.at(forces, j_glob.ravel(), -gvij.reshape(-1, 3))
            np.add.at(forces, k_glob.ravel(), -gvik.reshape(-1, 3))
            np.add.at(forces, np.repeat(atoms, len(a_pairs)),
                      (gvij + gvik).reshape(-1, 3))
        return total

    def describe(self) -> dict:
        p = self.p
        return {
            "model": self.name,
            "fidelity": self.fidelity.value,
            "cutoff_A": self.cutoff_A,
            "parameters": {
                "epsilon_eV": p.epsilon, "sigma_A": p.sigma, "a": p.a,
                "lambda": p.lam, "gamma": p.gamma, "A": p.A, "B": p.B,
                "p": p.p, "q": p.q, "cos_theta0": p.cos_theta0,
            },
            "approximations": [
                "Empirical two- plus three-body form; no electrons are represented.",
                "Fitted to the diamond structure and the melting behaviour of the "
                "bulk element; transferability to surfaces, defects and other "
                "coordinations is approximate.",
                ("Single species only; no charge, no polarisation, no dispersion."
                 if not self.impurities else
                 "Substitutional impurities " + ", ".join(self.impurities) +
                 " are treated with the host functional form and host epsilon, with "
                 "sigma rescaled by the covalent-radius ratio "
                 + ", ".join(f"{k}: {v:.4f}" for k, v in self._sigma_scale.items())
                 + ". This is a geometric approximation with no electronic content: "
                   "it gives the sign and rough magnitude of the local bond-length "
                   "change and nothing more."),
                "Abrupt cutoff at a*sigma; the potential and its first derivative "
                "go to zero there by construction.",
            ] + ([f"Variant {self.variant}: lambda and epsilon refitted to amorphous "
                  "silicon; crystalline properties differ from the original fit."]
                 if self.variant else []),
            "references": [p.source],
        }


FCC_A12 = 12.13188
FCC_A6 = 14.45392
FCC_R0_OVER_SIGMA = (2.0 * FCC_A12 / FCC_A6) ** (1.0 / 6.0)
FCC_ECOH_OVER_EPS = 2.0 * FCC_A6**2 / (4.0 * FCC_A12)


def fcc_shell_distances(r0: float, cutoff_A: float) -> np.ndarray:
    """Distances from one fcc site to every neighbour closer than ``cutoff_A``."""
    half = r0 / np.sqrt(2.0)
    m = int(np.ceil(cutoff_A / half)) + 1
    grid = np.arange(-m, m + 1)
    h, k, l = np.meshgrid(grid, grid, grid, indexing="ij")
    keep = ((h + k + l) % 2 == 0) & ~((h == 0) & (k == 0) & (l == 0))
    d = half * np.sqrt(h[keep] ** 2 + k[keep] ** 2 + l[keep] ** 2)
    return np.sort(d[d < cutoff_A])


def fcc_lj_parameters(r0: float, cohesive_energy_eV: float, cutoff_A: float,
                      shift: bool = True) -> Tuple[float, float]:
    """epsilon and sigma that put a cut 12-6 fcc crystal at ``r0`` with ``E_coh``.

    With the neighbour set inside the cutoff fixed, dE/dr0 = 0 gives
    sigma^6 = sum d^-6 / (2 sum d^-12) in closed form, and epsilon then
    follows from the energy per atom, including the shift at the cutoff.
    """
    d = fcc_shell_distances(r0, cutoff_A)
    if d.size == 0:
        raise UnsupportedSystem(f"A {cutoff_A:g} A cutoff contains no fcc neighbour at "
                                f"{r0:g} A.")
    sigma = float((np.sum(d ** -6.0) / (2.0 * np.sum(d ** -12.0))) ** (1.0 / 6.0))
    per_bond = (sigma / d) ** 12 - (sigma / d) ** 6
    if shift:
        per_bond = per_bond - ((sigma / cutoff_A) ** 12 - (sigma / cutoff_A) ** 6)
    energy_over_eps = 2.0 * float(per_bond.sum())
    return -cohesive_energy_eV / energy_over_eps, sigma


class LennardJones(Potential):
    """12-6 Lennard-Jones with per-element parameters and LB mixing."""

    name = "lennard-jones"
    fidelity = Fidelity.TIER1_CLASSICAL

    def __init__(self, params: Dict[str, Tuple[float, float]], cutoff_A: float = 10.0,
                 shift: bool = True, derivation: str = "") -> None:
        self.params = {pt.symbol(k): (float(e), float(s)) for k, (e, s) in params.items()}
        self.cutoff_A = float(cutoff_A)
        self.shift = bool(shift)
        self.derivation = derivation

    @staticmethod
    def from_material(material, cutoff_A: float = 10.0) -> "LennardJones":
        """Derive epsilon and sigma from cohesive energy and nn distance.

        Uses the fcc lattice sums over exactly the neighbours inside the
        cutoff, with the energy shift applied, so the potential as evaluated
        puts an fcc crystal at the tabulated nearest-neighbour distance with
        the tabulated cohesive energy.  That is exact only for a close-packed
        crystal of a pair-bonded species; the result is labelled as an
        approximation in the provenance record.
        """
        props = material.properties
        nn = props.get("nearest_neighbour_distance")
        if material.prototype == "fcc":
            r0 = material.lattice.parameters()[0] / np.sqrt(2.0)
        elif nn is not None:
            r0 = float(nn.value)
        else:
            a = material.lattice.parameters()[0]
            proto = material.prototype
            if proto == "fcc":
                r0 = a / np.sqrt(2.0)
            elif proto == "bcc":
                r0 = a * np.sqrt(3.0) / 2.0
            elif proto in ("diamond-cubic", "zinc-blende"):
                r0 = a * np.sqrt(3.0) / 4.0
            else:
                raise UnsupportedSystem(
                    f"Cannot derive Lennard-Jones parameters for prototype {proto!r}: "
                    "no nearest-neighbour distance is tabulated and the prototype has "
                    "no closed-form relation. Supply epsilon and sigma explicitly."
                )
        coh = props.get("cohesive_energy")
        if coh is None:
            raise UnsupportedSystem(
                f"{material.id} has no tabulated cohesive energy, so Lennard-Jones "
                "parameters cannot be derived. Supply them explicitly."
            )
        eps, sigma = fcc_lj_parameters(r0, float(coh.value), cutoff_A, shift=True)
        elements = material.elements()
        return LennardJones(
            {el: (eps, sigma) for el in elements},
            cutoff_A=cutoff_A,
            derivation=(
                f"epsilon = {eps:.5f} eV and sigma = {sigma:.5f} A, chosen so that an fcc "
                f"crystal with nearest-neighbour distance {r0:.4f} A is at its minimum with "
                f"cohesive energy {float(coh.value):g} eV/atom, summing exactly the shells "
                f"inside the {cutoff_A:g} A shifted cutoff this potential uses, for "
                f"{material.id}."
            ),
        )

    def composition(self, structure: Structure) -> Tuple[bool, str]:
        present = {pt.symbol(int(z)) for z in structure.numbers}
        missing = sorted(present - set(self.params))
        if missing:
            return False, (
                f"No Lennard-Jones parameters for {', '.join(missing)}. "
                f"Parameterised elements: {sorted(self.params)}."
            )
        return True, ""

    def _pair_arrays(self, structure: Structure, nl: NeighborList):
        syms = np.array([pt.symbol(int(z)) for z in structure.numbers])
        eps_i = np.array([self.params[s][0] for s in syms])
        sig_i = np.array([self.params[s][1] for s in syms])
        eps = np.sqrt(eps_i[nl.i] * eps_i[nl.j])
        sig = 0.5 * (sig_i[nl.i] + sig_i[nl.j])
        return eps, sig

    def energy_and_forces(self, structure: Structure) -> Tuple[float, np.ndarray]:
        self.check(structure)
        n = len(structure)
        forces = np.zeros((n, 3))
        if n < 2:
            return 0.0, forces
        nl = self.neighbours(structure, self.cutoff_A)
        if len(nl) == 0:
            return 0.0, forces
        eps, sig = self._pair_arrays(structure, nl)
        sr6 = (sig / nl.d) ** 6
        e = 4.0 * eps * (sr6**2 - sr6)
        if self.shift:
            src6 = (sig / self.cutoff_A) ** 6
            e = e - 4.0 * eps * (src6**2 - src6)
        de = 4.0 * eps * (-12.0 * sr6**2 + 6.0 * sr6) / nl.d
        unit = nl.D / nl.d[:, None]
        np.add.at(forces, nl.i, de[:, None] * unit)
        return 0.5 * float(e.sum()), forces

    def describe(self) -> dict:
        return {
            "model": self.name,
            "fidelity": self.fidelity.value,
            "cutoff_A": self.cutoff_A,
            "parameters": {k: {"epsilon_eV": v[0], "sigma_A": v[1]} for k, v in self.params.items()},
            "approximations": [
                "Pairwise additive 12-6 potential: no bond directionality and no "
                "many-body metallic screening.",
                "Lorentz-Berthelot mixing for unlike pairs.",
                ("Energy shifted to zero at the cutoff so that the energy is continuous; "
                 "the force is not shifted and has a small discontinuity at the cutoff."
                 if self.shift else "Unshifted; energy is discontinuous at the cutoff."),
                self.derivation or "Parameters supplied explicitly by the caller.",
            ],
            "references": [
                "J. E. Lennard-Jones, Proc. R. Soc. Lond. A 106 (1924) 463",
                "C. Kittel, Introduction to Solid State Physics, 8th ed., Ch. 3 (lattice sums)",
            ],
        }


class Harmonic(Potential):
    """Einstein springs to reference positions.  For minimiser/thermostat tests."""

    name = "harmonic-einstein"
    fidelity = Fidelity.NON_PHYSICAL

    def __init__(self, reference: np.ndarray, k_eV_A2: float = 1.0) -> None:
        self.reference = np.array(reference, dtype=float)
        self.k = float(k_eV_A2)
        self.cutoff_A = 0.0

    def energy_and_forces(self, structure: Structure) -> Tuple[float, np.ndarray]:
        d = structure.positions - self.reference
        return 0.5 * self.k * float((d * d).sum()), -self.k * d

    def describe(self) -> dict:
        return {
            "model": self.name, "fidelity": self.fidelity.value, "cutoff_A": 0.0,
            "parameters": {"k_eV_per_A2": self.k},
            "approximations": ["Not a physical model: springs to fixed reference sites."],
            "references": [],
        }
