"""Ground-state electron configurations, shell occupancies and quantum numbers.

Model note
----------
Configurations are generated from the Madelung (n+l, n) ordering rule and then
corrected with an explicit table of experimentally established exceptions
(Cr, Cu, Nb, Mo, Ru, Rh, Pd, Ag, La, Ce, Gd, Pt, Au, Ac, Th, Pa, U, Np, Cm, Lr).
The Madelung rule is an empirical ordering, not a theorem; it fails for the
tabulated exceptions and its reliability for Z > 103 is not established.  The
application therefore labels configurations for Z > 103 as ``predicted`` with
low confidence.

Ion configurations are produced by removing electrons from the highest-n
shell first (and within that shell, from the highest-l subshell), which is the
standard chemical rule for cations, and by Aufbau filling for anions.  This is
a rule of thumb, not a variational calculation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional, Sequence, Tuple

ORBITAL_LETTERS = "spdfghi"
SUBSHELL_CAPACITY = {l: 2 * (2 * l + 1) for l in range(7)}

_EXCEPTIONS = {
    24: {"4s": 1, "3d": 5},
    29: {"4s": 1, "3d": 10},
    41: {"5s": 1, "4d": 4},
    42: {"5s": 1, "4d": 5},
    44: {"5s": 1, "4d": 7},
    45: {"5s": 1, "4d": 8},
    46: {"5s": 0, "4d": 10},
    47: {"5s": 1, "4d": 10},
    57: {"5d": 1, "4f": 0},
    58: {"4f": 1, "5d": 1},
    64: {"4f": 7, "5d": 1},
    78: {"6s": 1, "5d": 9},
    79: {"6s": 1, "5d": 10},
    89: {"6d": 1, "5f": 0},
    90: {"6d": 2, "5f": 0},
    91: {"5f": 2, "6d": 1},
    92: {"5f": 3, "6d": 1},
    93: {"5f": 4, "6d": 1},
    96: {"5f": 7, "6d": 1},
    103: {"7s": 2, "7p": 1, "6d": 0},
}

_NOBLE_CORES = ((118, "Og"), (86, "Rn"), (54, "Xe"), (36, "Kr"), (18, "Ar"), (10, "Ne"), (2, "He"))


def madelung_order() -> Iterator[Tuple[int, int]]:
    """Yield ``(n, l)`` subshells in increasing (n+l, n) order."""
    for s in range(1, 16):
        for l in range((s - 1) // 2, -1, -1):
            n = s - l
            if n > l:
                yield n, l


@dataclass(frozen=True)
class Subshell:
    n: int
    l: int
    electrons: int

    @property
    def label(self) -> str:
        return f"{self.n}{ORBITAL_LETTERS[self.l]}"

    @property
    def capacity(self) -> int:
        return SUBSHELL_CAPACITY[self.l]

    @property
    def orbital_count(self) -> int:
        return 2 * self.l + 1

    @property
    def m_l_values(self) -> Tuple[int, ...]:
        return tuple(range(-self.l, self.l + 1))

    def __str__(self) -> str:
        return f"{self.label}{self.electrons}"


def _apply_exceptions(shells: List[Subshell], z: int) -> List[Subshell]:
    overrides = _EXCEPTIONS.get(z)
    if not overrides:
        return shells
    index = {s.label: i for i, s in enumerate(shells)}
    for label, count in overrides.items():
        n = int(label[0])
        l = ORBITAL_LETTERS.index(label[1])
        if label in index:
            i = index[label]
            shells[i] = Subshell(n, l, count)
        elif count:
            shells.append(Subshell(n, l, count))
    return [s for s in shells if s.electrons > 0]


def configuration(z: int, charge: int = 0) -> List[Subshell]:
    """Ground-state subshell occupancy for an atom or monatomic ion.

    Parameters
    ----------
    z
        Atomic number (number of protons).
    charge
        Net charge; ``+1`` removes one electron, ``-1`` adds one.

    Raises
    ------
    ValueError
        If the requested species has a non-positive electron count.
    """
    if z < 1:
        raise ValueError("Atomic number must be >= 1")
    n_electrons = z - charge
    if n_electrons < 0:
        raise ValueError(f"Z={z} with charge {charge:+d} implies {n_electrons} electrons")
    if n_electrons == 0:
        return []

    shells: List[Subshell] = []
    remaining = n_electrons
    for n, l in madelung_order():
        if remaining <= 0:
            break
        take = min(remaining, SUBSHELL_CAPACITY[l])
        shells.append(Subshell(n, l, take))
        remaining -= take
    if remaining > 0:
        raise ValueError(f"Could not place {remaining} electrons for Z={z}")

    if charge == 0:
        shells = _apply_exceptions(shells, z)
    elif charge > 0:
        shells = _apply_exceptions(
            [s for s in configuration_raw(z)], z
        )
        to_remove = charge
        while to_remove > 0:
            occupied = [s for s in shells if s.electrons > 0]
            if not occupied:
                raise ValueError("No electrons left to remove")
            target = max(occupied, key=lambda s: (s.n, s.l))
            i = shells.index(target)
            take = min(to_remove, target.electrons)
            shells[i] = Subshell(target.n, target.l, target.electrons - take)
            to_remove -= take
        shells = [s for s in shells if s.electrons > 0]

    return sorted(shells, key=lambda s: (s.n, s.l))


def configuration_raw(z: int) -> List[Subshell]:
    """Strict Madelung filling with no exception corrections."""
    shells: List[Subshell] = []
    remaining = z
    for n, l in madelung_order():
        if remaining <= 0:
            break
        take = min(remaining, SUBSHELL_CAPACITY[l])
        shells.append(Subshell(n, l, take))
        remaining -= take
    return shells


def configuration_string(z: int, charge: int = 0, abbreviated: bool = False) -> str:
    shells = configuration(z, charge)
    if not shells:
        return "(no electrons)"
    if abbreviated:
        n_electrons = z - charge
        for core_z, core_sym in _NOBLE_CORES:
            if core_z < n_electrons:
                core = configuration(core_z)
                core_labels = {s.label: s.electrons for s in core}
                rest = []
                ok = True
                for s in shells:
                    c = core_labels.get(s.label, 0)
                    if s.electrons < c:
                        ok = False
                        break
                    if s.electrons > c:
                        rest.append(Subshell(s.n, s.l, s.electrons - c))
                if ok:
                    body = " ".join(str(s) for s in sorted(rest, key=lambda s: (s.n, s.l)))
                    return f"[{core_sym}] {body}".strip()
    return " ".join(str(s) for s in shells)


def shell_occupancy(z: int, charge: int = 0) -> List[int]:
    """Electrons per principal shell (K, L, M, ... ) for the shell diagram."""
    shells = configuration(z, charge)
    if not shells:
        return []
    max_n = max(s.n for s in shells)
    occ = [0] * max_n
    for s in shells:
        occ[s.n - 1] += s.electrons
    return occ


def valence_electrons(z: int, charge: int = 0) -> int:
    """Electrons in the outermost principal shell plus any partially filled d/f.

    This is the chemically conventional count, not a rigorously defined
    observable.  For main-group elements it is the outer-shell count; for
    transition metals the incomplete (n-1)d electrons are included.
    """
    shells = configuration(z, charge)
    if not shells:
        return 0
    max_n = max(s.n for s in shells)
    total = sum(s.electrons for s in shells if s.n == max_n)
    for s in shells:
        if s.n < max_n and s.l >= 2 and s.electrons < s.capacity:
            total += s.electrons
    return total


def unpaired_electrons(z: int, charge: int = 0) -> int:
    """Number of unpaired electrons from Hund's first rule applied per subshell."""
    total = 0
    for s in configuration(z, charge):
        n_orb = s.orbital_count
        e = s.electrons
        total += e if e <= n_orb else 2 * n_orb - e
    return total


def term_spin_multiplicity(z: int, charge: int = 0) -> int:
    """Spin multiplicity 2S+1 implied by Hund's first rule."""
    return unpaired_electrons(z, charge) + 1


@dataclass(frozen=True)
class OrbitalState:
    """A single spin-orbital (n, l, m_l, m_s) with occupancy."""

    n: int
    l: int
    m_l: int
    m_s: float
    occupied: bool

    @property
    def label(self) -> str:
        return f"{self.n}{ORBITAL_LETTERS[self.l]}(m_l={self.m_l:+d}, m_s={self.m_s:+.1f})"


def spin_orbitals(z: int, charge: int = 0) -> List[OrbitalState]:
    """Enumerate spin-orbitals with Hund's-rule occupancy.

    Within a subshell electrons first singly occupy each ``m_l`` with
    ``m_s = +1/2`` (maximum multiplicity), then pair with ``m_s = -1/2``.
    The assignment of electrons to specific ``m_l`` values is a convention:
    the true many-electron state is generally a superposition of
    determinants.  This enumeration exists for teaching Pauli exclusion and
    Hund's rules, and is labelled as such in the UI.
    """
    out: List[OrbitalState] = []
    for s in configuration(z, charge):
        ml_values = list(s.m_l_values)
        n_single = min(s.electrons, len(ml_values))
        n_pair = max(0, s.electrons - len(ml_values))
        for i, ml in enumerate(ml_values):
            out.append(OrbitalState(s.n, s.l, ml, +0.5, i < n_single))
        for i, ml in enumerate(ml_values):
            out.append(OrbitalState(s.n, s.l, ml, -0.5, i < n_pair))
    return out


def configuration_confidence(z: int) -> Tuple[str, float, str]:
    """Return ``(origin, confidence, note)`` for a generated configuration."""
    if z in _EXCEPTIONS:
        return ("reference", 0.99, "Experimentally established Madelung exception.")
    if z <= 103:
        return ("estimated", 0.95, "Madelung (n+l, n) filling rule; no exception known for this Z.")
    return (
        "estimated",
        0.4,
        "Madelung rule extrapolated beyond Z=103; relativistic effects are not modelled.",
    )
