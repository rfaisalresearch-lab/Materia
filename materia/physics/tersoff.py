"""Tersoff bond-order potentials, in the form LAMMPS ``pair_style tersoff`` uses.

``E = 1/2 sum_i sum_j f_C(r_ij) [A e^(-lambda1 r_ij) - b_ij B e^(-lambda2 r_ij)]``
``b_ij = (1 + (beta zeta_ij)^n)^(-1/2n)``
``zeta_ij = sum_k f_C(r_ik) g(theta_ijk) exp[(lambda3 (r_ij - r_ik))^m]``
``g(theta) = gamma (1 + c^2/d^2 - c^2 / (d^2 + (cos theta - cos theta0)^2))``
``f_C(r) = 1/2 - 1/2 sin(pi/2 (r - R)/D)`` for ``R - D < r < R + D``.

Two-body parameters (n, beta, lambda2, B, lambda1, A and the pair R, D) come
from the ``i j j`` entry and three-body parameters (m, gamma, lambda3, c, d,
cos theta0 and the R, D of ``f_C(r_ik)``) from the ``i j k`` entry, as in
LAMMPS.  Forces are analytic derivatives of this energy.

Parameter files use the LAMMPS format.  Materia ships none: LAMMPS's files are
GPL-licensed and are downloaded on request into ``~/.cache/materia/
lammps-potentials`` by :func:`fetch_lammps_file`; each records its literature
source, which Materia copies into every result.

References
----------
J. Tersoff, Phys. Rev. B 37 (1988) 6991; Phys. Rev. B 38 (1988) 9902;
Phys. Rev. B 39 (1989) 5566.
"""

from __future__ import annotations

import math
import os
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..core_model.structure import Structure
from ..elements import periodic_table as pt
from ..provenance import Fidelity
from .potentials import Potential, UnsupportedSystem

LAMMPS_URL = "https://raw.githubusercontent.com/lammps/lammps/stable/potentials/"
FIELDS = ("m", "gamma", "lambda3", "c", "d", "costheta0", "n", "beta", "lambda2", "B", "R", "D",
          "lambda1", "A")


@dataclass(frozen=True)
class TersoffEntry:
    m: float
    gamma: float
    lambda3: float
    c: float
    d: float
    costheta0: float
    n: float
    beta: float
    lambda2: float
    B: float
    R: float
    D: float
    lambda1: float
    A: float


def cache_dir() -> Path:
    return Path(os.environ.get("MATERIA_LAMMPS_POTENTIALS", "") or
                Path.home() / ".cache" / "materia" / "lammps-potentials")


def fetch_lammps_file(name: str) -> Path:
    """A LAMMPS potential file from the LAMMPS repository, cached locally."""
    if "/" in name or not re.fullmatch(r"[\w.\-]+", name):
        raise UnsupportedSystem(f"{name!r} is not a plain LAMMPS potential file name.")
    target = cache_dir() / name
    if not target.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            with urllib.request.urlopen(LAMMPS_URL + name, timeout=60) as response:
                data = response.read()
        except OSError as exc:
            raise UnsupportedSystem(f"Could not download {name}: {exc}") from None
        target.with_suffix(".part").write_bytes(data)
        target.with_suffix(".part").replace(target)
    return target


def parse(text: str) -> Tuple[Dict[Tuple[str, str, str], TersoffEntry], List[str]]:
    """Entries keyed by element triplet, and the file's comment lines."""
    comments = [line[1:].strip() for line in text.splitlines() if line.strip().startswith("#")]
    tokens = []
    for line in text.splitlines():
        line = line.split("#", 1)[0]
        tokens.extend(line.split())
    if len(tokens) % 17:
        raise UnsupportedSystem("The Tersoff file does not hold whole 17-field entries.")
    entries = {}
    for k in range(0, len(tokens), 17):
        e1, e2, e3 = tokens[k:k + 3]
        values = [float(v) for v in tokens[k + 3:k + 17]]
        entries[(e1, e2, e3)] = TersoffEntry(*values)
    return entries, comments


class Tersoff(Potential):
    """A Tersoff potential read from a LAMMPS-format parameter file."""

    fidelity = Fidelity.TIER1_CLASSICAL

    def __init__(self, filename: str = "Si.tersoff", text: Optional[str] = None,
                 source: str = "", elements: Optional[List[str]] = None) -> None:
        self.file = source or Path(filename).name
        self.name = f"tersoff({self.file})"
        self._filename = filename
        self._text = text
        self._declared = sorted(elements) if elements else None
        self._loaded = False
        self.cutoff_A = 0.0
        if text is not None:
            self._load()

    def _load(self) -> None:
        if self._loaded:
            return
        text = self._text
        if text is None:
            path = (Path(self._filename) if Path(self._filename).is_file()
                    else fetch_lammps_file(self._filename))
            text = path.read_text()
        self.entries, comments = parse(text)
        self.citation = [c for c in comments if "CITATION" in c.upper() or "Phys" in c][:3]
        self.elements = sorted({e for key in self.entries for e in key})
        self.cutoff_A = max(e.R + e.D for e in self.entries.values())
        self._loaded = True

    def composition(self, structure: Structure) -> Tuple[bool, str]:
        try:
            self._load()
        except UnsupportedSystem as exc:
            return False, str(exc)
        present = sorted({pt.symbol(int(z)) for z in structure.numbers})
        missing = [e for e in present if e not in self.elements]
        if missing:
            return False, (f"{self.file} has no Tersoff parameters for {', '.join(missing)}; it "
                           f"covers {', '.join(self.elements)}.")
        for a in present:
            for b in present:
                for c in present:
                    if (a, b, c) not in self.entries:
                        return False, f"{self.file} lacks the {a}-{b}-{c} entry."
        return True, ""

    @staticmethod
    def _cutoff(r, R, D):
        inner = r < R - D
        outer = r > R + D
        middle = ~(inner | outer)
        value = np.where(inner, 1.0, 0.0)
        slope = np.zeros_like(r)
        x = math.pi / 2 * (r[middle] - R[middle]) / D[middle]
        value[middle] = 0.5 - 0.5 * np.sin(x)
        slope[middle] = -0.5 * np.cos(x) * math.pi / (2 * D[middle])
        return value, slope

    def energy_and_forces(self, structure: Structure) -> Tuple[float, np.ndarray]:
        ok, why = self.composition(structure)
        if not ok:
            raise UnsupportedSystem(why)
        n_atoms = len(structure)
        forces = np.zeros((n_atoms, 3))
        nl = self.neighbours(structure, self.cutoff_A)
        if len(nl) == 0:
            return 0.0, forces
        symbols = np.array([pt.symbol(int(z)) for z in structure.numbers])
        si, sj = symbols[nl.i], symbols[nl.j]

        def pair(field):
            return np.array([getattr(self.entries[(a, b, b)], field) for a, b in zip(si, sj)])

        R2, D2 = pair("R"), pair("D")
        keep = nl.d < R2 + D2
        i, j, vec, r = nl.i[keep], nl.j[keep], nl.D[keep], nl.d[keep]
        si, sj = si[keep], sj[keep]
        if len(i) == 0:
            return 0.0, forces
        P = {f: np.array([getattr(self.entries[(a, b, b)], f) for a, b in zip(si, sj)])
             for f in ("n", "beta", "lambda2", "B", "lambda1", "A", "R", "D")}
        fc, dfc = self._cutoff(r, P["R"], P["D"])
        fr = P["A"] * np.exp(-P["lambda1"] * r)
        fa = -P["B"] * np.exp(-P["lambda2"] * r)
        dfr = -P["lambda1"] * fr
        dfa = -P["lambda2"] * fa
        order = np.argsort(i, kind="stable")
        i, j, vec, r = i[order], j[order], vec[order], r[order]
        fc, dfc, fr, fa, dfr, dfa = (a[order] for a in (fc, dfc, fr, fa, dfr, dfa))
        si, sj = si[order], sj[order]
        P = {k: v[order] for k, v in P.items()}
        starts = np.searchsorted(i, np.arange(n_atoms))
        ends = np.searchsorted(i, np.arange(n_atoms), side="right")
        tri_p, tri_q = [], []
        for atom in range(n_atoms):
            idx = np.arange(starts[atom], ends[atom])
            if len(idx) < 2:
                continue
            p, q = np.meshgrid(idx, idx, indexing="ij")
            mask = p != q
            tri_p.append(p[mask])
            tri_q.append(q[mask])
        zeta = np.zeros(len(i))
        if tri_p:
            tp = np.concatenate(tri_p)
            tq = np.concatenate(tri_q)
            T = {f: np.array([getattr(self.entries[(si[a], sj[a], sj[b])], f)
                              for a, b in zip(tp, tq)])
                 for f in ("m", "gamma", "lambda3", "c", "d", "costheta0", "R", "D")}
            rij, rik = r[tp], r[tq]
            uij, uik = vec[tp], vec[tq]
            cos = np.einsum("ij,ij->i", uij, uik) / (rij * rik)
            fck, dfck = self._cutoff(rik, T["R"], T["D"])
            h = cos - T["costheta0"]
            c2, d2 = T["c"] ** 2, T["d"] ** 2
            g = T["gamma"] * (1 + c2 / d2 - c2 / (d2 + h * h))
            dg = T["gamma"] * c2 * 2 * h / (d2 + h * h) ** 2
            delta = rij - rik
            m3 = np.isclose(T["m"], 3.0)
            arg = np.where(m3, (T["lambda3"] * delta) ** 3, T["lambda3"] * delta)
            darg = np.where(m3, 3 * T["lambda3"] ** 3 * delta ** 2, T["lambda3"])
            ex = np.exp(arg)
            term = fck * g * ex
            np.add.at(zeta, tp, term)
        beta, n = P["beta"], P["n"]
        bz = (beta * zeta) ** n
        b = (1 + bz) ** (-0.5 / n)
        with np.errstate(divide="ignore", invalid="ignore"):
            db = np.where(zeta > 0, -0.5 * b * bz / (zeta * (1 + bz)), 0.0)
        energy = 0.5 * float(np.sum(fc * (fr + b * fa)))
        de_dr = 0.5 * (dfc * (fr + b * fa) + fc * (dfr + b * dfa))
        unit = vec / r[:, None]
        direct = de_dr[:, None] * unit
        np.add.at(forces, j, -direct)
        np.add.at(forces, i, direct)
        if tri_p:
            w = (0.5 * fc * fa * db)[tp]
            dterm_drij = fck * g * ex * darg
            dterm_drik = dfck * g * ex - fck * g * ex * darg
            dterm_dcos = fck * dg * ex
            eij = uij / rij[:, None]
            eik = uik / rik[:, None]
            dcos_dj = (eik - cos[:, None] * eij) / rij[:, None]
            dcos_dk = (eij - cos[:, None] * eik) / rik[:, None]
            grad_j = (dterm_drij[:, None] * eij + dterm_dcos[:, None] * dcos_dj) * w[:, None]
            grad_k = (dterm_drik[:, None] * eik + dterm_dcos[:, None] * dcos_dk) * w[:, None]
            np.add.at(forces, j[tp], -grad_j)
            np.add.at(forces, j[tq], -grad_k)
            np.add.at(forces, i[tp], grad_j + grad_k)
        return energy, forces

    def describe(self) -> dict:
        if not self._loaded:
            return {"model": self.name, "fidelity": self.fidelity.value, "cutoff_A": None,
                    "parameters": {"file": self.file, "elements": self._declared},
                    "approximations": ["Tersoff bond-order potential; parameters are read "
                                       "from the LAMMPS file on first use."],
                    "references": ["J. Tersoff, Phys. Rev. B 39 (1989) 5566"]}
        return {"model": self.name, "fidelity": self.fidelity.value, "cutoff_A": self.cutoff_A,
                "parameters": {"file": self.file, "elements": self.elements,
                               "entries": {"-".join(k): vars(v) for k, v in
                                           sorted(self.entries.items())}},
                "approximations": [
                    "Tersoff bond-order potential: bond strength depends on the local "
                    "coordination through b_ij; no electrons, no long-range terms.",
                    "Smooth sine cutoff between R - D and R + D, as in LAMMPS.",
                ],
                "references": self.citation + ["J. Tersoff, Phys. Rev. B 39 (1989) 5566"]}
