"""Dimensionally constrained discovery of candidate laws from data.

Two tools, both Materia-native:

Dimensional analysis
    Every variable carries a vector of exponents of the SI base dimensions
    (M, L, T, I, Theta, N, J).  The dimensionless groups of the Buckingham Pi
    theorem are an integer basis of the null space of the dimension matrix,
    found exactly with rational arithmetic and reduced to small integers.

Sparse regression (sequentially thresholded least squares)
    A target is fitted as a sparse linear combination of library terms by
    repeated least squares, zeroing coefficients below a threshold
    (S. L. Brunton, J. L. Proctor and J. N. Kutz, PNAS 113 (2016) 3932).
    Library terms must all carry the target's dimensions; terms that do not
    are refused before fitting.  The data are split into fitting and held-out
    parts; the result reports the held-out error and the stability of the
    selected terms over bootstrap resamples.

What comes out is a candidate law: an expression that describes these data
within the stated error.  It is not a theorem, and it says nothing outside
the range of the data.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..provenance import Fidelity, Origin, Provenance, Result

BASE = ("M", "L", "T", "I", "Theta", "N", "J")
SUBSET_LIMIT = 16
MAX_SUBSET = 4


class DiscoveryError(ValueError):
    pass


def dimension_vector(exponents: Mapping[str, float]) -> Tuple[Fraction, ...]:
    unknown = set(exponents) - set(BASE)
    if unknown:
        raise DiscoveryError(f"Unknown base dimensions {sorted(unknown)}; use {BASE}.")
    return tuple(Fraction(exponents.get(b, 0)).limit_denominator(1000) for b in BASE)


def _null_space(matrix: List[List[Fraction]]) -> List[List[Fraction]]:
    rows, cols = len(matrix), len(matrix[0])
    m = [row[:] for row in matrix]
    pivots = []
    r = 0
    for c in range(cols):
        pivot = next((i for i in range(r, rows) if m[i][c] != 0), None)
        if pivot is None:
            continue
        m[r], m[pivot] = m[pivot], m[r]
        lead = m[r][c]
        m[r] = [v / lead for v in m[r]]
        for i in range(rows):
            if i != r and m[i][c] != 0:
                factor = m[i][c]
                m[i] = [a - factor * b for a, b in zip(m[i], m[r])]
        pivots.append(c)
        r += 1
        if r == rows:
            break
    free = [c for c in range(cols) if c not in pivots]
    basis = []
    for f in free:
        vector = [Fraction(0)] * cols
        vector[f] = Fraction(1)
        for i, p in enumerate(pivots):
            vector[p] = -m[i][f]
        basis.append(vector)
    return basis


def pi_groups(variables: Mapping[str, Mapping[str, float]]) -> List[Dict[str, int]]:
    """Dimensionless groups as integer exponent maps over the named variables."""
    names = list(variables)
    if len(names) < 2:
        raise DiscoveryError("At least two variables are needed.")
    vectors = [dimension_vector(variables[n]) for n in names]
    matrix = [[vectors[j][i] for j in range(len(names))] for i in range(len(BASE))]
    out = []
    for vector in _null_space(matrix):
        denominator = math.lcm(*[v.denominator for v in vector])
        integers = [int(v * denominator) for v in vector]
        divisor = math.gcd(*[abs(v) for v in integers if v]) or 1
        integers = [v // divisor for v in integers]
        first = next(v for v in integers if v)
        if first < 0:
            integers = [-v for v in integers]
        out.append({n: e for n, e in zip(names, integers) if e})
    return out


def check_dimensions(target: Mapping[str, float],
                     terms: Mapping[str, Mapping[str, float]]) -> List[str]:
    """Names of library terms whose dimensions differ from the target's."""
    want = dimension_vector(target)
    return [name for name, dims in terms.items() if dimension_vector(dims) != want]


def stlsq(library: np.ndarray, target: np.ndarray, threshold: float,
          iterations: int = 20) -> np.ndarray:
    """Sequentially thresholded least squares on column-normalised data."""
    scale = np.linalg.norm(library, axis=0)
    if np.any(scale == 0):
        raise DiscoveryError("A library term is identically zero on the data.")
    normalised = library / scale
    coefficients = np.linalg.lstsq(normalised, target, rcond=None)[0]
    for _ in range(iterations):
        small = np.abs(coefficients) < threshold * np.abs(coefficients).max()
        coefficients[small] = 0.0
        active = ~small
        if not active.any():
            break
        refit = np.linalg.lstsq(normalised[:, active], target, rcond=None)[0]
        new = np.zeros_like(coefficients)
        new[active] = refit
        if np.allclose(new, coefficients):
            break
        coefficients = new
    return coefficients / scale


def discover(x: np.ndarray, y: np.ndarray, terms: Mapping[str, Callable[[np.ndarray], np.ndarray]],
             term_dimensions: Mapping[str, Mapping[str, float]],
             target_dimensions: Mapping[str, float], threshold="auto",
             tolerance: float = 0.1,
             holdout_fraction: float = 0.25, bootstrap: int = 50, seed: int = 0,
             target_name: str = "y") -> Result:
    """A sparse candidate law ``y = sum_k c_k term_k(x)`` with held-out and bootstrap checks."""
    if set(terms) != set(term_dimensions):
        raise DiscoveryError("Every term needs dimensions and every dimension a term.")
    wrong = check_dimensions(target_dimensions, term_dimensions)
    if wrong:
        raise DiscoveryError(f"Terms {wrong} do not have the target's dimensions; a law "
                             "mixing them would not be dimensionally consistent.")
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) != len(y) or len(y) < 2 * len(terms):
        raise DiscoveryError("Need at least twice as many samples as library terms.")
    names = list(terms)
    library = np.column_stack([np.asarray(terms[n](x), dtype=float) for n in names])
    if not np.all(np.isfinite(library)) or not np.all(np.isfinite(y)):
        raise DiscoveryError("The library or target has non-finite values on these data.")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y))
    held = order[:int(round(holdout_fraction * len(y)))]
    fit = order[len(held):]
    scale_y = float(np.sqrt(np.mean(y ** 2))) or 1.0
    def choose(fit_rows: np.ndarray):
        if threshold != "auto":
            return "stlsq", float(threshold), stlsq(library[fit_rows], y[fit_rows],
                                                    float(threshold))
        if not len(held):
            raise DiscoveryError("Automatic selection needs held-out data.")
        candidates = []
        kind = "stlsq"
        if len(names) <= SUBSET_LIMIT:
            from itertools import combinations
            kind = "best-subset"
            for size in range(1, min(MAX_SUBSET, len(names)) + 1):
                for subset in combinations(range(len(names)), size):
                    columns = list(subset)
                    fitted = np.linalg.lstsq(library[np.ix_(fit_rows, columns)], y[fit_rows],
                                             rcond=None)[0]
                    c = np.zeros(len(names))
                    c[columns] = fitted
                    error = float(np.sqrt(np.mean((library[held] @ c - y[held]) ** 2))) / scale_y
                    candidates.append((size, error, 0.0, c))
        for value in np.geomspace(1e-8, 0.5, 40):
            c = stlsq(library[fit_rows], y[fit_rows], float(value))
            error = float(np.sqrt(np.mean((library[held] @ c - y[held]) ** 2))) / scale_y
            candidates.append((int(np.count_nonzero(c)), error, float(value), c))
        best = min(e for _, e, _, _ in candidates)
        allowed = [item for item in candidates if item[1] <= best * (1 + tolerance) + 1e-15]
        _, _, chosen, c = min(allowed, key=lambda item: (item[0], item[1]))
        return kind, chosen, c

    method, used_threshold, coefficients = choose(fit)
    selected = [n for n, c in zip(names, coefficients) if c != 0]
    prediction = library @ coefficients
    scale = float(np.sqrt(np.mean(y ** 2))) or 1.0
    fit_error = float(np.sqrt(np.mean((prediction[fit] - y[fit]) ** 2))) / scale
    held_error = (float(np.sqrt(np.mean((prediction[held] - y[held]) ** 2))) / scale
                  if len(held) else None)
    counts = {n: 0 for n in names}
    for _ in range(int(bootstrap)):
        sample = rng.choice(fit, size=len(fit), replace=True)
        _, _, c = choose(sample)
        for n, v in zip(names, c):
            counts[n] += int(v != 0)
    stability = {n: counts[n] / max(1, bootstrap) for n in names}
    expression = " + ".join(f"({c:.10g})*{n}" for n, c in zip(names, coefficients) if c != 0)
    prov = Provenance(
        model=f"discovery/{method}", fidelity=Fidelity.NON_PHYSICAL, origin=Origin.ESTIMATED,
        approximations=["A candidate law fitted to these data by sparse regression over the "
                        "given dimensionally consistent library; not a theorem, and silent "
                        "outside the range of the data.",
                        f"Selection by {method} (threshold {float(used_threshold):g}) "
                        "choosing the sparsest model within the stated held-out tolerance of the "
                        "best when automatic; "
                        f"{len(held)} of {len(y)} samples held out; {bootstrap} bootstrap "
                        "resamples for term stability."],
        parameters={"terms": names, "threshold": float(used_threshold), "method": method,
                    "seed": seed,
                    "range": [float(np.min(x)), float(np.max(x))]},
        references=["S. L. Brunton, J. L. Proctor and J. N. Kutz, PNAS 113 (2016) 3932"],
        seed=seed)
    return Result("candidate_law", {"target": target_name, "expression": expression,
                                    "coefficients": dict(zip(names, coefficients.tolist())),
                                    "selected": selected}, "", prov,
                  extra={"relative_rms_fit": fit_error, "relative_rms_held_out": held_error,
                         "term_stability": stability, "samples": len(y)})
