"""Simulated atomic force microscopy.

Force model
-----------
The tip-sample interaction is split into a short-range chemical part and a
long-range van der Waals part:

``short range``
    A 12-6 Lennard-Jones interaction between the apex atom (or a small number
    of equivalent apex atoms) and every sample atom, with Lorentz-Berthelot
    mixing.  This gives site-dependent contrast with the correct symmetry and
    the correct sign changes, but it is *not* a chemical-bonding calculation:
    covalent tip-sample bonding, which dominates true atomic contrast on
    semiconductors, requires an electronic-structure force calculation.

``long range``
    The Hamaker sphere-plane result

        F_vdW(z) = - H R / (6 z^2)

    with ``H`` the Hamaker constant and ``R`` the apex radius of curvature.
    It carries no atomic contrast and is responsible for the large background
    force and frequency shift.

Modes
-----
``contact``
    The height at which the total vertical force equals a setpoint.
``constant-height``
    Force (or frequency shift) on a plane.
``fm-afm``
    Frequency shift of an oscillating cantilever, evaluated with Giessibl's
    large-amplitude integral

        df = -(f0 / (k A^2)) (1/pi) integral over q' of
             F_ts(z + A + q') q' / sqrt(A^2 - q'^2) dq'

    computed by Gauss-Chebyshev quadrature.  For ``A -> 0`` this reduces to the
    small-amplitude limit ``df = -(f0 / 2k) dF/dz``.

What is not modelled
--------------------
Covalent tip-sample bonding, tip relaxation and tip-atom flexing (which
dominate CO-tip imaging), dissipation channels, electrostatic force from
contact-potential differences, and cantilever dynamics beyond the
first-harmonic approximation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core_model.structure import Structure
from ..elements import periodic_table as pt
from ..physics.potentials import FCC_ECOH_OVER_EPS, FCC_R0_OVER_SIGMA
from ..provenance import Convergence, Fidelity, Origin, Provenance, Result
from .noise import NoiseModel, apply_noise, scan_noise_provenance
from .scan import ScanResult
from .tip import Tip

MODES = ("contact", "constant-height", "fm-afm", "constant-frequency-shift")

GENERIC_LJ = {
    "Si": (4.63 / FCC_ECOH_OVER_EPS, 2.3517 / FCC_R0_OVER_SIGMA),
    "Ge": (3.85 / FCC_ECOH_OVER_EPS, 2.4500 / FCC_R0_OVER_SIGMA),
    "C": (7.37 / FCC_ECOH_OVER_EPS, 1.5445 / FCC_R0_OVER_SIGMA),
    "Au": (3.81 / FCC_ECOH_OVER_EPS, 2.8837 / FCC_R0_OVER_SIGMA),
    "Ag": (2.95 / FCC_ECOH_OVER_EPS, 2.8892 / FCC_R0_OVER_SIGMA),
    "Cu": (3.49 / FCC_ECOH_OVER_EPS, 2.5561 / FCC_R0_OVER_SIGMA),
    "Pt": (5.84 / FCC_ECOH_OVER_EPS, 2.7748 / FCC_R0_OVER_SIGMA),
    "Ni": (4.44 / FCC_ECOH_OVER_EPS, 2.4918 / FCC_R0_OVER_SIGMA),
    "W": (8.90 / FCC_ECOH_OVER_EPS, 2.7411 / FCC_R0_OVER_SIGMA),
    "Ga": (2.81 / FCC_ECOH_OVER_EPS, 2.4400 / FCC_R0_OVER_SIGMA),
    "As": (2.96 / FCC_ECOH_OVER_EPS, 2.4481 / FCC_R0_OVER_SIGMA),
    "O": (2.60 / FCC_ECOH_OVER_EPS, 1.4800 / FCC_R0_OVER_SIGMA),
    "N": (4.92 / FCC_ECOH_OVER_EPS, 1.4500 / FCC_R0_OVER_SIGMA),
    "B": (5.81 / FCC_ECOH_OVER_EPS, 1.7000 / FCC_R0_OVER_SIGMA),
    "S": (2.85 / FCC_ECOH_OVER_EPS, 2.0500 / FCC_R0_OVER_SIGMA),
    "Se": (2.46 / FCC_ECOH_OVER_EPS, 2.3200 / FCC_R0_OVER_SIGMA),
    "Mo": (6.82 / FCC_ECOH_OVER_EPS, 2.7250 / FCC_R0_OVER_SIGMA),
    "In": (2.52 / FCC_ECOH_OVER_EPS, 3.2500 / FCC_R0_OVER_SIGMA),
    "P": (3.43 / FCC_ECOH_OVER_EPS, 2.2000 / FCC_R0_OVER_SIGMA),
    "Al": (3.39 / FCC_ECOH_OVER_EPS, 2.8635 / FCC_R0_OVER_SIGMA),
    "H": (2.26 / FCC_ECOH_OVER_EPS, 0.7400 / FCC_R0_OVER_SIGMA),
}


@dataclass
class AFMSettings:
    """Instrument settings for a simulated AFM scan."""

    mode: str = "fm-afm"
    resolution: Tuple[int, int] = (256, 256)
    window_A: Optional[Tuple[float, float, float, float]] = None
    height_A: float = 4.0
    force_setpoint_eV_A: float = 0.05
    frequency_shift_setpoint_Hz: float = -10.0
    resonance_frequency_Hz: float = 30000.0
    stiffness_N_m: float = 1800.0
    oscillation_amplitude_A: float = 1.0
    quadrature_points: int = 17
    z_min_A: float = 1.5
    z_max_A: float = 10.0
    lj_cutoff_A: float = 10.0
    lateral_images: int = 1
    square_pixels: bool = True
    scan_angle_deg: float = 0.0
    solver_iterations: int = 24
    noise: NoiseModel = field(default_factory=lambda: NoiseModel.realistic(0))

    def describe(self) -> dict:
        return {
            "mode": self.mode, "resolution": list(self.resolution),
            "window_A": list(self.window_A) if self.window_A else None,
            "height_A": self.height_A,
            "force_setpoint_eV_A": self.force_setpoint_eV_A,
            "frequency_shift_setpoint_Hz": self.frequency_shift_setpoint_Hz,
            "resonance_frequency_Hz": self.resonance_frequency_Hz,
            "stiffness_N_m": self.stiffness_N_m,
            "oscillation_amplitude_A": self.oscillation_amplitude_A,
            "quadrature_points": self.quadrature_points,
            "z_bracket_A": [self.z_min_A, self.z_max_A],
            "lj_cutoff_A": self.lj_cutoff_A,
            "square_pixels": self.square_pixels,
            "scan_angle_deg": self.scan_angle_deg,
            "noise": self.noise.describe(),
        }


EV_PER_A_TO_NN = 1.602176634
N_PER_M_TO_EV_A2 = 1.0 / 16.021766208


class AFMSimulator:
    """Classical-force AFM with Giessibl frequency-shift conversion."""

    name = "microscopy/afm-lj-hamaker"
    fidelity = Fidelity.TIER1_CLASSICAL

    def __init__(self, tip: Optional[Tip] = None,
                 apex_element: Optional[str] = None) -> None:
        self.tip = tip or Tip("W", radius_A=50.0)
        self.apex_element = apex_element or (
            self.tip.functionalisation or self.tip.material)
        if self.apex_element == "PtIr":
            self.apex_element = "Pt"
        if self.apex_element not in GENERIC_LJ:
            raise ValueError(
                f"No generic Lennard-Jones parameters for apex element "
                f"{self.apex_element!r}. Known: {sorted(GENERIC_LJ)}."
            )

    def _pair_parameters(self, structure: Structure) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        eps_t, sig_t = GENERIC_LJ[self.apex_element]
        eps, sig, missing = [], [], []
        for z in structure.numbers:
            sym = pt.symbol(int(z))
            if sym not in GENERIC_LJ:
                missing.append(sym)
                eps.append(np.nan)
                sig.append(np.nan)
            else:
                e, s = GENERIC_LJ[sym]
                eps.append(np.sqrt(e * eps_t))
                sig.append(0.5 * (s + sig_t))
        return np.array(eps), np.array(sig), sorted(set(missing))

    @staticmethod
    def _force_kernel(dxy2: np.ndarray, az: np.ndarray, z: np.ndarray,
                      eps: np.ndarray, sig2: np.ndarray, rc2: float,
                      hamaker_eV: float, radius_A: float, surface_z: float) -> np.ndarray:
        """Vertical force in eV/A for one chunk of tip positions.

        ``dxy2`` is the ``(pixels, atoms)`` squared lateral distance.  It does
        not depend on the tip height, so it is computed once per chunk and
        reused for every height evaluated there -- which is what makes
        frequency-shift quadrature and setpoint solving affordable.
        Positive force means repulsive.
        """
        dz = z[:, None] - az[None, :]
        r2 = dxy2 + dz * dz
        np.maximum(r2, 1e-6, out=r2)
        inv2 = np.where(r2 < rc2, 1.0 / r2, 0.0)
        s6 = (sig2[None, :] * inv2) ** 3
        coef = 4.0 * eps[None, :] * (-12.0 * s6 * s6 + 6.0 * s6) * inv2
        fz = -np.einsum("pa,pa->p", coef, dz)
        gap = np.maximum(z - surface_z, 0.5)
        return fz - hamaker_eV * radius_A / (6.0 * gap * gap)

    @staticmethod
    def _chebyshev_nodes(amplitude_A: float, n_quad: int):
        """Chebyshev-Gauss nodes and weights of the FIRST kind.

        The Giessibl integrand carries the weight ``1 / sqrt(A^2 - q^2)``, which
        is the first-kind weight. Substituting ``q = A cos(theta)`` turns the
        integral into ``int_0^pi f(A cos theta) d theta``, for which the
        first-kind rule with nodes ``theta_j = (2j - 1) pi / (2n)`` and equal
        weights ``pi / n`` is exact for polynomials up to degree ``2n - 1``.
        Using the second-kind node set with these weights instead would
        under-estimate the integral by a factor ``(n - 1) / (n + 1)``.
        """
        j = np.arange(1, n_quad + 1)
        theta = (2.0 * j - 1.0) * np.pi / (2.0 * n_quad)
        return amplitude_A * np.cos(theta), np.pi / n_quad * np.ones(n_quad)

    def frequency_shift(self, force_fn, z: np.ndarray, amplitude_A: float,
                        f0_Hz: float, k_N_m: float, n_quad: int = 17) -> np.ndarray:
        """Giessibl large-amplitude frequency shift, in Hz.

        ``force_fn(z_array) -> force_array`` in eV/A.  The integral

            df = -(f0 / (k A^2)) (1/pi) int F(z + A + q) q / sqrt(A^2 - q^2) dq

        is evaluated with Gauss-Chebyshev quadrature of the second kind, whose
        weight matches the ``1/sqrt(A^2 - q^2)`` kernel exactly.  As
        ``A -> 0`` this reduces to ``df = -(f0 / 2k) dF/dz``.
        """
        A = float(amplitude_A)
        if A <= 0:
            raise ValueError("Oscillation amplitude must be positive")
        k_eV = k_N_m * N_PER_M_TO_EV_A2
        q, w = self._chebyshev_nodes(A, n_quad)
        zz = np.asarray(z, dtype=float)
        total = np.zeros_like(zz)
        for qj, wj in zip(q, w):
            total = total + wj * force_fn(zz + A + qj) * qj
        return -(f0_Hz / (k_eV * A * A)) / np.pi * total

    def scan(self, structure: Structure, settings: Optional[AFMSettings] = None) -> ScanResult:
        settings = settings or AFMSettings()
        if settings.mode not in MODES:
            raise ValueError(f"Unknown AFM mode {settings.mode!r}; implemented: {MODES}")
        t0 = time.perf_counter()

        eps, sig, missing = self._pair_parameters(structure)
        if missing:
            return ScanResult.unsupported_result(
                self.name,
                f"No tip-sample interaction parameters for {', '.join(missing)}. "
                "Supply a potential explicitly or use an external force engine.",
                settings.describe(),
                suggested=["external:lammps", "external:gpaw (DFT forces)"])

        cell = structure.cell.matrix
        if settings.window_A is None:
            x0, y0 = 0.0, 0.0
            x1 = float(cell[0][0] + max(cell[1][0], 0.0))
            y1 = float(cell[1][1] + max(cell[0][1], 0.0))
            if x1 <= x0 or y1 <= y0:
                p = structure.positions
                x0, y0, x1, y1 = (float(p[:, 0].min()), float(p[:, 1].min()),
                                  float(p[:, 0].max()), float(p[:, 1].max()))
        else:
            x0, y0, x1, y1 = settings.window_A

        nx, ny = settings.resolution
        if settings.square_pixels and x1 > x0 and y1 > y0:
            pitch = (x1 - x0) / max(nx - 1, 1)
            ny = max(8, int(round((y1 - y0) / pitch)) + 1)
        xs = np.linspace(x0, x1, nx)
        ys = np.linspace(y0, y1, ny)
        gx, gy = np.meshgrid(xs, ys)
        if settings.scan_angle_deg:
            th = np.radians(settings.scan_angle_deg)
            cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
            gx, gy = (cx + (gx - cx) * np.cos(th) - (gy - cy) * np.sin(th),
                      cy + (gx - cx) * np.sin(th) + (gy - cy) * np.cos(th))
        xy = np.column_stack([gx.ravel(), gy.ravel()])

        shifts = ([(i, j) for i in range(-settings.lateral_images, settings.lateral_images + 1)
                   for j in range(-settings.lateral_images, settings.lateral_images + 1)]
                  if structure.cell.is_periodic else [(0, 0)])
        atom_pos = np.concatenate(
            [structure.positions + i * cell[0] + j * cell[1] for i, j in shifts])
        eps_rep = np.tile(eps, len(shifts))
        sig_rep = np.tile(sig, len(shifts))
        keep = ((atom_pos[:, 0] > x0 - settings.lj_cutoff_A)
                & (atom_pos[:, 0] < x1 + settings.lj_cutoff_A)
                & (atom_pos[:, 1] > y0 - settings.lj_cutoff_A)
                & (atom_pos[:, 1] < y1 + settings.lj_cutoff_A)
                & (atom_pos[:, 2] > structure.positions[:, 2].max() - settings.lj_cutoff_A))
        atom_pos, eps_rep, sig_rep = atom_pos[keep], eps_rep[keep], sig_rep[keep]

        surface_z = float(structure.positions[:, 2].max())
        rc2 = settings.lj_cutoff_A ** 2
        sig2 = sig_rep ** 2
        amp = settings.oscillation_amplitude_A
        nq = settings.quadrature_points
        f0, kcant = settings.resonance_frequency_Hz, settings.stiffness_N_m
        n_pix = xy.shape[0]

        flat: Dict[str, np.ndarray] = {"topography": np.empty(n_pix),
                                       "force": np.empty(n_pix)}
        wants_df = settings.mode in ("fm-afm", "constant-height",
                                     "constant-frequency-shift")
        if wants_df:
            flat["frequency_shift"] = np.empty(n_pix)
        residual, iterations = 0.0, 0
        table_spacing_A = None
        no_crossing = 0

        for p0 in range(0, n_pix, 4096):
            p1 = min(p0 + 4096, n_pix)
            tip_xy = xy[p0:p1]
            dx = tip_xy[:, 0][:, None] - atom_pos[None, :, 0]
            dy = tip_xy[:, 1][:, None] - atom_pos[None, :, 1]
            dxy2 = dx * dx + dy * dy
            local = dxy2.min(axis=0) < rc2
            dxy2 = np.ascontiguousarray(dxy2[:, local])
            az, eps_l, sig2_l = atom_pos[local, 2], eps_rep[local], sig2[local]
            m = p1 - p0

            def force(zv, _d=dxy2, _az=az, _e=eps_l, _s=sig2_l):
                return self._force_kernel(_d, _az, np.asarray(zv, dtype=float),
                                          _e, _s, rc2, self.tip.hamaker_eV,
                                          self.tip.radius_A, surface_z)

            def dfreq(zv):
                return self.frequency_shift(force, zv, amp, f0, kcant, nq)

            if settings.mode not in ("constant-height", "fm-afm"):
                zlo = surface_z + settings.z_min_A - amp
                zhi = surface_z + settings.z_max_A + 2.0 * amp
                n_lev = int(max(64, np.ceil((zhi - zlo) / 0.10)))
                ztab = np.linspace(zlo, zhi, n_lev)
                ftab = np.empty((n_lev, m))
                for li, lv in enumerate(ztab):
                    ftab[li] = force(np.full(m, lv))
                dzt = ztab[1] - ztab[0]
                cols = np.arange(m)
                table_spacing_A = float(dzt)

                def force(zv, _t=ztab, _f=ftab, _d=dzt, _c=cols, _n=n_lev):
                    t = np.clip((np.asarray(zv, dtype=float) - _t[0]) / _d,
                                0.0, _n - 1 - 1e-9)
                    i0 = t.astype(np.int64)
                    frac = t - i0
                    return _f[i0, _c] * (1.0 - frac) + _f[i0 + 1, _c] * frac

                def dfreq(zv):
                    return self.frequency_shift(force, zv, amp, f0, kcant, nq)

            if settings.mode in ("constant-height", "fm-afm"):
                z = np.full(m, surface_z + settings.height_A)
                flat["topography"][p0:p1] = z
                flat["force"][p0:p1] = force(z)
                flat["frequency_shift"][p0:p1] = dfreq(z)
            else:
                signal = force if settings.mode == "contact" else dfreq
                target = (settings.force_setpoint_eV_A if settings.mode == "contact"
                          else settings.frequency_shift_setpoint_Hz)
                levels = np.linspace(surface_z + settings.z_max_A,
                                     surface_z + settings.z_min_A, 40)
                vals = np.empty((len(levels), m))
                for li, lv in enumerate(levels):
                    vals[li] = signal(np.full(m, lv))
                sgn = np.sign(vals - target)
                change = sgn[:-1] * sgn[1:] <= 0
                has_root = change.any(axis=0)
                no_crossing += int((~has_root).sum())
                first = np.where(has_root, change.argmax(axis=0), len(levels) - 2)
                hi = levels[first]
                lo = levels[first + 1]
                for iterations in range(1, settings.solver_iterations + 1):
                    mid = 0.5 * (lo + hi)
                    s_mid = signal(mid)
                    s_hi = signal(hi)
                    root_between_mid_and_hi = (
                        np.sign(s_mid - target) * np.sign(s_hi - target) <= 0)
                    lo = np.where(root_between_mid_and_hi, mid, lo)
                    hi = np.where(root_between_mid_and_hi, hi, mid)
                    if float(np.abs(hi - lo).max()) < 1e-4:
                        break
                z = 0.5 * (lo + hi)
                final = signal(z)
                scale = max(abs(target), 1e-12)
                err = np.abs(final - target) / scale
                if has_root.any():
                    residual = max(residual, float(err[has_root].max()))
                flat["topography"][p0:p1] = z
                flat["force"][p0:p1] = force(z)
                if wants_df:
                    flat["frequency_shift"][p0:p1] = (
                        final if settings.mode == "constant-frequency-shift" else dfreq(z))

        channels = {k: v.reshape(ny, nx) for k, v in flat.items()}
        channels["force_nN"] = channels["force"] * EV_PER_A_TO_NN
        units = {"topography": "A", "force": "eV/A", "force_nN": "nN",
                 "frequency_shift": "Hz", "raw_signal": ""}
        primary = {"contact": "topography", "constant-frequency-shift": "topography",
                   "constant-height": "force", "fm-afm": "frequency_shift"}[settings.mode]

        clean = channels[primary].copy()
        noisy, applied = apply_noise(clean, settings.noise)
        channels["raw_signal"] = clean
        channels[primary] = noisy
        units["raw_signal"] = units.get(primary, "")

        prov = Provenance(
            model=self.name,
            fidelity=self.fidelity,
            origin=Origin.CALCULATED,
            approximations=[
                "Short-range force from a Lennard-Jones tip-sample pair interaction "
                "with generic parameters derived from cohesive energies; this is an "
                "estimate of dispersion and Pauli repulsion, NOT a chemical-bonding "
                "calculation. True atomic contrast on semiconductors is dominated by "
                "covalent tip-sample bonding, which needs an electronic-structure "
                "force calculation.",
                "Long-range background from the Hamaker sphere-plane expression with "
                f"H = {self.tip.hamaker_eV} eV and R = {self.tip.radius_A} A.",
                "Rigid tip and rigid sample: no tip-apex relaxation and no sample "
                "relaxation under the tip. CO-tip-style lateral flexing is not modelled.",
                ("Frequency shift from Giessibl's large-amplitude integral evaluated "
                 f"with {settings.quadrature_points}-point Gauss-Chebyshev quadrature."
                 if settings.mode in ("fm-afm", "constant-frequency-shift",
                                      "constant-height") else
                 "Static force mode; no cantilever dynamics."),
                "No electrostatic force: contact-potential differences and applied "
                "bias are not included.",
                "No dissipation channel.",
            ],
            tolerances={"setpoint_residual_relative": residual,
                        "bisection_iterations": iterations,
                        "force_table_spacing_A": table_spacing_A,
                        "pixels_without_setpoint_crossing": no_crossing},
            boundary_conditions=str(structure.cell.pbc),
            parameters={"settings": settings.describe(), "tip": self.tip.describe(),
                        "apex_element": self.apex_element,
                        "apex_lj": {"epsilon_eV": GENERIC_LJ[self.apex_element][0],
                                    "sigma_A": GENERIC_LJ[self.apex_element][1]}},
            references=[
                "F. J. Giessibl, Phys. Rev. B 56 (1997) 16010",
                "F. J. Giessibl, Rev. Mod. Phys. 75 (2003) 949",
                "H. C. Hamaker, Physica 4 (1937) 1058",
            ],
            seed=settings.noise.seed,
            notes=("Contrast maxima in an AFM image are force extrema, which on real "
                   "surfaces can be displaced from nuclear positions."),
        )
        conv = Convergence(
            converged=(settings.mode in ("constant-height", "fm-afm")) or residual < 1e-2,
            iterations=iterations, residual=residual,
            residual_metric="relative setpoint error",
            message=("Fixed-height mode; nothing to converge."
                     if settings.mode in ("constant-height", "fm-afm")
                     else (f"Bisection residual {residual:.2e}."
                           + (f" {no_crossing} pixel(s) never reached the setpoint "
                              f"within the height bracket and were clamped; those "
                              f"pixels are not valid measurements."
                              if no_crossing else ""))),
        )
        return ScanResult(
            technique="AFM", mode=settings.mode, channels=channels,
            primary_channel=primary, units=units,
            extent_A=(x0, y0, x1, y1), resolution=(nx, ny),
            structure=structure, provenance=prov, convergence=conv,
            noise_record=scan_noise_provenance(settings.noise, applied),
            settings=settings.describe(),
            wall_time_s=time.perf_counter() - t0,
        )

    def force_curve(self, structure: Structure, x: float, y: float,
                    z_range_A: Tuple[float, float] = (2.0, 12.0),
                    n_points: int = 120,
                    settings: Optional[AFMSettings] = None) -> Result:
        """Vertical force and frequency-shift spectroscopy at one lateral point."""
        settings = settings or AFMSettings()
        eps, sig, missing = self._pair_parameters(structure)
        if missing:
            from ..provenance import unsupported
            return unsupported(
                "force_curve", self.name,
                f"No tip-sample parameters for {', '.join(missing)}.",
                suggested_models=["external:lammps"], unit="eV/A")
        cell = structure.cell.matrix
        shifts = ([(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1)]
                  if structure.cell.is_periodic else [(0, 0)])
        atom_pos = np.concatenate(
            [structure.positions + i * cell[0] + j * cell[1] for i, j in shifts])
        eps_rep, sig_rep = np.tile(eps, len(shifts)), np.tile(sig, len(shifts))
        surface_z = float(structure.positions[:, 2].max())
        xy = np.array([[x, y]])

        dxy2 = ((atom_pos[:, 0] - x) ** 2 + (atom_pos[:, 1] - y) ** 2)[None, :]
        rc2 = settings.lj_cutoff_A ** 2

        def force(zv):
            return self._force_kernel(dxy2, atom_pos[:, 2], np.asarray(zv, dtype=float),
                                      eps_rep, sig_rep ** 2, rc2, self.tip.hamaker_eV,
                                      self.tip.radius_A, surface_z)

        heights = np.linspace(z_range_A[0], z_range_A[1], n_points)
        f = np.array([float(force(np.array([surface_z + h]))[0]) for h in heights])
        df = np.array([float(self.frequency_shift(
            force, np.array([surface_z + h]), settings.oscillation_amplitude_A,
            settings.resonance_frequency_Hz, settings.stiffness_N_m,
            settings.quadrature_points)[0]) for h in heights])
        prov = Provenance(
            model=self.name + "/force-curve",
            fidelity=self.fidelity,
            origin=Origin.CALCULATED,
            approximations=[
                "Lennard-Jones short range plus Hamaker long range; rigid tip and sample.",
                "No chemical bonding, no electrostatics, no dissipation.",
            ],
            parameters={"lateral_position_A": [x, y], "apex_element": self.apex_element,
                        "tip": self.tip.describe()},
            references=["F. J. Giessibl, Rev. Mod. Phys. 75 (2003) 949"],
        )
        return Result(
            "force_curve",
            {"height_above_surface_A": heights, "force_eV_A": f,
             "force_nN": f * EV_PER_A_TO_NN, "frequency_shift_Hz": df},
            "mixed", prov,
            extra={"minimum_force_eV_A": float(f.min()),
                   "height_of_minimum_A": float(heights[int(np.argmin(f))])})
