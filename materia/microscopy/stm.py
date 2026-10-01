"""Simulated scanning tunnelling microscopy.

Physical model
--------------
The tunnel current is evaluated in the Tersoff-Hamann approximation
(J. Tersoff and D. R. Hamann, Phys. Rev. B 31 (1985) 805): for a structureless
s-wave tip apex at position ``r_t`` and small bias ``V``,

    I(r_t, V)  ~  integral over [E_F, E_F + eV] of rho_s(r_t, E) dE

where ``rho_s`` is the *sample* local density of states evaluated at the centre
of curvature of the tip apex.  Materia evaluates ``rho_s`` from the
tight-binding eigenstates:

    psi_n(r) = sum_{i,alpha} c_{n,i,alpha} chi_alpha(r - R_i)

with vacuum tails modelled as atom-centred exponentials,

    chi_s(r)  = exp(-kappa r)
    chi_p(r)  = (u . rhat) exp(-kappa r)

and the decay constant taken from the effective barrier

    kappa = sqrt(2 m phi_eff) / hbar ,
    phi_eff = (phi_tip + phi_sample)/2 - |eV|/2 .

What this model does and does not tell you
------------------------------------------
* It reproduces the *pattern* of the LDOS at the chosen bias: which sites
  appear bright, how the contrast inverts between filled and empty states, and
  how the corrugation decays with tip height.
* Absolute currents are **not** predicted.  The prefactor of the
  Tersoff-Hamann expression contains the tip density of states and the
  apex geometry, neither of which is known.  Materia therefore reports the
  current in *arbitrary units* and calibrates the constant-current setpoint
  against the computed signal, which is exactly what the experimental feedback
  loop does.  The reported ``current_nA`` axis is a calibrated scale, labelled
  as such.
* A bright feature is a *maximum of the vacuum LDOS*, not necessarily an atom.
  Dangling bonds, adsorbates and antibonding states can all produce maxima that
  are displaced from, or absent at, nuclear positions.  Every hover result
  states which of these it believes it has found and with what confidence.
* Not modelled: tip-induced band bending, inelastic channels, the
  bias-dependent tip DOS, current-induced forces, and any many-body effect.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core_model.structure import Structure
from ..core_model.units import kappa_inv_angstrom
from ..elements import periodic_table as pt
from ..provenance import (
    Convergence,
    Fidelity,
    Origin,
    Provenance,
    Result,
    digest,
    unsupported,
)
from ..solvers.base import Capability
from ..solvers.tight_binding import TightBinding, eigh_in_window, robust_eigh
from .noise import NoiseModel, apply_noise, scan_noise_provenance
from .tip import Tip

DEPTH_CUTOFF_A = 5.0

MODES = ("constant-current", "constant-height")


@dataclass
class STMSettings:
    """Instrument settings for a simulated STM scan."""

    bias_V: float = 1.0
    setpoint_nA: float = 0.5
    mode: str = "constant-current"
    height_A: float = 5.0
    resolution: Tuple[int, int] = (256, 256)
    window_A: Optional[Tuple[float, float, float, float]] = None
    broadening_eV: float = 0.10
    kgrid: Tuple[int, int] = (2, 2)
    lateral_images: int = 1
    electronic_depth_A: float = 9.0
    z_min_A: float = 2.5
    z_max_A: float = 14.0
    feedback_iterations: int = 4
    feedback_gain: float = 1.0
    scan_angle_deg: float = 0.0
    square_pixels: bool = True
    noise: NoiseModel = field(default_factory=lambda: NoiseModel.realistic(0))

    def describe(self) -> dict:
        return {
            "bias_V": self.bias_V, "setpoint_nA": self.setpoint_nA, "mode": self.mode,
            "height_A": self.height_A, "resolution": list(self.resolution),
            "window_A": list(self.window_A) if self.window_A else None,
            "broadening_eV": self.broadening_eV,
            "kgrid": list(self.kgrid),
            "lateral_images": self.lateral_images,
            "electronic_depth_A": self.electronic_depth_A,
            "electronic_depth_A": self.electronic_depth_A,
            "z_bracket_A": [self.z_min_A, self.z_max_A],
            "feedback_iterations": self.feedback_iterations,
            "feedback_gain": self.feedback_gain,
            "scan_angle_deg": self.scan_angle_deg,
            "square_pixels": self.square_pixels,
            "noise": self.noise.describe(),
        }


class STMSimulator:
    """Tersoff-Hamann STM on top of a tight-binding electronic structure."""

    name = "microscopy/stm-tersoff-hamann"
    fidelity = Fidelity.TIER2_SEMI_EMPIRICAL

    _electronic_cache: Dict[str, tuple] = {}
    _cache_order: List[str] = []
    _cache_limit = 4

    def __init__(self, solver: TightBinding, tip: Optional[Tip] = None,
                 sample_work_function_eV: float = 4.85) -> None:
        self.solver = solver
        self.tip = tip or Tip()
        self.sample_work_function_eV = float(sample_work_function_eV)

    @classmethod
    def clear_cache(cls) -> None:
        cls._electronic_cache.clear()
        cls._cache_order.clear()

    def decay_constant(self, bias_V: float) -> Tuple[float, float]:
        """Return ``(kappa, phi_eff)`` for the given bias."""
        phi = 0.5 * (self.tip.work_function_eV + self.sample_work_function_eV) \
            - 0.5 * abs(bias_V)
        if phi <= 0.05:
            raise ValueError(
                f"Effective barrier {phi:.3f} eV is non-positive at bias {bias_V} V: "
                "the tunnelling approximation has broken down. Reduce |V| below "
                f"{self.tip.work_function_eV + self.sample_work_function_eV:.2f} V."
            )
        return kappa_inv_angstrom(phi), phi

    def vacuum_states(self, structure: Structure, bias_V: float,
                      broadening_eV: float = 0.1,
                      kgrid: Tuple[int, int] = (1, 1),
                      lateral_images: int = 1,
                      lateral_cutoff_A: float = 6.5,
                      window_A: Optional[Tuple[float, float, float, float]] = None,
                      electronic_depth_A: float = 9.0,
                      state_threshold: float = 1e-3):
        """Build the vacuum-tail expansion of every state in the bias window.

        The near-surface atoms are replicated into lateral periodic images so
        that a rectangular scan window over a non-orthogonal cell is filled
        correctly, and so that the tip sees the periodic continuation of the
        surface rather than an artificial edge.

        Returns ``(kpoint_blocks, rep_pos, rep_kind, info)`` where each block is
        ``(coeff, weights)`` for one k-point; ``coeff`` has shape
        ``(n_replicated_orbitals, n_states)``.
        """
        cache_key = digest({
            "solver": self.solver.name,
            "numbers": structure.numbers.tolist(),
            "positions": np.round(structure.positions, 6).tolist(),
            "cell": np.round(structure.cell.matrix, 6).tolist(),
            "charges": structure.formal_charges.tolist(),
            "bias": round(float(bias_V), 6),
            "broadening": round(float(broadening_eV), 6),
            "kgrid": list(kgrid),
            "lateral_images": lateral_images,
            "lateral_cutoff": lateral_cutoff_A,
            "depth": electronic_depth_A,
            "threshold": state_threshold,
            "window": [round(float(v), 4) for v in window_A] if window_A else None,
        })
        cached = STMSimulator._electronic_cache.get(cache_key)
        if cached is not None:
            blocks, rep_pos_atoms, orbitals, info = cached
            info = dict(info)
            info["from_cache"] = True
            return blocks, rep_pos_atoms, orbitals, info

        n_orb = len(self.solver.model.orbitals)
        cell = structure.cell.matrix
        periodic_xy = structure.cell.is_periodic

        full_z = structure.positions[:, 2]
        surface_top = float(full_z.max())
        keep = full_z >= surface_top - electronic_depth_A
        truncated = bool((~keep).any())
        if truncated:
            structure = structure.subset([int(i) for i in structure.ids[keep]])

        z = structure.positions[:, 2]
        top = z.max()
        near = z >= top - DEPTH_CUTOFF_A
        near_idx = np.nonzero(near)[0]
        if near_idx.size == 0:
            return None, None, None, {}

        if periodic_xy and lateral_images > 0:
            shifts = [(i, j) for i in range(-lateral_images, lateral_images + 1)
                      for j in range(-lateral_images, lateral_images + 1)]
        else:
            shifts = [(0, 0)]

        base_pos = structure.positions[near_idx]
        rep_pos_list, rep_shift_list, rep_atom_list = [], [], []
        for (i, j) in shifts:
            offset = i * cell[0] + j * cell[1]
            candidate = base_pos + offset
            if window_A is not None and len(shifts) > 1:
                x0, y0, x1, y1 = window_A
                inside = (
                    (candidate[:, 0] > x0 - lateral_cutoff_A)
                    & (candidate[:, 0] < x1 + lateral_cutoff_A)
                    & (candidate[:, 1] > y0 - lateral_cutoff_A)
                    & (candidate[:, 1] < y1 + lateral_cutoff_A)
                )
                if not inside.any():
                    continue
            else:
                inside = np.ones(len(candidate), dtype=bool)
            rep_pos_list.append(candidate[inside])
            rep_atom_list.append(near_idx[inside])
            rep_shift_list.append(np.tile(np.array([i, j, 0], dtype=float),
                                          (int(inside.sum()), 1)))
        rep_atom = np.concatenate(rep_atom_list)
        rep_pos_atoms = np.concatenate(rep_pos_list)
        rep_shift = np.concatenate(rep_shift_list)

        rep_orb_row = (np.repeat(rep_atom, n_orb) * n_orb
                       + np.tile(np.arange(n_orb), len(rep_atom)))
        rep_shift_orb = np.repeat(rep_shift, n_orb, axis=0)

        nkx, nky = (kgrid if periodic_xy else (1, 1))
        kpoints = [((ix + 0.5) / nkx - 0.5, (iy + 0.5) / nky - 0.5, 0.0)
                   for ix in range(nkx) for iy in range(nky)]
        if (nkx, nky) == (1, 1):
            kpoints = [(0.0, 0.0, 0.0)]

        def edge(x):
            return 0.5 * (1.0 + np.tanh(x / max(broadening_eV, 1e-6)))

        n_el = self.solver._n_electrons(structure)
        pad = 3.0 * broadening_eV

        H0 = self.solver.build_hamiltonian(structure)
        reference_evals = robust_eigh(H0, eigenvectors=False)[0]
        e_f_reference, _ = self.solver.fermi_level(reference_evals, n_el)
        lo_ref, hi_ref = ((e_f_reference, e_f_reference + bias_V) if bias_V >= 0
                          else (e_f_reference + bias_V, e_f_reference))

        blocks = []
        e_f_list, n_states_total = [], 0
        dropped = 0
        for kf in kpoints:
            if periodic_xy and any(abs(v) > 1e-12 for v in kf):
                H = self.solver.build_hamiltonian(structure, k_frac=kf)
            else:
                H = H0
            evals, evecs = eigh_in_window(H, lo_ref - pad, hi_ref + pad)
            if evals.size == 0:
                evals, evecs = eigh_in_window(H, lo_ref - 10.0 * pad, hi_ref + 10.0 * pad)
            if evals.size == 0:
                continue
            e_f = e_f_reference
            e_f_list.append(e_f)
            lo, hi = lo_ref, hi_ref

            w = edge(evals - lo) * edge(hi - evals)
            c = evecs[rep_orb_row]

            amplitude = np.einsum("ij,ij->j", c.real, c.real)
            if np.iscomplexobj(c):
                amplitude = amplitude + np.einsum("ij,ij->j", c.imag, c.imag)
            if amplitude.size:
                keep_state = amplitude > amplitude.max() * state_threshold
                if keep_state.any():
                    c = c[:, keep_state]
                    w = w[keep_state]
                    dropped += int((~keep_state).sum())

            if periodic_xy and any(abs(v) > 1e-12 for v in kf):
                kcart = np.asarray(kf, dtype=float) @ structure.cell.reciprocal
                phase = np.exp(1j * (rep_shift_orb @ cell) @ kcart)
                c = c * phase[:, None]
            blocks.append((np.ascontiguousarray(c), w / len(kpoints)))
            n_states_total += int(c.shape[1])

        e_f = float(np.mean(e_f_list))
        lo, hi = (e_f, e_f + bias_V) if bias_V >= 0 else (e_f + bias_V, e_f)
        info = {
            "n_states_in_window": n_states_total,
            "states_dropped_no_surface_weight": dropped,
            "state_amplitude_threshold": state_threshold,
            "electronic_atoms": int(len(structure)),
            "electronic_orbitals": int(len(structure) * n_orb),
            "electronic_truncated": truncated,
            "electronic_depth_A": float(electronic_depth_A),
            "n_atoms_in_vacuum_sum": int(near_idx.size),
            "n_replicated_atoms": int(len(rep_atom)),
            "n_kpoints": len(kpoints),
            "kpoints_frac": [list(k) for k in kpoints],
            "fermi_level_eV": e_f,
            "window_eV": [float(lo), float(hi)],
        }
        info["from_cache"] = False
        result = (blocks, rep_pos_atoms, tuple(self.solver.model.orbitals), info)
        STMSimulator._electronic_cache[cache_key] = result
        STMSimulator._cache_order.append(cache_key)
        while len(STMSimulator._cache_order) > STMSimulator._cache_limit:
            STMSimulator._electronic_cache.pop(STMSimulator._cache_order.pop(0), None)
        return result

    def _amplitudes(self, tip_xyz: np.ndarray, atom_pos: np.ndarray,
                    orbitals: Sequence[str], kappa: float) -> np.ndarray:
        """chi_alpha(r_tip - R) for every replicated orbital and tip point.

        Distances are formed once per *atom* rather than once per orbital,
        which is the dominant cost of a scan.  Returns ``(n_points,
        n_atoms * n_orbitals)`` ordered atom-major.
        """
        d = tip_xyz[:, None, :] - atom_pos[None, :, :]
        r = np.sqrt(np.einsum("pak,pak->pa", d, d))
        np.maximum(r, 1e-6, out=r)
        radial = np.exp(-kappa * r)
        n_orb = len(orbitals)
        out = np.empty((tip_xyz.shape[0], atom_pos.shape[0], n_orb))
        for o, kind in enumerate(orbitals):
            if kind in ("s", "s*"):
                out[:, :, o] = radial
            elif kind == "px":
                out[:, :, o] = d[:, :, 0] / r * radial
            elif kind == "py":
                out[:, :, o] = d[:, :, 1] / r * radial
            elif kind == "pz":
                out[:, :, o] = d[:, :, 2] / r * radial
            else:
                raise ValueError(f"No vacuum tail defined for orbital {kind!r}")
        return out.reshape(tip_xyz.shape[0], -1)

    def current_map(
        self,
        atom_pos: np.ndarray,
        orbitals: Sequence[str],
        blocks,
        xy: np.ndarray,
        z,
        kappa: float,
        chunk: int = 4096,
    ) -> np.ndarray:
        """Tunnel current (arbitrary units) at the given tip points."""
        n = xy.shape[0]
        out = np.zeros(n)
        zz = np.broadcast_to(np.asarray(z, dtype=float).reshape(-1), (n,))
        c_re = np.ascontiguousarray(np.concatenate([np.real(c) for c, _ in blocks], axis=1))
        c_im = np.ascontiguousarray(np.concatenate([np.imag(c) for c, _ in blocks], axis=1))
        w_all = np.concatenate([w for _, w in blocks])
        has_imag = bool(np.abs(c_im).max() > 0.0)
        for start in range(0, n, chunk):
            stop = min(start + chunk, n)
            tip = np.column_stack([xy[start:stop], zz[start:stop]])
            A = self._amplitudes(tip, atom_pos, orbitals, kappa)
            psi = A @ c_re
            acc = np.einsum("ps,s->p", psi * psi, w_all)
            if has_imag:
                psi = A @ c_im
                acc += np.einsum("ps,s->p", psi * psi, w_all)
            out[start:stop] = acc
        return out

    def scan(self, structure: Structure, settings: Optional[STMSettings] = None,
             progress=None) -> "ScanResult":
        from .scan import ScanResult

        settings = settings or STMSettings()
        if settings.mode not in MODES:
            raise ValueError(f"Unknown STM mode {settings.mode!r}; implemented: {MODES}")
        t0 = time.perf_counter()

        support = self.solver.supports(structure)
        if not support.ok:
            return ScanResult.unsupported_result(
                self.name, "; ".join(support.blocking), settings.describe(),
                suggested=["external:gpaw (Tersoff-Hamann from DFT)",
                           "external:quantum-espresso + Wannier90"])

        kappa, phi_eff = self.decay_constant(settings.bias_V)

        nx, ny = settings.resolution
        cell = structure.cell.matrix
        if settings.window_A is None:
            x0, y0 = 0.0, 0.0
            x1 = float(cell[0][0] + max(cell[1][0], 0.0))
            y1 = float(cell[1][1] + max(cell[0][1], 0.0))
            if x1 <= x0 or y1 <= y0:
                pxy = structure.positions
                x0, y0 = float(pxy[:, 0].min()), float(pxy[:, 1].min())
                x1, y1 = float(pxy[:, 0].max()), float(pxy[:, 1].max())
        else:
            x0, y0, x1, y1 = settings.window_A

        blocks, rep_pos, rep_orbitals, info = self.vacuum_states(
            structure, settings.bias_V, settings.broadening_eV,
            kgrid=settings.kgrid, lateral_images=settings.lateral_images,
            window_A=(x0, y0, x1, y1),
            electronic_depth_A=settings.electronic_depth_A)
        if blocks is None:
            return ScanResult.unsupported_result(
                self.name, "Electronic structure could not be computed for this system.",
                settings.describe(), suggested=["external:gpaw"])

        if settings.square_pixels and (x1 - x0) > 0 and (y1 - y0) > 0:
            pitch = (x1 - x0) / max(nx - 1, 1)
            ny = max(8, int(round((y1 - y0) / pitch)) + 1)
        xs = np.linspace(x0, x1, nx)
        ys = np.linspace(y0, y1, ny)
        gx, gy = np.meshgrid(xs, ys)
        if settings.scan_angle_deg:
            th = np.radians(settings.scan_angle_deg)
            cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
            rx = cx + (gx - cx) * np.cos(th) - (gy - cy) * np.sin(th)
            ry = cy + (gx - cx) * np.sin(th) + (gy - cy) * np.cos(th)
            gx, gy = rx, ry
        xy = np.column_stack([gx.ravel(), gy.ravel()])

        z_surface = float(structure.positions[:, 2].max())
        channels: Dict[str, np.ndarray] = {}
        residual = 0.0
        iterations = 0

        def current_at(zv):
            return self.current_map(rep_pos, rep_orbitals, blocks, xy, zv, kappa)

        if settings.mode == "constant-height":
            z_abs = z_surface + settings.height_A
            current = current_at(np.full(len(xy), z_abs))
            signal = current.reshape(ny, nx)
            channels["current"] = signal
            channels["topography"] = np.full_like(signal, z_abs)
            channels["ldos_vacuum"] = signal.copy()
            primary = "current"
            primary_unit = "arb. units"
        else:
            z1 = z_surface + settings.z_min_A
            z2 = z1 + 3.0
            i1 = current_at(np.full(len(xy), z1))
            i2 = current_at(np.full(len(xy), z2))
            floor = max(float(i1.max()), 1e-300) * 1e-25
            l1 = np.log(np.maximum(i1, floor))
            l2 = np.log(np.maximum(i2, floor))
            alpha = np.clip((l1 - l2) / (z2 - z1), 0.2, 20.0)

            i_set = float(np.exp(np.median(l1 - alpha * 1.5)))
            calibration = settings.setpoint_nA / max(i_set, 1e-300)
            log_set = np.log(max(i_set, floor))

            z = z1 + (l1 - log_set) / alpha
            z_lo, z_hi = z_surface + settings.z_min_A, z_surface + settings.z_max_A
            z = np.clip(z, z_lo, z_hi)
            cur = current_at(z)
            for iterations in range(1, settings.feedback_iterations + 1):
                err = np.log(np.maximum(cur, floor)) - log_set
                z_new = np.clip(z + settings.feedback_gain * err / alpha, z_lo, z_hi)
                cur_new = current_at(z_new)
                dz = z_new - z
                moved = np.abs(dz) > 1e-6
                if moved.any():
                    new_alpha = (np.log(np.maximum(cur, floor))
                                 - np.log(np.maximum(cur_new, floor)))[moved] / dz[moved]
                    alpha[moved] = np.clip(np.where(new_alpha > 0.2, new_alpha,
                                                    alpha[moved]), 0.2, 20.0)
                z, cur = z_new, cur_new
                interior = (z > z_lo + 1e-9) & (z < z_hi - 1e-9)
                rel = np.abs(cur / max(i_set, 1e-300) - 1.0)
                residual = float(rel[interior].max()) if interior.any() else float(rel.max())
                if residual < 2e-3:
                    break
            n_clipped = int((~((z > z_lo + 1e-9) & (z < z_hi - 1e-9))).sum())
            channels["topography"] = z.reshape(ny, nx)
            channels["current"] = (cur * calibration).reshape(ny, nx)
            channels["ldos_vacuum"] = current_at(
                np.full(len(xy), z_surface + settings.z_min_A + 1.5)).reshape(ny, nx)
            signal = channels["topography"]
            primary = "topography"
            primary_unit = "A"
            info["clipped_pixels"] = n_clipped

        clean = signal.copy()
        noisy, applied = apply_noise(clean, settings.noise)
        channels["raw_signal"] = clean
        channels[primary] = noisy

        elapsed = time.perf_counter() - t0
        prov = Provenance(
            model=self.name,
            fidelity=self.fidelity,
            origin=Origin.CALCULATED,
            approximations=[
                "Tersoff-Hamann: s-wave tip apex, low bias, tip DOS constant.",
                "Vacuum wavefunction tails approximated by atom-centred exponentials "
                f"with a single decay constant kappa = {kappa:.4f} 1/A from an "
                f"effective barrier of {phi_eff:.3f} eV.",
                "Current reported in arbitrary units; the constant-current setpoint is "
                "calibrated to the median computed signal, as an experimental feedback "
                "loop is. Absolute currents are not predicted.",
                "Sample electronic structure from a non-self-consistent tight-binding "
                "model: no tip-induced band bending and no response to the field.",
                f"Energy integration is a smoothly-windowed sum over "
                f"{info.get('n_states_in_window')} states sampled on a "
                f"{info.get('n_kpoints')}-point surface-Brillouin-zone grid.",
            ] + ([] if not info.get("electronic_truncated") else [
                f"The electronic structure was evaluated on the topmost "
                f"{info.get('electronic_depth_A'):.1f} A of the slab "
                f"({info.get('electronic_atoms')} atoms). Deeper atoms cannot reach "
                f"the vacuum region, but the truncation creates its own lower "
                f"surface whose states are excluded from the vacuum sum.",
            ]) + [
                ("Constant-current heights come from an iterative log-linear feedback "
                 f"solve; the residual |I/I_set - 1| after {iterations} iterations was "
                 f"{residual:.2e}." if settings.mode == "constant-current"
                 else "Constant-height mode: no feedback loop."),
            ],
            tolerances={"feedback_residual": residual,
                        "feedback_iterations": iterations,
                        "state_amplitude_threshold":
                            info.get("state_amplitude_threshold"),
                        "vacuum_depth_cutoff_A": DEPTH_CUTOFF_A},
            boundary_conditions=str(structure.cell.pbc),
            parameters={
                "settings": settings.describe(),
                "tip": self.tip.describe(),
                "sample_work_function_eV": self.sample_work_function_eV,
                "kappa_inv_A": kappa,
                "effective_barrier_eV": phi_eff,
                "electronic_states_in_window": info["n_states_in_window"],
                "atoms_in_vacuum_sum": info["n_atoms_in_vacuum_sum"],
                "replicated_atoms": info.get("n_replicated_atoms"),
                "electronic_atoms": info.get("electronic_atoms"),
                "electronic_orbitals": info.get("electronic_orbitals"),
                "electronic_from_cache": info.get("from_cache", False),
                "n_kpoints": info.get("n_kpoints"),
                "kpoints_frac": info.get("kpoints_frac"),
                "clipped_pixels": info.get("clipped_pixels", 0),
                "fermi_level_eV": info["fermi_level_eV"],
                "energy_window_eV": info["window_eV"],
            },
            references=[
                "J. Tersoff and D. R. Hamann, Phys. Rev. B 31 (1985) 805",
                "J. Bardeen, Phys. Rev. Lett. 6 (1961) 57",
                self.solver.model.reference,
            ],
            seed=settings.noise.seed,
            notes=("A bright feature is a maximum of the vacuum local density of "
                   "states, which need not coincide with an atomic nucleus."),
        )
        conv = Convergence(
            converged=(settings.mode == "constant-height") or residual < 1e-2,
            iterations=iterations,
            residual=residual,
            residual_metric="max |I/I_setpoint - 1|",
            tolerance=1e-3,
            message=("Constant-height scan: no feedback to converge."
                     if settings.mode == "constant-height"
                     else f"Feedback solve residual {residual:.2e}."),
        )
        return ScanResult(
            technique="STM",
            mode=settings.mode,
            channels=channels,
            primary_channel=primary,
            units={"topography": "A", "current": "nA", "raw_signal": primary_unit},
            extent_A=(x0, y0, x1, y1),
            resolution=(nx, ny),
            structure=structure,
            provenance=prov,
            convergence=conv,
            noise_record=scan_noise_provenance(settings.noise, applied),
            settings=settings.describe(),
            wall_time_s=elapsed,
            electronic=None,
        )

    def spectroscopy(self, structure: Structure, x: float, y: float, z: float,
                     bias_range_V: Tuple[float, float] = (-2.0, 2.0),
                     n_points: int = 201, broadening_eV: float = 0.05,
                     kgrid: Tuple[int, int] = (2, 2)) -> Result:
        """Simulated dI/dV point spectrum at a fixed tip position.

        Within Tersoff-Hamann, dI/dV at bias V is proportional to the sample
        LDOS at ``E_F + eV`` evaluated at the tip position.
        """
        support = self.solver.supports(structure)
        if not support.ok:
            return unsupported("dIdV", self.name, "; ".join(support.blocking),
                               suggested_models=["external:gpaw"], unit="arb. units")

        n_orb = len(self.solver.model.orbitals)
        cell = structure.cell.matrix
        zc = structure.positions[:, 2]
        near_idx = np.nonzero(zc >= zc.max() - DEPTH_CUTOFF_A)[0]
        shifts = ([(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1)]
                  if structure.cell.is_periodic else [(0, 0)])
        atom_pos = np.concatenate(
            [structure.positions[near_idx] + i * cell[0] + j * cell[1] for i, j in shifts])
        rep_atom = np.tile(near_idx, len(shifts))
        rep_shift = np.repeat(np.array([[i, j, 0] for i, j in shifts], dtype=float),
                              len(near_idx), axis=0)
        rep_orb_row = (np.repeat(rep_atom, n_orb) * n_orb
                       + np.tile(np.arange(n_orb), len(rep_atom)))
        rep_shift_orb = np.repeat(rep_shift, n_orb, axis=0)

        kappa, phi = self.decay_constant(0.0)
        A = self._amplitudes(np.array([[x, y, z]]), atom_pos,
                             self.solver.model.orbitals, kappa)[0]

        nkx, nky = (kgrid if structure.cell.is_periodic else (1, 1))
        kpts = [((ix + 0.5) / nkx - 0.5, (iy + 0.5) / nky - 0.5, 0.0)
                for ix in range(nkx) for iy in range(nky)]
        v = np.linspace(bias_range_V[0], bias_range_V[1], n_points)
        didv = np.zeros(n_points)
        e_f_list = []
        n_el = self.solver._n_electrons(structure)
        sigma = max(broadening_eV, 1e-6)
        for kf in kpts:
            H = (self.solver.build_hamiltonian(structure, k_frac=kf)
                 if any(abs(q) > 1e-12 for q in kf)
                 else self.solver.build_hamiltonian(structure))
            evals, evecs = robust_eigh(H)
            e_f, _ = self.solver.fermi_level(evals, n_el)
            e_f_list.append(e_f)
            c = evecs[rep_orb_row]
            if any(abs(q) > 1e-12 for q in kf):
                kcart = np.asarray(kf, dtype=float) @ structure.cell.reciprocal
                c = c * np.exp(1j * (rep_shift_orb @ cell) @ kcart)[:, None]
            weight = np.abs(A @ c) ** 2
            e = e_f + v
            g = np.exp(-0.5 * ((e[:, None] - evals[None, :]) / sigma) ** 2) / (
                sigma * np.sqrt(2 * np.pi))
            didv += 2.0 * (g * weight[None, :]).sum(axis=1) / len(kpts)

        e_f = float(np.mean(e_f_list))
        prov = Provenance(
            model=self.name + "/sts",
            fidelity=self.fidelity,
            origin=Origin.CALCULATED,
            approximations=[
                "dI/dV proportional to the tip-position LDOS (Tersoff-Hamann).",
                "Constant tip DOS and a bias-independent decay constant taken at V = 0; "
                "the real barrier narrows with bias, which raises the high-bias signal.",
                "No tip-induced band bending and no inelastic channels.",
                f"Surface Brillouin zone sampled on a {len(kpts)}-point grid.",
            ],
            tolerances={"gaussian_broadening_eV": broadening_eV},
            parameters={"tip_position_A": [x, y, z], "kappa_inv_A": kappa,
                        "fermi_level_eV": e_f, "n_kpoints": len(kpts)},
            references=["J. Tersoff and D. R. Hamann, Phys. Rev. B 31 (1985) 805"],
        )
        return Result("dIdV", {"bias_V": v, "dIdV": didv, "energy_eV": e_f + v},
                      "arb. units", prov,
                      extra={"fermi_level_eV": e_f,
                             "note": "Vertical scale is arbitrary; the shape is meaningful."})
