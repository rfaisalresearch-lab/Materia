"""Conversion between LAMMPS ``metal`` units and Materia's internal units.

Every value that crosses the boundary goes through one of these functions, so
the factor applied is written down once and tested once.

======================  ======================  ======================
quantity                LAMMPS ``metal``        Materia
======================  ======================  ======================
length                  angstrom                angstrom
energy                  eV                      eV
time                    picosecond              femtosecond
velocity                angstrom / ps           angstrom / fs
force                   eV / angstrom           eV / angstrom
mass                    g / mol                 u
temperature             K                       K
pressure                bar                     eV / angstrom^3 (stress)
======================  ======================  ======================

Time and velocity factors are exact powers of ten.  The pressure and kinetic
energy factors are the constants LAMMPS itself applies in ``metal`` units
(``force->nktv2p`` and ``force->mvv2e`` in its ``update.cpp``), so converting
back inverts exactly what LAMMPS computed internally.  They differ from the
CODATA 2018 values Materia uses elsewhere by less than one part in 10^7, and
:data:`CODATA_EV_A3_IN_BAR` records the exact value for comparison.

Stress sign: LAMMPS reports the pressure tensor ``P``, positive under
compression.  Materia and ASE report the stress ``sigma = -P``.
"""

from __future__ import annotations

import numpy as np

from ...core_model.units import ATOMIC_MASS_UNIT_KG, ELEMENTARY_CHARGE_C

UNIT_STYLE = "metal"

FS_PER_PS = 1000.0

LAMMPS_NKTV2P_METAL = 1.6021765e6

LAMMPS_MVV2E_METAL = 1.0364269e-4

LAMMPS_BOLTZMANN_METAL = 8.617343e-5

CODATA_EV_A3_IN_BAR = ELEMENTARY_CHARGE_C / 1.0e-30 / 1.0e5

CODATA_MVV2E_METAL = ATOMIC_MASS_UNIT_KG * 100.0 ** 2 / ELEMENTARY_CHARGE_C


def fs_to_ps(value_fs):
    return np.asarray(value_fs, dtype=float) / FS_PER_PS if np.ndim(value_fs) else \
        float(value_fs) / FS_PER_PS


def ps_to_fs(value_ps):
    return np.asarray(value_ps, dtype=float) * FS_PER_PS if np.ndim(value_ps) else \
        float(value_ps) * FS_PER_PS


def velocity_to_lammps(value_A_fs) -> np.ndarray:
    """Angstrom per femtosecond to angstrom per picosecond."""
    return np.asarray(value_A_fs, dtype=float) * FS_PER_PS


def velocity_from_lammps(value_A_ps) -> np.ndarray:
    """Angstrom per picosecond to angstrom per femtosecond."""
    return np.asarray(value_A_ps, dtype=float) / FS_PER_PS


def pressure_bar_to_stress_eV_A3(pressure_bar) -> np.ndarray:
    """A LAMMPS pressure tensor in bar to a stress tensor in eV/A^3, ``sigma = -P``."""
    return -np.asarray(pressure_bar, dtype=float) / LAMMPS_NKTV2P_METAL


def stress_eV_A3_to_pressure_bar(stress_eV_A3) -> np.ndarray:
    return -np.asarray(stress_eV_A3, dtype=float) * LAMMPS_NKTV2P_METAL


def pressure_voigt_to_tensor(pxx: float, pyy: float, pzz: float,
                             pxy: float, pxz: float, pyz: float) -> np.ndarray:
    return np.array([[pxx, pxy, pxz], [pxy, pyy, pyz], [pxz, pyz, pzz]], dtype=float)


def kinetic_energy_eV(masses_g_mol, velocities_A_ps) -> float:
    """Kinetic energy exactly as LAMMPS computes it in ``metal`` units."""
    m = np.asarray(masses_g_mol, dtype=float)
    v = np.asarray(velocities_A_ps, dtype=float).reshape(len(m), 3)
    return float(0.5 * LAMMPS_MVV2E_METAL * np.sum(m[:, None] * v * v))
