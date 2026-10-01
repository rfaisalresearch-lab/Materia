"""Physical validation of the shipped embedded-atom potentials.

Three kinds of evidence are kept apart, because they mean different things.

1. Agreement with independent implementations of the same potential file.
   ASE's EAM calculator reads the identical file with a different spline, and
   the NIST Interatomic Potentials Repository computed properties of the same
   NIST retabulation with LAMMPS (iprPy, static calculations at 0 K). This
   shows Materia evaluates the potential correctly. Reference values:
   https://www.ctcms.nist.gov/potentials/entry/2004--Zhou-X-W-Johnson-R-A-Wadley-H-N-G--<El>/
   2004--Zhou-X-W--<El>--LAMMPS--ipr2/calc.html, retrieved 2026-09-25.

2. Agreement with the parameter publication (Zhou, Johnson and Wadley,
   Phys. Rev. B 69 (2004) 144113). The article's property tables were not
   accessible when this suite was written, so no case claims agreement with
   them.

3. Comparison with experiment, using the sourced experimental values in the
   material library. The potential was fitted to properties like these, so
   agreement is expected and is not evidence of predictive accuracy. Bounds
   are loose and the reason is stated.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from materia.core_model.cell import Cell
from materia.core_model.structure import Structure
from materia.elements import periodic_table as pt
from materia.materials import load
from materia.physics import eam
from materia.solvers.classical import ClassicalSolver

pytestmark = pytest.mark.validation

EV_A3_TO_GPA = 160.21766208
EV_A2_TO_MJ_M2 = 16021.766208
FCC = np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0.0]])
BCC = np.array([[0, 0, 0], [0.5, 0.5, 0.5]])

NIST = {
    "Cu": {"structure": "fcc", "a_A": (3.6146974394, 3.6147910832, 3.6149788819),
           "E_coh_eV": 3.54, "C11": 169.989, "C12": 122.225, "C44": 75.927,
           "vacancy_eV": 1.279, "surfaces": {"111": 1501.45, "100": 1563.12},
           "material": "copper"},
    "Au": {"structure": "fcc", "a_A": (4.0800534236,), "E_coh_eV": 3.93,
           "C11": 186.378, "C12": 157.343, "C44": 42.071, "vacancy_eV": 1.001,
           "surfaces": {"111": 907.51, "100": 1016.05}, "material": "gold"},
    "W": {"structure": "bcc", "a_A": (3.1648494550,), "E_coh_eV": 8.76,
          "C11": 522.536, "C12": 204.219, "C44": 160.753, "vacancy_eV": 3.575,
          "surfaces": {"110": 2567.68, "100": 2983.46}, "material": "tungsten"},
}
METALS = sorted(NIST)


def crystal(element: str, structure: str, a: float, repeat: int = 1) -> Structure:
    basis = FCC if structure == "fcc" else BCC
    s = Structure([pt.element(element).number] * len(basis), basis * a, Cell.cubic(a))
    return s.repeat(repeat, repeat, repeat) if repeat > 1 else s


def strained(s: Structure, F: np.ndarray) -> Structure:
    out = s.copy()
    out.cell = Cell(s.cell.matrix @ F.T, s.cell.pbc)
    out.positions = s.positions @ F.T
    return out


@pytest.fixture(scope="module")
def equilibrium():
    """Lattice constant of the lowest-energy cubic structure of each metal."""
    out = {}
    for el in METALS:
        p = eam.load_shipped(f"{el}-Zhou04")
        ref = NIST[el]["a_A"][0]
        structure = NIST[el]["structure"]
        found = minimize_scalar(
            lambda a: p.energy(crystal(el, structure, a)) / len(crystal(el, structure, a)),
            bounds=(0.97 * ref, 1.03 * ref), method="bounded", options={"xatol": 1e-11})
        out[el] = (p, structure, float(found.x), -float(found.fun))
    return out


def elastic_constants(p, s: Structure, h: float = 1e-8):
    sigma = lambda F: p.evaluate(strained(s, F))["stress_eV_A3"]
    up, down = sigma(np.diag([1 + h, 1, 1])), sigma(np.diag([1 - h, 1, 1]))
    shear = np.array([[0, 0, 0], [0, 0, h / 2], [0, h / 2, 0]])
    c11 = (up[0, 0] - down[0, 0]) / (2 * h) * EV_A3_TO_GPA
    c12 = (up[1, 1] - down[1, 1]) / (2 * h) * EV_A3_TO_GPA
    c44 = (sigma(np.eye(3) + shear)[1, 2] - sigma(np.eye(3) - shear)[1, 2]) / (2 * h) * EV_A3_TO_GPA
    return c11, c12, c44


def reference_state(equilibrium, el):
    """The state NIST evaluated elastic constants and defects in."""
    p, structure, a_min, _ = equilibrium[el]
    a = NIST[el]["a_A"][0] if el == "Cu" else a_min
    return p, structure, a


@pytest.mark.parametrize("el", METALS)
def test_ground_state_structure(equilibrium, el):
    p, structure, a, e_coh = equilibrium[el]
    other = "bcc" if structure == "fcc" else "fcc"
    scale = 0.8 if other == "bcc" else 1.3
    alt = minimize_scalar(lambda x: p.energy(crystal(el, other, x)) / len(crystal(el, other, x)),
                          bounds=sorted((a * scale * 0.95, a * scale * 1.05)), method="bounded",
                          options={"xatol": 1e-9})
    assert -alt.fun < e_coh - 0.02


@pytest.mark.parametrize("el", ["Au", "W"])
def test_lattice_constant_matches_lammps(equilibrium, el):
    assert equilibrium[el][2] == pytest.approx(NIST[el]["a_A"][0], abs=5e-8)


def test_copper_energy_steps_at_the_cutoff_near_equilibrium(equilibrium):
    """Cu's equilibrium lies at the cutoff, and the tabulated functions do not vanish there.

    The fifth-neighbour shell of fcc Cu sits at sqrt(5/2) a, which reaches the
    5.7158 A cutoff at a_c = 3.614959 A. The energy steps up by about 1e-5 eV
    per atom as that shell leaves the cutoff. Below a_c the energy falls all
    the way to the step, with no stationary point; above it there is a genuine
    minimum at 3.6149789 A, which is the value NIST's LAMMPS dynamic relaxation
    finds (3.6149788819 A). NIST's static box relaxations stop on the lower
    branch at 3.61470 and 3.61479 A.
    """
    p, _, a, _ = equilibrium["Cu"]
    crossing = p.cutoff_A / np.sqrt(2.5)
    assert crossing == pytest.approx(3.614959, abs=1e-6)
    per_atom = lambda x: p.energy(crystal("Cu", "fcc", x)) / 4
    outer = minimize_scalar(per_atom, bounds=(crossing + 1e-9, crossing + 0.01),
                            method="bounded", options={"xatol": 1e-11})
    assert outer.x == pytest.approx(3.6149788819, abs=5e-8)
    step = per_atom(crossing + 1e-7) - per_atom(crossing - 1e-7)
    assert 5e-6 < step < 5e-5
    slope = (per_atom(crossing - 1e-5) - per_atom(crossing - 2e-5)) / 1e-5
    assert slope < 0
    assert per_atom(crossing - 1e-7) < outer.fun


@pytest.mark.parametrize("el", METALS)
def test_cohesive_energy_matches_lammps(equilibrium, el):
    assert equilibrium[el][3] == pytest.approx(NIST[el]["E_coh_eV"], abs=0.005)


@pytest.mark.parametrize("el", METALS)
def test_elastic_constants_match_lammps(equilibrium, el):
    p, structure, a = reference_state(equilibrium, el)
    c11, c12, c44 = elastic_constants(p, crystal(el, structure, a))
    assert c11 == pytest.approx(NIST[el]["C11"], abs=0.05)
    assert c12 == pytest.approx(NIST[el]["C12"], abs=0.05)
    assert c44 == pytest.approx(NIST[el]["C44"], abs=0.05)


@pytest.mark.parametrize("el", METALS)
def test_bulk_modulus_from_hydrostatic_strain(equilibrium, el):
    p, structure, a = reference_state(equilibrium, el)
    s = crystal(el, structure, a)
    h = 1e-8
    pressure = lambda F: -np.trace(p.evaluate(strained(s, F))["stress_eV_A3"]) / 3.0
    b = -(pressure(np.eye(3) * (1 + h)) - pressure(np.eye(3) * (1 - h))) / (6 * h) * EV_A3_TO_GPA
    expected = (NIST[el]["C11"] + 2 * NIST[el]["C12"]) / 3.0
    assert b == pytest.approx(expected, rel=2e-4)


def test_tungsten_bulk_modulus_from_the_energy_curve(equilibrium):
    """An energy-only route to B, independent of the stress code."""
    p, structure, a, _ = equilibrium["W"]
    volume = lambda x: x ** 3 / 2
    energy = lambda x: p.energy(crystal("W", "bcc", x)) / 2
    h = 1e-4 * a
    curvature = (energy(a + h) + energy(a - h) - 2 * energy(a)) / h ** 2
    b = curvature * a ** 2 / (9 * volume(a)) * EV_A3_TO_GPA
    assert b == pytest.approx((522.536 + 2 * 204.219) / 3, rel=2e-3)


@pytest.mark.parametrize("el", METALS)
def test_vacancy_formation_energy_matches_lammps(equilibrium, el):
    p, structure, a = reference_state(equilibrium, el)
    bulk = crystal(el, structure, a, 4 if structure == "fcc" else 5)
    e_bulk = p.energy(bulk)
    vacancy = bulk.copy()
    vacancy.remove_atoms([int(vacancy.ids[0])])
    out = ClassicalSolver(p).relax(vacancy, fmax_eV_A=1e-5, max_steps=5000)
    assert out.convergence.converged
    n = len(bulk)
    e_f = out.results["energy"].value - (n - 1) / n * e_bulk
    assert e_f == pytest.approx(NIST[el]["vacancy_eV"], abs=0.005)


@pytest.mark.parametrize("el", METALS)
def test_surface_energies_match_lammps(equilibrium, el):
    ase_build = pytest.importorskip("ase.build")
    p, structure, a = reference_state(equilibrium, el)
    e_atom = p.energy(crystal(el, structure, a)) / (4 if structure == "fcc" else 2)
    builders = {"fcc": {"111": ase_build.fcc111, "100": ase_build.fcc100},
                "bcc": {"110": ase_build.bcc110, "100": ase_build.bcc100}}[structure]
    for face, reference in NIST[el]["surfaces"].items():
        atoms = builders[face](el, size=(1, 1, 14), a=a, vacuum=10.0)
        slab = Structure(atoms.numbers, atoms.positions,
                         Cell(np.array(atoms.cell), (True, True, False)))
        out = ClassicalSolver(p).relax(slab, fmax_eV_A=1e-5, max_steps=5000)
        assert out.convergence.converged
        area = np.linalg.norm(np.cross(atoms.cell[0], atoms.cell[1]))
        gamma = (out.results["energy"].value - len(slab) * e_atom) / (2 * area) * EV_A2_TO_MJ_M2
        assert gamma == pytest.approx(reference, rel=5e-4), face


@pytest.mark.parametrize("el", METALS)
def test_energy_and_forces_match_ase_on_the_same_file(equilibrium, el):
    """ASE reads the identical file and interpolates it with its own spline."""
    eam_calc = pytest.importorskip("ase.calculators.eam")
    ase_build = pytest.importorskip("ase.build")
    p, structure, a, _ = equilibrium[el]
    rng = np.random.default_rng(11)
    atoms = ase_build.bulk(el, structure, a=a, cubic=True).repeat((3, 3, 3))
    atoms.positions = atoms.positions + rng.normal(0.0, 0.08, atoms.positions.shape)
    atoms.calc = eam_calc.EAM(potential=p.identity.path)
    s = Structure(atoms.numbers, atoms.positions, Cell(np.array(atoms.cell)))
    out = p.evaluate(s)
    assert out["energy_eV"] == pytest.approx(atoms.get_potential_energy(), abs=1e-8 * len(s))
    np.testing.assert_allclose(out["forces_eV_A"], atoms.get_forces(), atol=5e-7)


@pytest.mark.parametrize("el", METALS)
def test_comparison_with_experiment(equilibrium, el):
    """Loose bounds: 0 K potential values against room-temperature measurements.

    Thermal expansion alone moves the lattice constant by a few tenths of a
    per cent between 0 K and room temperature, and elastic constants soften
    by a few per cent, so tighter bounds would be testing the wrong thing.
    """
    material = load(NIST[el]["material"])
    p, structure, a, e_coh = equilibrium[el]
    assert a == pytest.approx(material.lattice.parameters()[0], rel=5e-3)
    assert e_coh == pytest.approx(float(material.property("cohesive_energy").value), rel=0.05)
    ps, st, ar = reference_state(equilibrium, el)
    c11, c12, c44 = elastic_constants(ps, crystal(el, st, ar))
    for name, value in (("C11", c11), ("C12", c12), ("C44", c44)):
        measured = float(material.property(f"elastic_{name}").value)
        assert value == pytest.approx(measured, rel=0.05), name


def test_nve_dynamics_conserves_energy():
    """Velocity Verlet on the EAM surface: total energy drift at 1 fs is small."""
    p = eam.load_shipped("Cu-Zhou04")
    s = crystal("Cu", "fcc", 3.615, 3)
    out = ClassicalSolver(p).dynamics(s, steps=400, dt_fs=1.0, temperature_K=600.0,
                                      thermostat="none", seed=3)
    total = np.array(out.results["trajectory"].value["total_eV"])
    kinetic = np.array(out.results["trajectory"].value["kinetic_eV"])
    assert np.ptp(total) < 0.02 * kinetic.mean()
