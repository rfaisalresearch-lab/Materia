"""Embedded-atom potentials: implementation tests.

Parser, identity, force and stress consistency, invariances, determinism,
caching, selection and refusal. Agreement with published reference
calculations is physical validation and lives in
``tests/validation/test_eam_validation.py``.
"""

from __future__ import annotations

import hashlib
import json
import shutil

import numpy as np
import pytest

from materia.core_model.cell import Cell
from materia.core_model.structure import Structure
from materia.elements import periodic_table as pt
from materia.physics import eam
from materia.physics.neighbors import COLLISION_TOLERANCE_A
from materia.physics.potentials import OverlappingAtoms, UnsupportedSystem
from materia.solvers import registry
from materia.solvers.classical import ClassicalSolver

FCC = np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0.0]])
BCC = np.array([[0, 0, 0], [0.5, 0.5, 0.5]])
SHIPPED = {"Cu": ("Cu-Zhou04", FCC, 3.615), "Au": ("Au-Zhou04", FCC, 4.08),
           "W": ("W-Zhou04", BCC, 3.1648)}


def crystal(el, basis, a, repeat=3, jitter=0.0, seed=4, pbc=(True, True, True)):
    s = Structure([pt.element(el).number] * len(basis), basis * a, Cell.cubic(a, pbc))
    s = s.repeat(repeat, repeat, repeat) if repeat > 1 else s
    if jitter:
        s.positions = s.positions + np.random.default_rng(seed).normal(0, jitter, s.positions.shape)
    return s


def moved(s, i, axis, h):
    t = s.copy()
    p = t.positions.copy()
    p[i, axis] += h
    t.positions = p
    return t


def synthetic_setfl(elements=("Cu", "Ni"), n=600, cutoff=5.0) -> str:
    """A smooth two-element setfl file for testing the multi-species path.

    The functions are invented for the test and describe no real material.
    """
    dr = cutoff / (n - 1)
    drho = 60.0 / (n - 1)
    r = np.arange(n) * dr
    rho = np.arange(n) * drho
    taper = np.where(r < cutoff, (1 - r / cutoff) ** 4, 0.0)
    lines = ["synthetic test file", "not a physical potential", "materia unit test",
             f"{len(elements)} " + " ".join(elements),
             f"{n} {drho!r} {n} {dr!r} {cutoff!r}"]
    body = []
    for k, el in enumerate(elements):
        z = pt.element(el).number
        lines.append(f"{z} {pt.element(el).standard_atomic_weight} 3.6 fcc")
        F = -(1.0 + 0.3 * k) * np.sqrt(rho) + 0.01 * rho
        f = (2.0 + k) * np.exp(-1.0 * (r - 2.5)) * taper
        lines.extend(" ".join(f"{v:.16e}" for v in F[i:i + 5]) for i in range(0, n, 5))
        lines.extend(" ".join(f"{v:.16e}" for v in f[i:i + 5]) for i in range(0, n, 5))
    for i in range(len(elements)):
        for j in range(i + 1):
            phi = (1.5 + 0.2 * (i + j)) * np.exp(-2.0 * (r - 2.0)) * taper
            lines.extend(" ".join(f"{v:.16e}" for v in (r * phi)[k:k + 5]) for k in range(0, n, 5))
    return "\n".join(lines + body) + "\n"


@pytest.fixture
def alloy_file(tmp_path):
    path = tmp_path / "CuNi_test.eam.alloy"
    path.write_text(synthetic_setfl())
    return path


def test_manifest_checksums_match_the_shipped_files():
    for entry in eam.catalog():
        data = (eam.POTENTIAL_DIR / entry["file"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == entry["sha256"]
        assert entry["license"] == "Public domain"
        assert entry["license_text"] == "This work is dedicated to the public domain."
        assert any("144113" in c for c in entry["citations"])


def test_shipped_potentials_are_registered_solvers():
    for pid in eam.shipped_ids():
        solver = registry.create(f"eam/{pid}")
        assert solver.name == f"classical/eam/{pid}"


def test_header_is_parsed():
    p = eam.load_shipped("W-Zhou04")
    h = p.table.header()
    assert h["elements"] == ["W"] and h["atomic_numbers"] == [74]
    assert h["nrho"] == 2000 and h["nr"] == 2000
    assert p.cutoff_A == pytest.approx(6.128704555450525)


def test_altered_shipped_file_is_refused(tmp_path, monkeypatch):
    for name in ("manifest.json", "Cu_Zhou04.eam.alloy"):
        shutil.copy(eam.POTENTIAL_DIR / name, tmp_path / name)
    data = (tmp_path / "Cu_Zhou04.eam.alloy").read_text().replace("0.0000000000000000E+00",
                                                                   "1.0000000000000000E-12", 1)
    (tmp_path / "Cu_Zhou04.eam.alloy").write_text(data)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["potentials"] = [e for e in manifest["potentials"] if e["id"] == "Cu-Zhou04"]
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(eam, "POTENTIAL_DIR", tmp_path)
    monkeypatch.setattr(eam, "MANIFEST_PATH", tmp_path / "manifest.json")
    with pytest.raises(eam.EAMFileError, match="checksum"):
        eam.load_shipped("Cu-Zhou04")


@pytest.mark.parametrize("damage, message", [
    (lambda t: "\n".join(t.splitlines()[:-40]) + "\n", "truncated"),
    (lambda t: t + "1.0 2.0\n", "after the last pair table"),
    (lambda t: t.replace("\n2 Cu Ni\n", "\n3 Cu Ni\n", 1), "declares 3"),
    (lambda t: t.replace("\n2 Cu Ni\n", "\n2 Cu Xx\n", 1), "unknown element"),
    (lambda t: t.replace("\n2 Cu Cu\n", "\n2 Cu Cu\n", 1).replace("\n2 Cu Ni\n", "\n2 Cu Cu\n", 1),
     "twice"),
    (lambda t: t.replace("\n29 ", "\n30 ", 1), "atomic number 30"),
    (lambda t: t.replace(" 5.0\n", " 9.0\n", 1), "beyond the radial table"),
    (lambda t: _replace_value(t, "nan"), "not finite"),
    (lambda t: _replace_value(t, "abc"), "non-numeric"),
])
def test_malformed_files_are_refused(tmp_path, damage, message):
    path = tmp_path / "bad.eam.alloy"
    path.write_text(damage(synthetic_setfl()))
    with pytest.raises(eam.EAMFileError, match=message):
        eam.load_file(str(path))


def _replace_value(text: str, token: str) -> str:
    lines = text.splitlines()
    parts = lines[20].split()
    parts[1] = token
    lines[20] = " ".join(parts)
    return "\n".join(lines) + "\n"


def test_missing_file_is_refused(tmp_path):
    with pytest.raises(eam.EAMFileError, match="No such"):
        eam.load_file(str(tmp_path / "absent.eam.alloy"))


def test_user_file_identity_records_checksum_and_unknown_licence(alloy_file):
    p = eam.load_file(str(alloy_file))
    assert p.identity.sha256 == hashlib.sha256(alloy_file.read_bytes()).hexdigest()
    assert p.identity.shipped is False
    assert "unknown" in p.identity.license
    assert p.elements == ("Cu", "Ni")


def test_unsupported_element_is_refused():
    p = eam.load_shipped("Au-Zhou04")
    s = crystal("Au", FCC, 4.08, repeat=1)
    s.substitute(int(s.ids[0]), "Cu")
    ok, why = p.supports(s)
    assert not ok and "Cu" in why
    with pytest.raises(UnsupportedSystem):
        p.energy(s)


@pytest.mark.parametrize("el", sorted(SHIPPED))
def test_forces_are_the_negative_energy_gradient(el):
    pid, basis, a = SHIPPED[el]
    p = eam.load_shipped(pid)
    s = crystal(el, basis, a, repeat=2, jitter=0.1)
    f = p.evaluate(s)["forces_eV_A"]
    h = 1e-5
    for i in (0, 3, len(s) - 1):
        for axis in range(3):
            fd = -(eam.load_shipped(pid).energy(moved(s, i, axis, h))
                   - eam.load_shipped(pid).energy(moved(s, i, axis, -h))) / (2 * h)
            assert fd == pytest.approx(f[i, axis], abs=1e-7)


def test_alloy_forces_are_the_negative_energy_gradient(alloy_file):
    p = eam.load_file(str(alloy_file))
    s = crystal("Cu", FCC, 3.5, repeat=2, jitter=0.1)
    for k in range(0, len(s), 3):
        s.substitute(int(s.ids[k]), "Ni")
    f = p.evaluate(s)["forces_eV_A"]
    h = 1e-5
    for i in range(0, len(s), 5):
        for axis in range(3):
            fd = -(eam.load_file(str(alloy_file)).energy(moved(s, i, axis, h))
                   - eam.load_file(str(alloy_file)).energy(moved(s, i, axis, -h))) / (2 * h)
            assert fd == pytest.approx(f[i, axis], abs=1e-7)


def test_alloy_matches_ase_on_the_same_file(alloy_file):
    eam_calc = pytest.importorskip("ase.calculators.eam")
    ase = pytest.importorskip("ase")
    s = crystal("Cu", FCC, 3.5, repeat=2, jitter=0.1)
    for k in range(0, len(s), 3):
        s.substitute(int(s.ids[k]), "Ni")
    atoms = ase.Atoms(numbers=s.numbers, positions=s.positions, cell=s.cell.matrix, pbc=True)
    atoms.calc = eam_calc.EAM(potential=str(alloy_file))
    out = eam.load_file(str(alloy_file)).evaluate(s)
    assert out["energy_eV"] == pytest.approx(atoms.get_potential_energy(), abs=1e-6)
    np.testing.assert_allclose(out["forces_eV_A"], atoms.get_forces(), atol=1e-5)


def test_stress_is_the_strain_derivative_of_the_energy():
    p = eam.load_shipped("W-Zhou04")
    s = crystal("W", BCC, 3.10, repeat=2, jitter=0.03)
    sigma = p.evaluate(s)["stress_eV_A3"]
    h = 1e-6
    for a, b in ((0, 0), (1, 1), (1, 2), (0, 2)):
        eps = np.zeros((3, 3))
        eps[a, b] = eps[b, a] = h / (1 if a == b else 2)

        def energy(sign):
            F = np.eye(3) + sign * eps
            t = s.copy()
            t.cell = Cell(s.cell.matrix @ F.T)
            t.positions = s.positions @ F.T
            return eam.load_shipped("W-Zhou04").energy(t)

        derivative = (energy(1) - energy(-1)) / (2 * h * s.cell.volume)
        assert sigma[a, b] == pytest.approx(derivative, rel=1e-5, abs=1e-8)


def test_stress_is_only_defined_for_fully_periodic_cells():
    p = eam.load_shipped("Cu-Zhou04")
    slab = crystal("Cu", FCC, 3.615, repeat=2, pbc=(True, True, False))
    assert p.evaluate(slab)["stress_eV_A3"] is None


@pytest.mark.parametrize("el", sorted(SHIPPED))
def test_perfect_crystal_has_zero_net_and_atomic_forces(el):
    pid, basis, a = SHIPPED[el]
    f = eam.load_shipped(pid).evaluate(crystal(el, basis, a))["forces_eV_A"]
    assert np.abs(f).max() < 1e-10


def test_rigid_translation_and_periodic_images_change_nothing():
    p = eam.load_shipped("Cu-Zhou04")
    s = crystal("Cu", FCC, 3.615, jitter=0.08)
    ref = p.evaluate(s)
    t = s.copy()
    t.translate([0.37, -1.21, 2.05])
    u = s.copy()
    pos = u.positions.copy()
    pos[5] += s.cell.matrix[0] - 2 * s.cell.matrix[2]
    u.positions = pos
    for other in (t, u):
        out = eam.load_shipped("Cu-Zhou04").evaluate(other)
        assert out["energy_eV"] == pytest.approx(ref["energy_eV"], abs=1e-9)
        np.testing.assert_allclose(out["forces_eV_A"], ref["forces_eV_A"], atol=1e-9)


def test_primitive_and_conventional_cells_give_the_same_energy_per_atom():
    p = eam.load_shipped("Au-Zhou04")
    a = 4.07
    primitive = Structure([79], [[0, 0, 0]],
                          Cell(np.array([[0, a / 2, a / 2], [a / 2, 0, a / 2], [a / 2, a / 2, 0]])))
    conventional = crystal("Au", FCC, a, repeat=3)
    assert p.energy(primitive) == pytest.approx(p.energy(conventional) / len(conventional),
                                                abs=1e-10)


def test_repeated_runs_are_bitwise_identical():
    s = crystal("W", BCC, 3.16, jitter=0.05)
    a = eam.load_shipped("W-Zhou04").evaluate(s)
    b = eam.load_shipped("W-Zhou04").evaluate(s)
    assert a["energy_eV"] == b["energy_eV"]
    assert np.array_equal(a["forces_eV_A"], b["forces_eV_A"])


def test_cached_neighbour_list_matches_a_fresh_one():
    p = eam.load_shipped("Cu-Zhou04")
    s = crystal("Cu", FCC, 3.615, jitter=0.05)
    p.evaluate(s)
    shifted = s.copy()
    shifted.positions = s.positions + np.random.default_rng(8).normal(0, 0.05, s.positions.shape)
    cached = p.evaluate(shifted)
    fresh = eam.load_shipped("Cu-Zhou04").evaluate(shifted)
    assert p.last_evaluation["neighbour_list_builds"] == 1
    assert cached["energy_eV"] == pytest.approx(fresh["energy_eV"], abs=1e-10)
    np.testing.assert_allclose(cached["forces_eV_A"], fresh["forces_eV_A"], atol=1e-10)
    far = s.copy()
    far.positions = s.positions + 0.4
    far.positions[0] += 0.4
    p.evaluate(far)
    assert p.last_evaluation["neighbour_list_builds"] == 2


def test_open_and_slab_boundaries():
    p = eam.load_shipped("Au-Zhou04")
    cluster = crystal("Au", FCC, 4.08, repeat=2)
    cluster.cell = Cell.none()
    slab = crystal("Au", FCC, 4.08, repeat=2, pbc=(True, True, False))
    bulk = crystal("Au", FCC, 4.08, repeat=2)
    e_cluster, e_slab, e_bulk = (p.energy(x) for x in (cluster, slab, bulk))
    assert e_bulk < e_slab < e_cluster < 0
    assert np.abs(p.evaluate(cluster)["forces_eV_A"].sum(axis=0)).max() < 1e-10


def test_density_beyond_the_table_is_extrapolated_and_reported():
    p = eam.load_shipped("W-Zhou04")
    out = p.evaluate(crystal("W", BCC, 1.9, repeat=2))
    assert out["rho_extrapolated"] == 16
    assert np.isfinite(out["energy_eV"])


def test_empty_structure():
    out = eam.load_shipped("Cu-Zhou04").evaluate(Structure([], np.zeros((0, 3)), Cell.cubic(3.6)))
    assert out["energy_eV"] == 0.0


def test_shipped_selection_is_stable_and_refuses_mixtures():
    assert eam.select_shipped(["Cu"]) == ("Cu-Zhou04", "")
    assert eam.select_shipped(["Au"])[0] == "Au-Zhou04"
    assert eam.select_shipped(["W"])[0] == "W-Zhou04"
    chosen, why = eam.select_shipped(["Cu", "Au"])
    assert chosen is None and "does not combine files" in why
    chosen, why = eam.select_shipped(["Si"])
    assert chosen is None and "Si" in why


def test_recommended_model_for_the_three_metals_is_eam():
    from materia.materials import load
    from materia.python_api.api import Lab

    lab = Lab()
    for material, pid in (("copper", "Cu-Zhou04"), ("gold", "Au-Zhou04"),
                          ("tungsten", "W-Zhou04")):
        definition = load(material)
        assert definition.recommended_models["relax"] == f"eam/{pid}"
        assert registry.resolve_recommended(definition, "relax") == f"eam/{pid}"
        handle = lab.materials.load(material).bulk(activate=False)
        solver = lab._resolve_solver("recommended", handle.structure, definition, "relax")
        assert solver.name == f"classical/eam/{pid}"


def test_lennard_jones_stays_available_and_labelled_approximate():
    from materia.materials import load

    solver = registry.create("lennard-jones", material=load("copper"))
    text = " ".join(solver.potential.describe()["approximations"])
    assert "no many-body metallic screening" in text


def test_provenance_records_identity_and_cutoff_residuals():
    solver = ClassicalSolver(eam.load_shipped("Cu-Zhou04"))
    out = solver.single_point(crystal("Cu", FCC, 3.615, repeat=2))
    prov = out["energy"].provenance
    identity = prov.parameters["potential"]
    assert identity["sha256"] == eam.catalog()[[e["id"] for e in eam.catalog()].index(
        "Cu-Zhou04")]["sha256"]
    assert identity["license"] == "Public domain"
    assert prov.fidelity.value == "tier1-classical"
    assert abs(prov.parameters["cutoff_residuals"]["f_Cu"]) < 1e-5
    assert any("144113" in r for r in prov.references)


def pair(distance, cell=None):
    return Structure([29, 29], [[0.0, 0.0, 0.0], [distance, 0.0, 0.0]], cell or Cell.none())


@pytest.mark.parametrize("distance", [0.0, 1e-10, 1e-8, 0.5 * COLLISION_TOLERANCE_A,
                                      COLLISION_TOLERANCE_A])
def test_coincident_isolated_atoms_are_refused(distance):
    p = eam.load_shipped("Cu-Zhou04")
    ok, why = p.supports(pair(distance))
    assert not ok and "collision tolerance" in why and "#1 (Cu) and #2 (Cu)" in why
    with pytest.raises(OverlappingAtoms) as caught:
        p.evaluate(pair(distance))
    assert caught.value.coincidence.distance == pytest.approx(distance, abs=1e-15)


def test_close_but_distinct_atoms_have_finite_repulsion():
    p = eam.load_shipped("Cu-Zhou04")
    for distance in (2 * COLLISION_TOLERANCE_A, 0.5, 1.0):
        out = p.evaluate(pair(distance))
        assert out["n_pairs"] == 2
        assert np.isfinite(out["energy_eV"]) and out["energy_eV"] > 0
    assert eam.load_shipped("Cu-Zhou04").evaluate(pair(0.5))["energy_eV"] == pytest.approx(
        251.8181428211417, rel=1e-12)


def test_periodic_image_overlap_is_refused():
    p = eam.load_shipped("Cu-Zhou04")
    s = pair(3.615, Cell.cubic(3.615))
    ok, why = p.supports(s)
    assert not ok and "periodic image" in why and "(-1, 0, 0)" in why
    with pytest.raises(OverlappingAtoms):
        p.evaluate(s)
    lone = Structure([29], [[0, 0, 0]], Cell.orthorhombic(3.0, 3.0, COLLISION_TOLERANCE_A / 2))
    with pytest.raises(OverlappingAtoms, match="its own periodic image"):
        eam.load_shipped("Cu-Zhou04").evaluate(lone)


def test_true_self_pair_is_excluded_by_index_and_image_only():
    a = 4.08
    primitive = Structure([79], [[0, 0, 0]],
                          Cell(np.array([[0, a / 2, a / 2], [a / 2, 0, a / 2], [a / 2, a / 2, 0]])))
    i, j, shift = eam.pair_search(primitive.positions, primitive.cell, 3.0)
    assert len(i) == 12 and np.all(i == 0) and np.all(j == 0)
    assert np.all(np.linalg.norm(shift, axis=1) > 2.8)
    out = eam.load_shipped("Au-Zhou04").evaluate(primitive)
    assert np.isfinite(out["energy_eV"]) and np.abs(out["forces_eV_A"]).max() < 1e-10


def test_reused_neighbour_list_still_detects_coincidence():
    p = eam.load_shipped("Cu-Zhou04")
    p.evaluate(pair(0.1))
    assert p.last_evaluation["neighbour_list_builds"] == 1
    with pytest.raises(OverlappingAtoms):
        p.evaluate(pair(0.0))
    assert p._cache.builds == 1


def test_a_refused_geometry_never_leaves_a_cached_list_and_recovers():
    p = eam.load_shipped("Cu-Zhou04")
    with pytest.raises(OverlappingAtoms):
        p.evaluate(pair(0.0))
    assert p._cache.key is None and p._cache.builds == 0
    fixed = p.evaluate(pair(2.5))
    reference = eam.load_shipped("Cu-Zhou04").evaluate(pair(2.5))
    assert fixed["energy_eV"] == reference["energy_eV"]
    np.testing.assert_array_equal(fixed["forces_eV_A"], reference["forces_eV_A"])


def test_classical_solver_refuses_every_task_on_coincident_atoms():
    solver = ClassicalSolver(eam.load_shipped("Cu-Zhou04"))
    s = crystal("Cu", FCC, 3.615, repeat=2)
    positions = s.positions.copy()
    positions[1] = positions[0]
    s.positions = positions
    before = s.positions.copy()
    single = solver.single_point(s)
    assert not single["energy"].supported and single["energy"].value is None
    assert not solver.relax(s)["relaxed_structure"].supported
    assert not solver.dynamics(s, steps=5)["trajectory"].supported
    np.testing.assert_array_equal(s.positions, before)
