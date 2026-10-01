"""Crystal, surface and defect construction."""

from __future__ import annotations

import math

import numpy as np
import pytest

from materia.materials import load
from materia.physics.neighbors import neighbor_list
from materia.structure_builder import bulk, make_surface
from materia.structure_builder.defects import (
    add_adatom,
    add_dopant,
    add_interstitial,
    create_vacancy,
    nearest_site,
    substitute_atom,
)
from materia.structure_builder.lattice import apply_strain, lattice_planes
from materia.structure_builder.primitive import primitive_basis, primitive_structure
from materia.structure_builder.surface import _broken_bonds, stable_termination_shift


def test_silicon_bulk_geometry():
    cell = bulk(load("silicon"))
    assert len(cell) == 8
    assert cell.cell.lengths == pytest.approx([5.4310205] * 3)
    nl = neighbor_list(cell.positions, cell.cell, 2.6)
    assert set(nl.counts()) == {4}
    assert float(nl.d.min()) == pytest.approx(5.4310205 * math.sqrt(3) / 4, rel=1e-6)


def test_fcc_metal_coordination_is_twelve():
    gold = bulk(load("gold"))
    nl = neighbor_list(gold.positions, gold.cell, 3.2)
    assert set(nl.counts()) == {12}
    assert float(nl.d.min()) == pytest.approx(4.0782 / math.sqrt(2), rel=1e-6)


def test_bcc_metal_coordination_is_eight():
    tungsten = bulk(load("tungsten"))
    nl = neighbor_list(tungsten.positions, tungsten.cell, 2.9)
    assert set(nl.counts()) == {8}


@pytest.mark.parametrize(
    "material,multiplicity",
    [("silicon", 4), ("gold", 4), ("tungsten", 2), ("sapphire", 3),
     ("graphene", 1), ("silicon_dioxide", 1), ("silicon_carbide_4h", 1)],
)
def test_primitive_lattice_detection(material, multiplicity):
    cell = bulk(load(material))
    _, detected = primitive_basis(cell)
    assert detected == multiplicity
    primitive = primitive_structure(cell)
    assert len(primitive) * multiplicity == len(cell)
    assert primitive.cell.volume * multiplicity == pytest.approx(cell.cell.volume, rel=1e-5)


@pytest.mark.parametrize(
    "miller,in_plane,angle",
    [((1, 1, 1), 5.4310205 / math.sqrt(2), 60.0),
     ((1, 0, 0), 5.4310205 / math.sqrt(2), 90.0)],
)
def test_silicon_surface_cells(miller, in_plane, angle):
    slab = make_surface(load("silicon"), miller, size=(1, 1, 2), vacuum_A=10.0)
    info = slab.info["surface"]
    assert info["in_plane_a_A"] == pytest.approx(in_plane, rel=1e-6)
    assert info["in_plane_b_A"] == pytest.approx(in_plane, rel=1e-6)
    assert info["in_plane_angle_deg"] == pytest.approx(angle, abs=1e-6)


def test_interplanar_spacings_match_analytic_values():
    a = 5.4310205
    assert lattice_planes(load("silicon"), (1, 0, 0)) == pytest.approx(a, rel=1e-9)
    assert lattice_planes(load("silicon"), (1, 1, 0)) == pytest.approx(a / math.sqrt(2), rel=1e-9)
    assert lattice_planes(load("silicon"), (1, 1, 1)) == pytest.approx(a / math.sqrt(3), rel=1e-9)


def test_surface_real_interplanar_spacing_includes_centring():
    slab = make_surface(load("silicon"), (1, 0, 0), size=(1, 1, 2), vacuum_A=10.0)
    info = slab.info["surface"]
    assert info["conventional_d_hkl_A"] == pytest.approx(5.4310205, rel=1e-6)
    assert info["interplanar_spacing_A"] == pytest.approx(5.4310205 / 2, rel=1e-6)


def test_surface_cell_areas_match_analytic_values():
    a = 5.4310205

    def area(slab):
        m = slab.cell.matrix
        return float(np.linalg.norm(np.cross(m[0], m[1])))

    assert area(make_surface(load("silicon"), (1, 1, 1), size=(1, 1, 2))) == \
        pytest.approx(a * a * math.sqrt(3) / 4, rel=1e-6)
    assert area(make_surface(load("silicon"), (1, 0, 0), size=(1, 1, 2))) == \
        pytest.approx(a * a / 2, rel=1e-6)
    assert area(make_surface(load("silicon"), (1, 1, 0), size=(1, 1, 2))) == \
        pytest.approx(a * a / math.sqrt(2), rel=1e-6)


def test_slab_is_two_dimensionally_periodic_with_vacuum():
    slab = make_surface(load("silicon"), (1, 1, 1), size=(2, 2, 3), vacuum_A=15.0)
    assert slab.cell.pbc == (True, True, False)
    z = slab.positions[:, 2]
    assert z.min() == pytest.approx(15.0, abs=1e-6)
    assert slab.cell.matrix[2][2] - z.max() == pytest.approx(15.0, abs=1e-6)


def test_slab_atoms_lie_inside_the_two_dimensional_cell():
    slab = make_surface(load("silicon"), (1, 1, 1), size=(2, 2, 3), vacuum_A=12.0)
    a1, a2, _ = slab.cell.matrix
    basis = np.array([[a1[0], a1[1]], [a2[0], a2[1]]])
    frac = slab.positions[:, :2] @ np.linalg.inv(basis)
    assert frac.min() > -1e-6
    assert frac.max() < 1.0 + 1e-6


def test_stable_termination_leaves_one_dangling_bond_on_silicon_111():
    shift, candidates = stable_termination_shift(load("silicon"), (1, 1, 1))
    assert shift == pytest.approx(0.25, abs=1e-6)
    scores = sorted({c["broken_bonds"] for c in candidates})
    assert scores[0] < scores[-1]
    slab = make_surface(load("silicon"), (1, 1, 1), size=(2, 2, 4), vacuum_A=12.0)
    total, top = _broken_bonds(slab)
    assert top == 4


def test_silicon_100_has_two_dangling_bonds_per_surface_atom():
    slab = make_surface(load("silicon"), (1, 0, 0), size=(2, 2, 4), vacuum_A=12.0)
    _, top = _broken_bonds(slab)
    assert top == 8


def test_surface_records_full_provenance():
    slab = make_surface(load("silicon"), (1, 1, 1), size=(1, 1, 2))
    provenance = slab.info["provenance"]
    assert provenance["origin"] == "calculated"
    assert provenance["fidelity"] == "tier0-structural"
    assert any("no surface reconstruction" in a for a in provenance["approximations"])
    assert provenance["boundary_conditions"].startswith("2D periodic")


def test_strain_scales_cell_and_positions_affinely():
    slab = make_surface(load("silicon"), (1, 1, 1), size=(1, 1, 2))
    strained = apply_strain(slab, [0.02, 0.02, 0.0])
    assert strained.cell.lengths[0] == pytest.approx(slab.cell.lengths[0] * 1.02, rel=1e-9)
    assert strained.positions[:, 2] == pytest.approx(slab.positions[:, 2])


def test_vacancy_records_what_was_removed_and_marks_neighbours(si111_small):
    slab = si111_small.copy()
    target = int(slab.ids[len(slab) // 2])
    record = create_vacancy(slab, target)
    assert record["type"] == "vacancy"
    assert record["element"] == "Si"
    assert not slab.has_id(target)
    assert record["neighbour_ids"]
    for neighbour in record["neighbour_ids"]:
        assert str(slab.roles[slab.index_of(neighbour)]) == "vacancy-neighbour"


def test_dopant_substitution_keeps_the_site(si111_small):
    slab = si111_small.copy()
    site = nearest_site(slab, slab.positions[0])
    before = slab.positions[slab.index_of(site)].copy()
    record = add_dopant(slab, "P", site=site)
    assert record["to"] == "P"
    assert slab.positions[slab.index_of(site)] == pytest.approx(before)
    assert str(slab.roles[slab.index_of(site)]) == "dopant"


def test_overlapping_interstitial_is_refused(si111_small):
    slab = si111_small.copy()
    with pytest.raises(Exception) as excinfo:
        add_interstitial(slab, "Si", slab.positions[0] + 0.1)
    assert "minimum separation" in str(excinfo.value)


def test_adatom_sits_above_the_surface(si111_small):
    slab = si111_small.copy()
    top = slab.positions[:, 2].max()
    record = add_adatom(slab, "Si", (2.0, 2.0), height_A=2.2)
    assert record["position_A"][2] > top
    assert str(slab.roles[slab.index_of(record["atom_id"])]) == "adatom"


def test_adatom_default_height_is_the_covalent_contact(si111_small):
    from materia.elements import periodic_table as pt

    slab = si111_small.copy()
    top = int(np.argmax(slab.positions[:, 2]))
    x, y, _ = slab.positions[top]
    record = add_adatom(slab, "H", (x, y))
    added = slab.positions[slab.index_of(record["atom_id"])]
    contact = pt.covalent_radius("H") + pt.covalent_radius("Si")
    assert record["bonded_to_atom_id"] == int(slab.ids[top])
    assert np.linalg.norm(added - slab.positions[top]) == pytest.approx(contact, abs=1e-9)
    for xy in ((x + 1.3, y + 0.4), (x - 0.7, y + 1.9)):
        other = si111_small.copy()
        rec = add_adatom(other, "O", xy)
        position = other.positions[other.index_of(rec["atom_id"])]
        rest = np.delete(other.positions, other.index_of(rec["atom_id"]), axis=0)
        distances = np.linalg.norm(rest - position, axis=1)
        limit = pt.covalent_radius("O") + pt.covalent_radius("Si")
        assert distances.min() == pytest.approx(limit, abs=1e-9)
        assert "covalent" in rec["placement"]
