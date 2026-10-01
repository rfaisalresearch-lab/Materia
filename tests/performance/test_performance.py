"""Performance guards.

These assert only that the scaling behaviour and the level-of-detail limits
hold; they are not a benchmark. Measured timings for this machine are produced
by ``materia bench`` and recorded in docs/PERFORMANCE.md.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from materia.benchmarks import _slab_for
from materia.materials import load
from materia.physics.neighbors import neighbor_list
from materia.physics.potentials import StillingerWeber
from materia.structure_builder import make_surface

pytestmark = pytest.mark.performance


@pytest.mark.parametrize("target", [1000, 10000])
def test_neighbour_list_scales_close_to_linearly(target):
    slab = _slab_for(target)
    start = time.perf_counter()
    nl = neighbor_list(slab.positions, slab.cell, 3.0)
    elapsed = time.perf_counter() - start
    assert len(nl) > len(slab)
    assert elapsed < 0.5 + len(slab) * 5e-5


def test_hundred_thousand_atom_neighbour_list_completes():
    slab = _slab_for(100000)
    assert len(slab) > 80000
    start = time.perf_counter()
    nl = neighbor_list(slab.positions, slab.cell, 3.0)
    elapsed = time.perf_counter() - start
    assert len(nl) > 0
    assert elapsed < 30.0


def test_potential_evaluation_scales_with_atom_count():
    small = _slab_for(1000)
    large = _slab_for(10000)
    potential = StillingerWeber("Si")

    def timed(structure):
        potential.energy_and_forces(structure)
        start = time.perf_counter()
        potential.energy_and_forces(structure)
        return time.perf_counter() - start

    ratio = timed(large) / max(timed(small), 1e-6)
    size_ratio = len(large) / len(small)
    assert ratio < size_ratio * 4


def test_renderer_payload_applies_level_of_detail():
    from materia.desktop_ui.service import Service

    service = Service()
    slab = _slab_for(100000)
    service.project.add_structure(slab)
    payload = service.render_payload(max_atoms=20000, include_bonds=False)
    assert payload["truncated"] is True
    assert payload["stride"] > 1
    assert len(payload["ids"]) <= 20000
    assert "Level of detail active" in payload["note"]


def test_renderer_payload_is_complete_for_small_regions():
    from materia.desktop_ui.service import Service

    service = Service()
    slab = make_surface(load("silicon"), (1, 1, 1), size=(4, 4, 4), vacuum_A=12.0)
    service.project.add_structure(slab)
    payload = service.render_payload()
    assert payload["truncated"] is False
    assert payload["stride"] == 1
    assert len(payload["ids"]) == len(slab)
    assert payload["bonds"]


def test_neighbour_list_guards_against_memory_blowups():
    slab = _slab_for(4000)
    with pytest.raises(MemoryError) as excinfo:
        neighbor_list(slab.positions, slab.cell, 40.0, max_pairs=1000)
    assert "Reduce the cutoff" in str(excinfo.value)


def test_electronic_structure_cache_shortens_repeat_scans():
    from materia.microscopy.noise import NoiseModel
    from materia.microscopy.stm import STMSettings, STMSimulator
    from materia.solvers.tight_binding import TightBinding

    STMSimulator.clear_cache()
    slab = make_surface(load("silicon"), (1, 1, 1), size=(3, 3, 3), vacuum_A=12.0)
    simulator = STMSimulator(TightBinding("sp3s*-Si"))
    settings = STMSettings(resolution=(48, 48), kgrid=(1, 1),
                           noise=NoiseModel.quiet(0))

    start = time.perf_counter()
    first = simulator.scan(slab, settings)
    cold = time.perf_counter() - start

    start = time.perf_counter()
    second = simulator.scan(slab, settings)
    warm = time.perf_counter() - start

    assert first.provenance.parameters["electronic_from_cache"] is False
    assert second.provenance.parameters["electronic_from_cache"] is True
    assert warm < cold
    assert second.channel("topography") == pytest.approx(first.channel("topography"))


@pytest.mark.parametrize("slab", [False, True])
def test_ewald_on_a_thousand_ions_stays_interactive(slab):
    from materia.benchmarks import rock_salt_supercell
    from materia.physics.electrostatics import ChargeModel, compute

    structure = rock_salt_supercell(1000, slab=slab)
    model = ChargeModel.per_element({"Na": 1.0, "Cl": -1.0})
    start = time.perf_counter()
    out = compute(structure, model)
    elapsed = time.perf_counter() - start
    assert out.converged
    assert elapsed < 10.0, f"{len(structure)}-ion Ewald with its check took {elapsed:.2f} s"


def test_ewald_cost_grows_close_to_n_to_the_three_halves():
    """Optimally split Ewald costs about N^1.5; a direct pair sum costs N^2."""
    from materia.benchmarks import rock_salt_supercell
    from materia.physics.electrostatics import ChargeModel, EwaldSettings, compute

    model = ChargeModel.per_element({"Na": 1.0, "Cl": -1.0})
    settings = EwaldSettings(check_convergence=False)
    times = {}
    for target in (1000, 8000):
        structure = rock_salt_supercell(target)
        start = time.perf_counter()
        compute(structure, model, settings)
        times[len(structure)] = time.perf_counter() - start
    (n1, t1), (n2, t2) = sorted(times.items())
    assert t2 / t1 < (n2 / n1) ** 1.75


def test_eam_scales_linearly_and_reuses_its_neighbour_list():
    from materia.core_model.cell import Cell
    from materia.core_model.structure import Structure
    from materia.physics import eam

    fcc = np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
    times = {}
    for side in (6, 12):
        s = Structure([29] * 4, fcc * 3.615, Cell.cubic(3.615)).repeat(side, side, side)
        potential = eam.load_shipped("Cu-Zhou04")
        potential.evaluate(s)
        nudged = s.copy()
        nudged.positions = s.positions + 0.01
        start = time.perf_counter()
        potential.evaluate(nudged)
        times[len(s)] = time.perf_counter() - start
        assert potential.last_evaluation["neighbour_list_builds"] == 1
    (n1, t1), (n2, t2) = sorted(times.items())
    assert t2 / t1 < 2.0 * n2 / n1
    assert t2 < 5.0


def test_dft_specification_check_stays_interactive_for_a_thousand_atoms():
    """The panel re-checks the whole specification on every edit, so building
    and checking one must stay well under a second even for a large cell. The
    check includes the coincident-atom search, the boundary and vacuum
    measurement and the dataset lookup (cached by file size and time)."""
    from materia.experiments.dft import spec as specs
    from materia.solvers.gpaw_driver import discover

    slab = _slab_for(1000)
    environment = discover()
    specs.check(specs.build(slab, environment), environment)
    start = time.perf_counter()
    for _ in range(3):
        report = specs.check(specs.build(slab, environment), environment)
    elapsed = (time.perf_counter() - start) / 3
    assert report.boundary["boundary"] == "slab"
    assert elapsed < 1.0


def test_chunked_array_store_round_trips_a_large_grid_quickly(tmp_path):
    """A 128^3 double-precision grid (16 MB, four chunks) is written and read
    back through the project archive with every checksum verified."""
    import zipfile

    from materia.project_format import arrays

    data = np.random.default_rng(3).random((128, 128, 128))
    stored = arrays.StoredArray(data=data, unit="electrons/A^3", kind="volumetric")
    path = tmp_path / "grid.zip"
    start = time.perf_counter()
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        manifest = arrays.write(archive, "grid", stored)
    with zipfile.ZipFile(path) as archive:
        back = arrays.read(archive, "grid")
    elapsed = time.perf_counter() - start
    assert len(manifest["chunks"]) >= 4
    assert np.array_equal(back.data, data)
    assert elapsed < 10.0
