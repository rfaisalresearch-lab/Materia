"""The LAMMPS adapter's own pieces: units, discovery, specification, native files, parsing."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from materia.core_model.cell import Cell
from materia.core_model.structure import Structure
from materia.core_model.units import U_A2_FS2_TO_EV
from materia.physics import eam
from materia.solvers.lammps import environment as env_module
from materia.solvers.lammps import native, parse, spec as specs, units
from materia.solvers.lammps.environment import LAMMPSEnvironment


def copper(repeat=(2, 2, 2)) -> Structure:
    from materia.python_api.api import Lab

    return Lab().materials.load("copper").bulk(repeat=repeat).structure


def fake_environment(**changes) -> LAMMPSEnvironment:
    values = dict(kind="executable", path="/opt/lammps/bin/lmp", source="test",
                  version="29 Aug 2024 - Update 1", version_id=20240829,
                  packages=["MANYBODY"],
                  styles={"atom": ["atomic"], "pair": ["eam/alloy"],
                          "minimize": ["cg", "fire", "sd"],
                          "fix": ["nve", "nvt", "langevin", "setforce"]})
    values.update(changes)
    return LAMMPSEnvironment(**values)


def build(structure=None, task="energy", environment=None, **settings):
    structure = structure if structure is not None else copper()
    potential = eam.load_shipped("Cu-Zhou04")
    return specs.build(structure, task, potential, environment or fake_environment(),
                       "s1", settings)


class TestUnits:
    def test_time_and_velocity_factors_are_exact(self):
        assert units.fs_to_ps(2.0) == 0.002
        assert units.ps_to_fs(0.002) == 2.0
        v = np.array([[0.01, -0.02, 0.5]])
        assert np.array_equal(units.velocity_to_lammps(v), v * 1000.0)
        assert np.allclose(units.velocity_from_lammps(units.velocity_to_lammps(v)), v,
                           rtol=0, atol=1e-18)

    def test_stress_sign_and_factor(self):
        pressure = units.pressure_voigt_to_tensor(1.6021765e6, 0, 0, 0, 0, 0)
        stress = units.pressure_bar_to_stress_eV_A3(pressure)
        assert stress[0, 0] == pytest.approx(-1.0, rel=1e-15)
        assert units.stress_eV_A3_to_pressure_bar(stress)[0, 0] == pytest.approx(1.6021765e6)
        assert pressure[0, 1] == pressure[1, 0]

    def test_lammps_constants_agree_with_codata(self):
        assert units.LAMMPS_NKTV2P_METAL == pytest.approx(units.CODATA_EV_A3_IN_BAR, rel=1e-7)
        assert units.LAMMPS_MVV2E_METAL == pytest.approx(units.CODATA_MVV2E_METAL, rel=1e-7)
        assert units.CODATA_MVV2E_METAL == pytest.approx(U_A2_FS2_TO_EV * 1e-6, rel=1e-12)

    def test_kinetic_energy_matches_materia_units(self):
        masses = np.array([63.546, 1.008])
        v_fs = np.array([[0.01, 0.0, 0.0], [0.0, 0.03, -0.02]])
        materia = 0.5 * np.sum(masses[:, None] * v_fs ** 2) * U_A2_FS2_TO_EV
        lammps = units.kinetic_energy_eV(masses, units.velocity_to_lammps(v_fs))
        assert lammps == pytest.approx(materia, rel=1e-6)


HELP = """
Large-scale Atomic/Molecular Massively Parallel Simulator - 29 Aug 2024 - Update 1
Git info (stable / stable_29Aug2024_update1)

Installed packages:

KSPACE MANYBODY MOLECULE

List of individual style options included in this LAMMPS executable

* Atom styles:

atomic          charge          full

* Minimize styles:

cg              fire            sd

* Pair styles:

eam             eam/alloy       lj/cut

* Fix styles

langevin        nve             nvt             setforce

"""


class TestEnvironment:
    def test_help_is_parsed_for_version_packages_and_styles(self):
        parsed = env_module.parse_help(HELP)
        assert parsed["version"] == "29 Aug 2024 - Update 1"
        assert parsed["git_info"].startswith("Git info")
        assert parsed["packages"] == ["KSPACE", "MANYBODY", "MOLECULE"]
        assert parsed["styles"]["pair"] == ["eam", "eam/alloy", "lj/cut"]
        assert parsed["styles"]["fix"] == ["langevin", "nve", "nvt", "setforce"]
        assert parsed["styles"]["minimize"] == ["cg", "fire", "sd"]

    def test_version_helpers(self):
        assert env_module.version_from_log_header("LAMMPS (2 Aug 2023)") == "2 Aug 2023"
        assert env_module.version_from_log_header("LAMMPS 2 Aug 2023") == ""
        assert env_module.version_id("29 Aug 2024 - Update 1") == 20240829
        assert env_module.version_id("garbage") is None

    def test_nothing_installed_fails_closed_with_an_install_path(self, no_lammps):
        found = env_module.discover(refresh=True)
        assert not found.available
        report = found.as_dict()
        assert report["available"] is False
        assert "conda install -c conda-forge lammps" in report["install_hint"]
        assert env_module.EXECUTABLE_ENV_VAR in report["install_hint"]
        assert report["candidates"] == []

    def test_fake_executable_is_found_and_reported_exactly(self, fake_lammps):
        found = env_module.discover(refresh=True)
        assert found.available
        assert found.kind == "executable" and found.path == str(fake_lammps)
        assert found.source == env_module.EXECUTABLE_ENV_VAR
        assert found.version == "29 Aug 2024 - Update 1"
        assert found.version_id == 20240829
        assert "MANYBODY" in found.packages
        assert found.has_style("pair", "eam/alloy")
        assert found.candidates[0]["ok"] is True

    def test_explicit_choice_that_fails_is_not_replaced(self, tmp_path, monkeypatch,
                                                        fake_lammps):
        if os.name == "nt":
            bogus = tmp_path / "not-lammps.cmd"
            bogus.write_text("@echo hello\r\n")
        else:
            bogus = tmp_path / "not-lammps"
            bogus.write_text("#!/bin/sh\necho hello\n")
            bogus.chmod(0o755)
        monkeypatch.setenv("PATH", str(fake_lammps.parent))
        monkeypatch.setenv(env_module.EXECUTABLE_ENV_VAR, str(bogus))
        found = env_module.discover(refresh=True)
        assert not found.available
        assert found.path == str(bogus)
        assert "version banner" in found.error
        assert len(found.candidates) == 1

    def test_path_search_finds_named_executables(self, tmp_path, monkeypatch, fake_lammps):
        monkeypatch.delenv(env_module.EXECUTABLE_ENV_VAR)
        monkeypatch.setattr(env_module, "_current_interpreter_has_module", lambda: False)
        found = env_module.discover(refresh=True, search_path=str(fake_lammps.parent))
        assert found.available and found.source == "PATH"

    def test_python_module_route_is_probed_out_of_process(self, fake_lammps_module):
        found = env_module.discover(refresh=True)
        assert found.available, found.error
        assert found.kind == "python-module"
        assert found.version == "29 Aug 2024 - Update 1"
        assert found.module_path.endswith("__init__.py")
        assert found.has_style("pair", "eam/alloy")
        command = found.command("in.lammps", "log.lammps")
        assert command[1].endswith("module_worker.py")

    def test_missing_package_is_reported_as_missing_style(self, fake_lammps, monkeypatch):
        monkeypatch.setenv("FAKE_LAMMPS_PACKAGES", "KSPACE")
        found = env_module.discover(refresh=True)
        assert found.available
        spec = build(environment=found)
        report = specs.check(spec, found)
        assert any("pair style eam/alloy" in m for m in report.blocking["lammps"])


class TestSpec:
    def test_spec_is_frozen_versioned_and_round_trips(self):
        spec = build(task="md", steps=20, sample_every=5)
        assert spec.schema == specs.SPEC_SCHEMA and spec.version == specs.SPEC_VERSION
        assert spec.units == "metal" and spec.atom_style == "atomic"
        with pytest.raises(Exception):
            spec.steps = 3
        again = specs.LAMMPSRunSpec.from_dict(json.loads(json.dumps(spec.as_dict())))
        assert again == spec
        assert again.digest == spec.digest
        assert len(spec.digest) == 64
        changed = specs.changed(spec, steps=40)
        assert changed.steps == 40 and changed.digest != spec.digest

    def test_unknown_and_mistyped_settings_are_refused(self):
        with pytest.raises(specs.SpecError, match="Unknown setting"):
            build(task="energy", steps=10)
        with pytest.raises(specs.SpecError, match="cannot be"):
            build(task="md", steps=1.5)
        with pytest.raises(specs.SpecError, match="Unknown LAMMPS task"):
            build(task="neb")
        with pytest.raises(specs.SpecError, match="version"):
            data = build().as_dict()
            data["version"] = "9.9"
            specs.LAMMPSRunSpec.from_dict(data)

    def test_inapplicable_settings_are_none_not_carried(self):
        spec = build(task="energy")
        assert spec.steps is None and spec.ensemble is None and spec.ftol_eV_A is None
        relax = build(task="relax")
        assert relax.steps is None and relax.min_style == "cg"

    def test_default_outputs_follow_periodicity(self):
        assert build().outputs == ("energy", "forces", "stress", "thermo")
        cluster = copper()
        cluster.cell = Cell.none()
        assert "stress" not in build(cluster).outputs
        assert build(task="md", steps=10, sample_every=5).outputs[-1] == "trajectory"

    def test_isotopes_get_their_own_types_with_the_same_element(self):
        s = copper()
        s.mass_numbers[:4] = 65
        spec = build(s)
        assert len(spec.type_table) == 2
        assert {row[1] for row in spec.type_table} == {"Cu"}
        masses = sorted(row[3] for row in spec.type_table)
        assert masses[1] > masses[0]
        script = native.render_input(spec)
        assert "pair_coeff * * Cu_Zhou04.eam.alloy Cu Cu" in script
        for row in spec.type_table:
            assert f"mass {row[0]} {row[3]!r}" in script

    def test_valid_spec_passes(self):
        assert specs.check(build(), fake_environment()).ok

    @pytest.mark.parametrize("settings, variable", [
        ({"task": "md", "steps": 10, "sample_every": 3}, "sample_every"),
        ({"task": "md", "steps": 10, "sample_every": 5, "pressure_bar": 1.0}, "pressure_bar"),
        ({"task": "md", "steps": 10, "sample_every": 5, "timestep_fs": 20.0}, "timestep_fs"),
        ({"task": "md", "steps": 10, "sample_every": 5, "ensemble": "npt"}, "ensemble"),
        ({"task": "md", "steps": 10, "sample_every": 5, "ensemble": "langevin",
          "temperature_K": 0.0}, "temperature_K"),
        ({"task": "md", "steps": 10, "sample_every": 5, "ensemble": "langevin",
          "seed": 0}, "seed"),
        ({"task": "relax", "ftol_eV_A": 0.0}, "ftol_eV_A"),
        ({"task": "relax", "min_style": "hftn"}, "min_style"),
        ({"task": "relax", "max_iterations": 10, "max_evaluations": 5}, "max_evaluations"),
        ({"task": "energy", "outputs": ["energy"]}, "outputs"),
        ({"task": "energy", "outputs": ["energy", "forces", "trajectory"]}, "outputs"),
    ])
    def test_refusals_are_keyed_by_variable(self, settings, variable):
        settings = dict(settings)
        task = settings.pop("task")
        report = specs.check(build(task=task, **settings), fake_environment())
        assert variable in report.blocking, report.as_dict()

    def test_stress_is_refused_without_full_periodicity(self):
        s = copper()
        s.cell = Cell(s.cell.matrix, (True, True, False))
        report = specs.check(build(s, outputs=["energy", "forces", "stress"]),
                             fake_environment())
        assert "outputs" in report.blocking

    def test_unrepresentable_cells_are_refused_not_rotated(self):
        s = copper()
        m = s.cell.matrix.copy()
        m[0, 1] = 0.5
        s.cell = Cell(m, (True, True, True))
        report = specs.check(build(s), fake_environment())
        assert "cell_A" in report.blocking
        assert "does not rotate" in report.blocking["cell_A"][0]
        slab = copper()
        tilted = slab.cell.matrix.copy()
        tilted[2, 0] = 0.4
        slab.cell = Cell(tilted, (True, True, False))
        report = specs.check(build(slab, outputs=["energy", "forces"]), fake_environment())
        assert "cell_A" in report.blocking

    def test_missing_element_and_mass_are_refused(self):
        s = copper()
        s.numbers[0] = 79
        spec = build(s)
        assert "potential" in specs.check(spec, fake_environment()).blocking
        good = build()
        broken = specs.LAMMPSRunSpec.from_dict({**good.as_dict(), "type_table": [
            [1, "Cu", 29, float("nan")]]})
        assert "masses" in specs.check(broken, fake_environment()).blocking

    def test_ids_outside_lammps_range_are_refused(self):
        s = copper()
        s.ids[0] = 0
        assert "atom_ids" in specs.check(build(s), fake_environment()).blocking

    def test_changed_potential_file_is_refused(self, tmp_path):
        good = build()
        copy = tmp_path / "Cu.eam.alloy"
        copy.write_bytes(open(good.potential["path"], "rb").read() + b"\n")
        data = good.as_dict()
        data["potential"] = {**data["potential"], "path": str(copy)}
        report = specs.check(specs.LAMMPSRunSpec.from_dict(data), fake_environment())
        assert "potential" in report.blocking

    def test_environment_changes_are_refused(self):
        spec = build()
        newer = fake_environment(version="7 Feb 2024")
        assert "lammps" in specs.check(spec, newer).blocking
        gone = LAMMPSEnvironment()
        report = specs.check(spec, gone)
        assert "conda install" in report.blocking["lammps"][0]
        unpinned = build(environment=LAMMPSEnvironment())
        assert "Freeze it again" in specs.check(unpinned, fake_environment()).blocking[
            "lammps"][0]

    def test_fixed_atoms_with_velocity_are_refused_for_dynamics(self):
        s = copper()
        s.fixed[0] = True
        s.velocities[0] = [0.01, 0.0, 0.0]
        report = specs.check(build(s, task="md", steps=10, sample_every=5),
                             fake_environment())
        assert "velocities_A_fs" in report.blocking


class TestNative:
    def test_box_wrap_round_trips_positions(self):
        s = copper()
        rng = np.random.default_rng(1)
        positions = s.positions + rng.normal(0, 3.0, s.positions.shape)
        box = native.box_geometry(s.cell.matrix, s.cell.pbc, positions)
        assert np.all(box.wrapped >= -1e-12)
        assert np.all(box.wrapped < s.cell.lengths + 1e-12)
        assert np.abs(box.images).max() >= 1
        back = native.unwrap(box.wrapped, box.images, s.cell.matrix, s.cell.pbc)
        assert np.allclose(back, positions, rtol=0, atol=1e-12)

    def test_triclinic_and_slab_boxes(self):
        hexagonal = Cell(np.array([[3.0, 0, 0], [-1.5, 2.598, 0], [0, 0, 10.0]]),
                         (True, True, False))
        points = np.array([[0.0, 0.0, 5.0], [4.0, 1.0, -2.0]])
        box = native.box_geometry(hexagonal.matrix, hexagonal.pbc, points)
        assert box.triclinic and box.tilt[0] == -1.5
        assert box.lo[2] == pytest.approx(-3.0) and box.hi[2] == pytest.approx(6.0)
        assert np.allclose(native.unwrap(box.wrapped, box.images, hexagonal.matrix,
                                         hexagonal.pbc), points, atol=1e-12)
        assert np.all(box.images[:, 2] == 0)

    def test_id_ranges(self):
        assert native.id_ranges([5, 1, 2, 3, 9, 10]) == ["1:3", "5", "9:10"]
        assert native.id_ranges([]) == []

    def test_data_file_is_deterministic_and_complete(self):
        spec = build()
        data = native.render_data(spec)
        assert data == native.render_data(specs.LAMMPSRunSpec.from_dict(spec.as_dict()))
        assert f"{spec.n_atoms} atoms" in data and "1 atom types" in data
        assert "\nMasses\n" in data and "\nAtoms\n" in data and "\nVelocities\n" in data
        assert "xy xz yz" not in data
        atoms = data.split("\nAtoms\n\n")[1].split("\n\nVelocities")[0].splitlines()
        assert len(atoms) == spec.n_atoms
        first = atoms[0].split()
        assert int(first[0]) == spec.atom_ids[0] and len(first) == 8
        assert "#" not in data

    def test_velocities_are_written_in_angstrom_per_picosecond(self):
        s = copper()
        s.velocities[0] = [0.01, -0.02, 0.003]
        data = native.render_data(build(s))
        line = data.split("\nVelocities\n\n")[1].splitlines()[0].split()
        assert [float(v) for v in line[1:]] == pytest.approx([10.0, -20.0, 3.0])

    def test_input_scripts_per_task(self):
        energy = native.render_input(build())
        assert "run 0" in energy and "minimize" not in energy and "timestep" not in energy
        assert energy.strip().endswith(f'print "{native.COMPLETE}"')
        relax = native.render_input(build(task="relax", ftol_eV_A=0.002))
        assert "min_modify norm max" in relax and "minimize 0.0 0.002 1000 10000" in relax
        md = native.render_input(build(task="md", steps=100, timestep_fs=2.0, sample_every=10,
                                       ensemble="langevin", temperature_K=500.0,
                                       damping_fs=50.0, seed=7))
        assert "timestep 0.002" in md
        assert "fix materia_thermostat all langevin 500.0 500.0 0.05 7 zero yes" in md
        assert "dump materia_trajectory all custom 10 trajectory.dump" in md
        assert "unfix materia_thermostat" in md and "run 100" in md
        nvt = native.render_input(build(task="md", steps=10, sample_every=5, ensemble="nvt",
                                        temperature_K=300.0, damping_fs=100.0))
        assert "fix materia_integrate all nvt temp 300.0 300.0 0.1" in nvt
        created = native.render_input(build(task="md", steps=10, sample_every=5,
                                            initial_velocities="create", seed=3))
        assert "velocity all create 300.0 3 dist gaussian mom yes rot no loop geom" in created
        for script in (energy, relax, md, nvt, created):
            assert "#" not in script

    def test_fixed_atoms_use_groups_and_are_released_for_final_forces(self):
        s = copper()
        s.fixed[[0, 1, 2, 5]] = True
        script = native.render_input(build(s, task="relax"))
        assert "group materia_fixed id 1:3 6" in script
        assert "fix materia_freeze materia_fixed setforce 0.0 0.0 0.0" in script
        assert script.index("unfix materia_freeze") < script.index(native.SECTION_FINAL)

    def test_write_inputs_verifies_the_potential_copy(self, tmp_path):
        spec = build()
        identities = native.write_inputs(spec, str(tmp_path))
        assert identities["Cu_Zhou04.eam.alloy"]["sha256"] == spec.potential["sha256"]
        assert identities["in.lammps"]["sha256"] == native.sha256_text(
            (tmp_path / "in.lammps").read_text())


LOG = """LAMMPS (29 Aug 2024 - Update 1)
units metal
print "@@MATERIA section main"
@@MATERIA section main
minimize 0.0 0.01 100 1000
   Step Time PotEng KinEng TotEng Temp Press Pxx Pyy Pzz Pxy Pxz Pyz Atoms
   0 0 -10.5 0 -10.5 0 1 1 1 1 0 0 0 4
WARNING: something harmless
   3 0 -10.75 0 -10.75 0 1 1 1 1 0 0 0 4
Loop time of 0.01 on 1 procs for 3 steps with 4 atoms

Minimization stats:
  Stopping criterion = force tolerance
  Energy initial, next-to-last, final =
    -10.5   -10.7   -10.75
  Force two-norm initial, final = 1.0 0.001
  Force max component initial, final = 1.0 0.0005
  Final line search alpha, max atom move = 0.0 0.0
  Iterations, force evaluations = 3 4

print "@@MATERIA section final"
@@MATERIA section final
   Step Time PotEng KinEng TotEng Temp Press Pxx Pyy Pzz Pxy Pxz Pyz Atoms
   3 0 -10.75 0 -10.75 0 1 1 1 1 0 0 0 4
Loop time of 0.001 on 1 procs for 0 steps with 4 atoms
@@MATERIA complete
Total wall time: 0:00:00
"""


class TestParse:
    def test_log_sections_minimisation_and_markers(self):
        parsed = parse.parse_log(LOG)
        assert parsed.version == "29 Aug 2024 - Update 1"
        assert parsed.complete and parsed.total_wall_time == "0:00:00"
        assert parsed.sections["main"].column("pe").tolist() == [-10.5, -10.75]
        assert parsed.sections["final"].rows.shape == (1, 14)
        assert parsed.minimization["stopping_criterion"] == "force tolerance"
        assert parsed.minimization["iterations"] == 3
        assert parsed.minimization["energy_final_eV"] == -10.75
        assert parsed.warnings == ["WARNING: something harmless"]

    def test_truncated_or_malformed_tables_are_errors(self):
        cut = LOG.split("Loop time of 0.01")[0]
        with pytest.raises(parse.OutputError, match="truncated"):
            parse.parse_log(cut)
        bad = LOG.replace("   3 0 -10.75 0 -10.75 0 1 1 1 1 0 0 0 4\nLoop time of 0.01",
                          "   3 0 -10.75 0 -10.75 0 1 1 1 1 0 0 4\nLoop time of 0.01")
        with pytest.raises(parse.OutputError, match="13 values"):
            parse.parse_log(bad)
        worse = LOG.replace("   0 0 -10.5", "   0 0 abc")
        with pytest.raises(parse.OutputError, match="not a number"):
            parse.parse_log(worse)

    def test_errors_are_collected(self):
        parsed = parse.parse_log("LAMMPS (1 Jan 2020)\nERROR: Lost atoms (src/thermo.cpp:1)\n")
        assert parsed.errors and not parsed.complete

    def test_dump_parsing_is_strict(self):
        dump = ("ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n2\nITEM: BOX BOUNDS pp pp pp\n"
                "0 1\n0 1\n0 1\nITEM: ATOMS id type xu yu zu\n1 1 0 0 0\n2 1 0.5 0.5 0.5\n")
        frames = parse.parse_dump(dump, ("id", "type", "xu", "yu", "zu"))
        assert len(frames) == 1 and frames[0].column("xu").tolist() == [0.0, 0.5]
        with pytest.raises(parse.OutputError, match="truncated"):
            parse.parse_dump(dump.rsplit("\n", 2)[0] + "\n", ("id", "type", "xu", "yu", "zu"))
        with pytest.raises(parse.OutputError, match="columns"):
            parse.parse_dump(dump, ("id", "type", "x", "y", "z"))
        with pytest.raises(parse.OutputError, match="not a number"):
            parse.parse_dump(dump.replace("0.5 0.5 0.5", "0.5 x 0.5"),
                             ("id", "type", "xu", "yu", "zu"))
        with pytest.raises(parse.OutputError, match="TIMESTEP"):
            parse.parse_dump("garbage\n" + dump, ("id", "type", "xu", "yu", "zu"))
