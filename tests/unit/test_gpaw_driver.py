"""Software behaviour of the GPAW driver.

These cases are about the driver, not about density functional theory: they
check discovery, configuration validation, structure conversion, refusal paths
and the shape of the provenance record.  Almost all of them run whether or not
GPAW is installed, because refusing correctly when it is absent is most of what
this layer has to get right.  The physics is checked separately in the
physical-validation suite.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from materia.core_model.cell import Cell
from materia.core_model.structure import Structure
from materia.solvers.gpaw_driver import (
    FUNCTIONALS,
    ConversionError,
    GPAWEnvironment,
    GPAWSettings,
    GPAWSettingsError,
    GPAWSolver,
    describe_presets,
    discover,
    from_worker_spec,
    preset,
    to_worker_spec,
    validate,
)
from materia.solvers.gpaw_driver.environment import INTERPRETER_ENV_VAR
from materia.structure_builder import make_surface
from materia.materials import load


def molecule(distance: float = 0.74, box: float = 6.0) -> Structure:
    centre = box / 2
    return Structure(
        np.array([1, 1]),
        np.array([[centre, centre, centre - distance / 2],
                  [centre, centre, centre + distance / 2]]),
        Cell(np.eye(3) * box, (False, False, False)),
    )


class TestEnvironmentReporting:

    def test_code_and_datasets_are_reported_separately(self):
        environment = discover()
        assert isinstance(environment.code_available, bool)
        assert isinstance(environment.datasets_available, bool)
        assert environment.operational == (
            environment.code_available and environment.datasets_available)

    def test_importable_gpaw_without_datasets_is_not_operational(self):
        """An importable package with no PAW setups cannot run anything, so it
        must not be reported as a working solver."""
        environment = GPAWEnvironment(
            interpreter="/somewhere/python", source="test",
            code_available=True, datasets_available=False,
            gpaw_version="25.7.0", setup_paths=["/nowhere"])
        assert environment.operational is False
        assert "no PAW datasets" in environment.blocking_reason()
        assert "gpaw-data" in environment.install_hint()

    def test_missing_code_and_missing_data_give_different_instructions(self):
        no_code = GPAWEnvironment(interpreter="/p", code_available=False)
        no_data = GPAWEnvironment(interpreter="/p", code_available=True,
                                  gpaw_version="25.7.0", datasets_available=False)
        assert "not importable" in no_code.blocking_reason()
        assert "no PAW datasets" in no_data.blocking_reason()
        assert no_code.install_hint() != no_data.install_hint()
        assert "conda create" in no_code.install_hint()

    def test_no_interpreter_names_the_environment_variable(self):
        nothing = GPAWEnvironment()
        assert INTERPRETER_ENV_VAR in nothing.blocking_reason()

    def test_environment_serialises(self):
        json.dumps(discover().as_dict())


class TestSettingsValidation:

    def test_presets_are_valid_and_documented(self):
        for entry in describe_presets():
            assert entry["note"]
            assert validate(entry["settings"]).task == "energy"

    def test_units_are_named(self):
        units = GPAWSettings().units()
        assert units["cutoff_eV"] == "eV"
        assert units["grid_spacing_A"] == "A"
        assert units["smearing_eV"] == "eV"
        assert units["energy_tol_eV_per_electron"] == "eV/electron"

    @pytest.mark.parametrize("raw,fragment", [
        ({"task": "relax"}, "are not implemented"),
        ({"task": "band_structure"}, "task must be one of"),
        ({"xc": "B3LYP"}, "xc must be one of"),
        ({"mode": "wavelet"}, "mode must be one of"),
        ({"mode": "pw", "cutoff_eV": 50}, "below 100 eV"),
        ({"mode": "pw", "cutoff_eV": "x"}, "must be a number"),
        ({"mode": "fd", "grid_spacing_A": 0.5}, "coarser than"),
        ({"mode": "fd", "grid_spacing_A": -0.1}, "positive, finite"),
        ({"mode": "lcao", "basis": ""}, "must name an LCAO basis"),
        ({"kpoints": (0, 1, 1)}, "at least 1"),
        ({"kpoints": (1, 1)}, "three divisions"),
        ({"occupations": "gaussian"}, "occupations must be one of"),
        ({"smearing_eV": -0.1}, "must not be negative"),
        ({"occupations": "fermi-dirac", "smearing_eV": 0.0}, "fixed occupation"),
        ({"max_iterations": 0}, "at least 1"),
        ({"max_iterations": 2.5}, "whole number"),
        ({"energy_tol_eV_per_electron": 0}, "positive, finite"),
        ({"unknown_knob": 1}, "Unknown GPAW setting"),
    ])
    def test_bad_settings_are_refused_with_the_reason(self, raw, fragment):
        with pytest.raises(GPAWSettingsError, match=fragment):
            validate(raw)

    def test_a_task_refusal_says_the_driver_will_not_approximate(self):
        with pytest.raises(GPAWSettingsError) as excinfo:
            validate({"task": "relax"})
        assert "refused rather than approximated" in str(excinfo.value)

    def test_nothing_is_silently_corrected(self):
        settings = validate({"mode": "fd", "grid_spacing_A": 0.22, "xc": "LDA"})
        assert settings.grid_spacing_A == 0.22
        assert settings.xc == "LDA"
        assert settings.mode == "fd"


class TestSettingsAgainstTheStructure:

    def test_plane_waves_are_refused_on_an_open_cell(self):
        """GPAW's plane-wave mode treats the cell as periodic whatever the
        structure says, which silently changes the boundary conditions. It is
        refused rather than run under a different physical setup."""
        slab = make_surface(load("silicon"), (1, 1, 1), size=(1, 1, 2), vacuum_A=8.0)
        with pytest.raises(GPAWSettingsError, match="periodic boundaries in all three"):
            validate({"mode": "pw", "cutoff_eV": 340.0}, slab)

    def test_kpoints_along_a_vacuum_direction_are_refused(self):
        slab = make_surface(load("silicon"), (1, 1, 1), size=(1, 1, 2), vacuum_A=8.0)
        with pytest.raises(GPAWSettingsError, match="not a periodic direction"):
            validate({"mode": "fd", "kpoints": (2, 2, 2)}, slab)

    def test_in_plane_kpoints_on_a_slab_are_accepted(self):
        slab = make_surface(load("silicon"), (1, 1, 1), size=(1, 1, 2), vacuum_A=8.0)
        settings = validate({"mode": "fd", "kpoints": (4, 4, 1)}, slab)
        assert settings.kpoints == (4, 4, 1)

    def test_a_charge_is_refused_before_any_electron_counting(self):
        """The old rule tried to decide whether a charged system was open-shell.
        It double-subtracted the charge and let a one-electron H2+ through as a
        closed shell. Charged systems are now refused outright, so no electron
        parity argument is needed."""
        with pytest.raises(GPAWSettingsError, match="(?i)charged systems"):
            validate({"charge": 1.0, "spin_polarized": False, "mode": "fd"}, molecule())

    def test_an_empty_structure_is_refused(self):
        empty = Structure(np.array([], dtype=int), np.zeros((0, 3)), Cell(np.eye(3) * 5))
        with pytest.raises(GPAWSettingsError, match="no atoms"):
            validate({"mode": "fd"}, empty)

    def test_a_surface_preset_fits_a_slab(self):
        slab = make_surface(load("silicon"), (1, 1, 1), size=(1, 1, 2), vacuum_A=8.0)
        settings = preset("surface", slab)
        assert settings.mode == "fd"
        assert settings.kpoints[2] == 1
        assert "PBE" in settings.summary()


class TestConversion:

    def test_round_trip_preserves_everything_gpaw_can_use(self):
        slab = make_surface(load("silicon"), (1, 1, 1), size=(2, 2, 2), vacuum_A=8.0,
                            fix_bottom_layers=2)
        slab.magnetic_moments[0] = 1.5
        spec = to_worker_spec(slab)
        back = from_worker_spec(spec)
        assert np.array_equal(back.numbers, slab.numbers)
        assert np.allclose(back.positions, slab.positions)
        assert np.allclose(back.cell.matrix, slab.cell.matrix)
        assert tuple(back.cell.pbc) == tuple(slab.cell.pbc)
        assert np.array_equal(back.fixed, slab.fixed)
        assert np.allclose(back.magnetic_moments, slab.magnetic_moments)
        assert np.array_equal(back.ids, slab.ids)

    def test_the_spec_is_json_serialisable(self):
        json.dumps(to_worker_spec(molecule()))

    def test_forces_come_back_attached_to_atom_ids(self):
        from materia.solvers.gpaw_driver.conversion import apply_forces, forces_by_id

        slab = make_surface(load("silicon"), (1, 1, 1), size=(1, 1, 2), vacuum_A=8.0)
        spec = to_worker_spec(slab)
        forces = [[0.1 * i, 0.0, -0.2] for i in range(len(slab))]
        by_id = forces_by_id(spec, forces)
        assert set(by_id) == {int(i) for i in slab.ids}
        apply_forces(slab, by_id)
        assert np.allclose(slab.forces[2], [0.2, 0.0, -0.2])

    def test_a_wrong_number_of_forces_is_refused(self):
        from materia.solvers.gpaw_driver.conversion import forces_by_id

        spec = to_worker_spec(molecule())
        with pytest.raises(ConversionError, match="force vectors"):
            forces_by_id(spec, [[0.0, 0.0, 0.0]])

    def test_forces_for_atoms_that_no_longer_exist_are_refused(self):
        from materia.solvers.gpaw_driver.conversion import apply_forces

        structure = molecule()
        with pytest.raises(ConversionError, match="no longer holds atom id"):
            apply_forces(structure, {9999: [0.0, 0.0, 0.0]})
        assert not np.isfinite(structure.forces).any()

    def test_an_empty_structure_cannot_be_converted(self):
        empty = Structure(np.array([], dtype=int), np.zeros((0, 3)), Cell(np.eye(3) * 5))
        with pytest.raises(ConversionError, match="no atoms"):
            to_worker_spec(empty)


class TestSolverDeclaration:

    def test_only_energy_and_forces_are_advertised(self):
        """GPAW can do far more than this driver drives. Advertising a
        capability the driver does not implement would be a false claim about
        Materia, not about GPAW."""
        solver = GPAWSolver()
        assert {c.value for c in solver.capabilities} == {"energy", "forces"}
        description = solver.describe()
        assert description["fidelity"] == "tier3-external-first-principles"
        for absent in ("relax", "band_structure", "local_dos", "charge_density",
                       "spin_polarized", "charged_system", "stress", "stm_ldos"):
            assert absent not in description["capabilities"]
            assert absent in description["missing_capabilities"]

    def test_a_task_it_does_not_drive_is_refused(self):
        out = GPAWSolver().run(molecule(), task="relax")
        refusal = out.results["relax"]
        assert not refusal.supported
        assert "does not implement" in refusal.unsupported_reason
        assert "will not pretend" in refusal.unsupported_reason

    def test_it_is_registered_under_its_own_name(self):
        from materia.solvers import registry

        assert "external:gpaw" in registry.available()
        assert isinstance(registry.create("external:gpaw"), GPAWSolver)

    def test_an_unavailable_gpaw_refuses_with_installation_instructions(self):
        solver = GPAWSolver(environment=GPAWEnvironment(interpreter=None))
        out = solver.single_point(molecule())
        refusal = out.results["energy"]
        assert not refusal.supported
        assert INTERPRETER_ENV_VAR in refusal.unsupported_reason
        assert refusal.extra["install_hint"]
        assert refusal.value is None

    def test_a_missing_dataset_element_is_refused_by_name(self):
        environment = GPAWEnvironment(
            interpreter="/p", source="test", code_available=True,
            datasets_available=True, gpaw_version="25.7.0",
            dataset_dirs=[{"path": "/d", "files": 3}], dataset_elements=["H", "He"])
        report = GPAWSolver(environment=environment).supports(
            make_surface(load("silicon"), (1, 1, 1), size=(1, 1, 2), vacuum_A=8.0))
        assert not report.ok
        assert any("No PAW dataset is installed for Si" in b for b in report.blocking)


class TestASEFidelityAudit:
    """ASE is an interface, not a fidelity level."""

    def test_the_ase_adapter_itself_claims_no_tier(self):
        from materia.solvers.external import ADAPTERS

        ase = next(a for a in ADAPTERS if a.name == "external:ase")
        assert ase.fidelity.value == "non-physical"
        assert "computes nothing itself" in ase.description

    @pytest.mark.parametrize("name,expected", [
        ("GPAW", "tier3-external-first-principles"),
        ("Espresso", "tier3-external-first-principles"),
        ("DFTB", "tier2-semi-empirical"),
        ("EMT", "tier1-classical"),
        ("LennardJones", "tier1-classical"),
    ])
    def test_the_tier_comes_from_the_calculator(self, name, expected):
        from materia.solvers.external import calculator_fidelity

        calculator = type(name, (), {})()
        tier, note = calculator_fidelity(calculator)
        assert tier.value == expected
        assert name in note

    def test_an_unrecognised_calculator_gets_no_tier_and_says_so(self):
        from materia.solvers.external import calculator_fidelity

        tier, note = calculator_fidelity(type("Mystery", (), {})())
        assert tier.value == "non-physical"
        assert "does not recognise" in note
        assert "classify it yourself" in note


class TestRealCalculation:
    """Cases that actually run GPAW. Skipped when there is none to run."""

    def test_a_molecule_converges_with_full_provenance(self, needs_gpaw):
        structure = molecule()
        out = GPAWSolver().single_point(structure, {
            "xc": "LDA", "mode": "fd", "grid_spacing_A": 0.18,
            "occupations": "fixed", "smearing_eV": 0.0,
            "energy_tol_eV_per_electron": 1e-3, "max_iterations": 120})
        energy = out.results["energy"]
        assert energy.supported
        assert energy.unit == "eV"
        assert -10.0 < energy.value < -3.0
        assert out.convergence.converged is True
        assert out.convergence.iterations > 0

        parameters = energy.provenance.parameters
        for key in ("gpaw_version", "ase_version", "gpaw_python", "mpi_world_size",
                    "xc", "mode", "grid_spacing_A", "kpoints", "occupations",
                    "charge_e", "spin_polarized", "scf_iterations", "wall_time_s"):
            assert key in parameters, key
        assert parameters["gpaw_version"]
        assert parameters["paw_datasets"]
        assert parameters["paw_datasets"][0]["fingerprint"]
        assert energy.provenance.dataset
        assert energy.provenance.fidelity.value == "tier3-external-first-principles"
        assert energy.provenance.origin.value == "calculated"
        assert "exact" not in energy.provenance.notes.lower().replace("not an exact", "")
        assert energy.provenance.tolerances["energy_eV_per_electron"] == 1e-3

    def test_the_structure_is_not_mutated_by_a_calculation(self, needs_gpaw):
        structure = molecule()
        before = structure.positions.copy()
        GPAWSolver().single_point(structure, {
            "xc": "LDA", "mode": "fd", "grid_spacing_A": 0.20,
            "occupations": "fixed", "smearing_eV": 0.0,
            "energy_tol_eV_per_electron": 1e-2, "max_iterations": 60})
        assert np.allclose(structure.positions, before)
        assert not np.isfinite(structure.forces).any()

    def test_a_run_that_does_not_converge_returns_no_energy(self, needs_gpaw):
        """A half-solved Kohn-Sham problem is not a total energy."""
        out = GPAWSolver().single_point(molecule(), {
            "xc": "LDA", "mode": "fd", "grid_spacing_A": 0.20,
            "occupations": "fixed", "smearing_eV": 0.0,
            "energy_tol_eV_per_electron": 1e-12,
            "density_tol_electrons": 1e-12, "max_iterations": 3})
        energy = out.results["energy"]
        assert energy.value is None
        assert not energy.supported
        assert energy.provenance.origin.value == "unsupported"
        assert "did not reach self-consistency" in energy.unsupported_reason
        assert energy.extra["status"] == "not-converged"
        assert out.convergence.converged is False
        assert "forces" not in out.results

    def test_a_cancelled_run_returns_no_energy(self, needs_gpaw):
        out = GPAWSolver().single_point(
            molecule(), {"xc": "LDA", "mode": "fd", "grid_spacing_A": 0.18,
                         "occupations": "fixed", "smearing_eV": 0.0,
                         "max_iterations": 200},
            cancelled=lambda: True)
        energy = out.results["energy"]
        assert energy.value is None
        assert energy.extra["status"] == "cancelled"
        assert "cancelled" in energy.unsupported_reason
        assert out.convergence.converged is False

    def test_a_failed_calculation_reports_the_error(self, needs_gpaw):
        broken = Structure(np.array([1, 1]),
                           np.array([[3.0, 3.0, 3.0], [3.0, 3.0, 3.0]]),
                           Cell(np.eye(3) * 6.0, (False, False, False)))
        out = GPAWSolver().single_point(broken, {
            "xc": "LDA", "mode": "fd", "grid_spacing_A": 0.25,
            "occupations": "fixed", "smearing_eV": 0.0, "max_iterations": 20})
        energy = out.results["energy"]
        assert energy.value is None
        assert energy.extra["status"] in ("failed", "not-converged")


class TestChargedAndSpinPolarisedAreRefused:
    """The driver has not been exercised on charged or spin-polarised systems,
    so it declines them instead of handing GPAW an input it has never checked.

    The electron bookkeeping is the reason this needs its own suite.
    ``Structure.total_electrons`` already subtracts formal charges, so
    subtracting a requested charge again made a one-electron H2+ look like a
    closed shell and let a spin-paired calculation through.
    """

    @staticmethod
    def cation(charge: float = 1.0) -> Structure:
        structure = molecule()
        structure.formal_charges[0] = charge
        return structure

    def test_the_electron_count_is_nuclear_charge_minus_formal_charge(self):
        neutral = molecule()
        assert neutral.total_electrons() == pytest.approx(2.0)
        assert neutral.total_charge() == pytest.approx(0.0)
        cation = self.cation(1.0)
        assert cation.total_charge() == pytest.approx(1.0)
        assert cation.total_electrons() == pytest.approx(1.0)
        anion = self.cation(-1.0)
        assert anion.total_electrons() == pytest.approx(3.0)

    def test_a_neutral_closed_shell_is_still_accepted(self):
        settings = validate({"mode": "fd", "grid_spacing_A": 0.2}, molecule())
        assert settings.charge == 0.0
        assert settings.spin_polarized is False

    @pytest.mark.parametrize("charge", [1.0, -1.0, 0.5, 2])
    def test_a_requested_charge_is_refused(self, charge):
        with pytest.raises(GPAWSettingsError, match="(?i)charged systems"):
            validate({"mode": "fd", "charge": charge})

    def test_a_structure_that_carries_charge_is_refused(self):
        with pytest.raises(GPAWSettingsError, match="net formal charge"):
            validate({"mode": "fd"}, self.cation(1.0))

    def test_a_negatively_charged_structure_is_refused(self):
        with pytest.raises(GPAWSettingsError, match="net formal charge"):
            validate({"mode": "fd"}, self.cation(-1.0))

    def test_a_charge_mismatch_names_both_values(self):
        with pytest.raises(GPAWSettingsError) as excinfo:
            validate({"mode": "fd", "charge": 2.0}, self.cation(1.0))
        message = str(excinfo.value)
        assert "Charged systems" in message
        assert "+1" in message
        assert "+2" in message

    def test_spin_polarisation_is_refused(self):
        with pytest.raises(GPAWSettingsError, match="(?i)spin-polarised"):
            validate({"mode": "fd", "spin_polarized": True})

    def test_initial_magnetic_moments_are_refused(self):
        structure = molecule()
        structure.magnetic_moments[0] = 1.0
        with pytest.raises(GPAWSettingsError, match="magnetic moments"):
            validate({"mode": "fd"}, structure)

    @pytest.mark.parametrize("value", ["false", "true", "0", 1, 0, None, []])
    def test_spin_polarized_must_be_a_real_boolean(self, value):
        with pytest.raises(GPAWSettingsError, match="spin_polarized must be"):
            validate({"mode": "fd", "spin_polarized": value})

    @pytest.mark.parametrize("value", ["0", "one", None, []])
    def test_charge_must_be_a_real_number(self, value):
        with pytest.raises(GPAWSettingsError, match="charge must be"):
            validate({"mode": "fd", "charge": value})

    def test_no_gpaw_process_starts_for_a_refused_request(self, monkeypatch):
        from materia.solvers.gpaw_driver import runner

        def explode(*args, **kwargs):
            raise AssertionError("a GPAW subprocess was started for a refused request")

        from materia.solvers.base import SupportReport

        monkeypatch.setattr(runner, "run", explode)
        monkeypatch.setattr(GPAWSolver, "supports",
                            lambda self, structure: SupportReport(True, [], []))
        for settings in ({"mode": "fd", "charge": 1.0},
                         {"mode": "fd", "spin_polarized": True}):
            with pytest.raises(GPAWSettingsError):
                GPAWSolver().single_point(molecule(), settings)
        structure = molecule()
        structure.magnetic_moments[0] = 1.0
        with pytest.raises(GPAWSettingsError):
            GPAWSolver().single_point(structure, {"mode": "fd"})

    def test_the_refusal_names_the_capability(self):
        with pytest.raises(GPAWSettingsError) as excinfo:
            validate({"mode": "fd", "charge": 1.0})
        assert "charged_system" in str(excinfo.value)
        with pytest.raises(GPAWSettingsError) as excinfo:
            validate({"mode": "fd", "spin_polarized": True})
        assert "spin_polarized" in str(excinfo.value)

    def test_neither_capability_is_declared(self):
        declared = {c.value for c in GPAWSolver().capabilities}
        assert "charged_system" not in declared
        assert "spin_polarized" not in declared


class TestPythonApiInputIdentity:

    def test_partial_settings_are_validated_before_the_digest(self, monkeypatch):
        from materia.provenance import Convergence, Fidelity, Origin, Provenance, Result
        from materia.python_api import Lab
        from materia.solvers.base import SolverResult
        from materia.solvers.gpaw_driver.conversion import input_digest

        lab = Lab()
        surface = lab.materials.load("silicon").create_surface(
            size=(1, 1, 2), orientation=(1, 1, 1), vacuum_angstrom=8.0)
        structure = surface.structure
        requested = {
            "mode": "fd",
            "grid_spacing_A": 0.25,
            "kpoints": [1, 1, 1],
            "xc": "LDA",
        }
        expected = validate(requested, structure).as_dict()
        captured = {}

        class Solver:
            def single_point(self, target, settings, **kwargs):
                captured["settings"] = settings
                provenance = Provenance(
                    "test", Fidelity.TIER3_EXTERNAL, Origin.CALCULATED)
                convergence = Convergence(True, 1)
                result = SolverResult("test", target, convergence=convergence)
                result.results["energy"] = Result(
                    "energy", -1.0, "eV", provenance, convergence=convergence,
                    extra={"status": "converged"})
                return result

        monkeypatch.setattr(lab.dft, "_solver", lambda: Solver())
        energy = lab.dft.energy(surface, settings=requested)

        assert captured["settings"] == expected
        assert energy.provenance.inputs_digest == input_digest(
            to_worker_spec(structure), expected)
