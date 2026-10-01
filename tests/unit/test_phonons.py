"""Finite-displacement harmonic analysis: settings, sum rules, classification, refusals."""

from __future__ import annotations

import math

import numpy as np
import pytest

from materia.core_model.cell import Cell
from materia.core_model.structure import Structure
from materia.core_model.units import BOLTZMANN_EV_K
from materia.physics import phonons as P
from materia.physics.potentials import Harmonic, LennardJones
from materia.project_format.arrays import StoredArray
from materia.provenance import Fidelity, Origin

EPS, SIG = 0.0103, 3.40
R0 = 2 ** (1 / 6) * SIG


def box(numbers, positions, size=20.0, pbc=(False, False, False)):
    return Structure(numbers, np.asarray(positions, dtype=float),
                     Cell(np.eye(3) * size, pbc))


def dimer(distance=R0):
    return box([18, 18], [[5, 5, 5], [5, 5, 5 + distance]])


def lj():
    return LennardJones({"Ar": (EPS, SIG)}, cutoff_A=12.0)


def thz(eigenvalue):
    return math.copysign(math.sqrt(abs(eigenvalue) * P.EIGENVALUE_TO_RAD_S2),
                         eigenvalue) / (2 * math.pi * 1e12)


def saddle(kx=1.0, ky=0.5, kz=2.0):
    def callback(positions):
        x = positions - 5.0
        return -np.column_stack([kx * x[:, 0], -ky * x[:, 1], kz * x[:, 2]])
    return callback


class TestSettings:
    @pytest.mark.parametrize("change, fragment", [
        ({"displacement_A": 0.5}, "displacement_A"),
        ({"stencil": 3}, "stencil"),
        ({"sum_rule": "both"}, "sum_rule"),
        ({"max_residual_force_eV_A": 0.0}, "max_residual"),
        ({"asr_tolerance": 1.5}, "asr_tolerance"),
        ({"q_point": (0.0, float("nan"), 0.0)}, "finite"),
    ])
    def test_invalid_settings(self, change, fragment):
        with pytest.raises(P.PhononError, match=fragment):
            P.PhononSettings(**change).validate()

    @pytest.mark.parametrize("q", [(0.5, 0.0, 0.0), (0.25, 0.25, 0.0)])
    def test_dispersion_is_refused(self, q):
        with pytest.raises(P.PhononRefused, match="Only the Gamma point"):
            P.PhononSettings(q_point=q).validate()

    def test_equivalent_gamma_points_are_accepted(self):
        assert P.PhononSettings(q_point=(1.0, 0.0, -2.0)).validate()


class TestDimer:
    @pytest.mark.parametrize("stencil, tolerance", [(2, 1e-3), (4, 1e-7)])
    def test_stretch_matches_the_closed_form(self, stencil, tolerance):
        result = P.harmonic_analysis(dimer(), lj(), P.PhononSettings(stencil=stencil))
        mass = dimer().masses()[0]
        k = 36 * 2 ** (2 / 3) * EPS / SIG ** 2
        exact = thz(k / (mass / 2))
        assert result.frequencies_THz[-1] == pytest.approx(exact, rel=tolerance)
        assert result.kinds == ["rigid-body"] * 5 + ["real"]
        zpe = 0.5 * exact * P.THZ_TO_MEV / 1000
        assert result.zero_point_energy_eV == pytest.approx(zpe, rel=tolerance)

    def test_heteronuclear_dimer_uses_the_masses(self):
        params = {"Ar": (EPS, SIG), "Kr": (0.0140, 3.65)}
        eps, sig = math.sqrt(EPS * 0.0140), 0.5 * (SIG + 3.65)
        r0 = 2 ** (1 / 6) * sig
        s = box([18, 36], [[5, 5, 5], [5, 5, 5 + r0]])
        result = P.harmonic_analysis(s, LennardJones(params, cutoff_A=12.0),
                                     P.PhononSettings(stencil=4))
        m1, m2 = s.masses()
        exact = thz(36 * 2 ** (2 / 3) * eps / sig ** 2 * (m1 + m2) / (m1 * m2))
        assert result.frequencies_THz[-1] == pytest.approx(exact, rel=1e-7)
        stretch = result.displacements[-1]
        assert abs(stretch[0, 2]) / abs(stretch[1, 2]) == pytest.approx(m2 / m1, rel=1e-6)

    def test_reduced_mass_identity(self):
        result = P.harmonic_analysis(dimer(), lj(), P.PhononSettings(stencil=4))
        l = result.displacements[-1].ravel()
        curvature = l @ result.force_constants_eV_A2 @ l
        assert result.eigenvalues_eV_A2_u[-1] == pytest.approx(
            curvature / result.reduced_masses_u[-1], rel=1e-10)
        assert result.reduced_masses_u[-1] == pytest.approx(dimer().masses()[0], rel=1e-10)
        assert result.participation[-1] == pytest.approx(1.0, rel=1e-10)

    def test_harmonic_thermodynamics_matches_one_oscillator(self):
        result = P.harmonic_analysis(dimer(), lj(), P.PhononSettings(stencil=4))
        temperatures = np.array([0.0, 300.0, 1.0e7])
        thermo = result.thermodynamics(temperatures)
        energy = result.frequencies_THz[-1] * P.THZ_TO_MEV / 1000.0
        x = energy / (BOLTZMANN_EV_K * temperatures[1])
        expected_free = 0.5 * energy + BOLTZMANN_EV_K * temperatures[1] * math.log(
            -math.expm1(-x))
        expected_internal = 0.5 * energy + energy / math.expm1(x)
        expected_cv = BOLTZMANN_EV_K * x ** 2 * math.exp(x) / math.expm1(x) ** 2
        assert thermo["helmholtz_free_energy_eV"][0] == pytest.approx(0.5 * energy)
        assert thermo["internal_energy_eV"][0] == pytest.approx(0.5 * energy)
        assert thermo["entropy_eV_K"][0] == 0.0
        assert thermo["heat_capacity_eV_K"][0] == 0.0
        assert thermo["helmholtz_free_energy_eV"][1] == pytest.approx(expected_free)
        assert thermo["internal_energy_eV"][1] == pytest.approx(expected_internal)
        assert thermo["heat_capacity_eV_K"][1] == pytest.approx(expected_cv)
        assert thermo["heat_capacity_eV_K"][2] == pytest.approx(
            BOLTZMANN_EV_K, rel=1e-8)

    def test_thermodynamic_inputs_are_validated(self):
        result = P.harmonic_analysis(dimer(), lj())
        for bad in ([], [[100.0]], [-1.0], [float("nan")]):
            with pytest.raises(P.PhononError, match="temperatures_K"):
                result.thermodynamics(bad)

    @pytest.mark.parametrize("rule", P.SUM_RULES)
    def test_every_sum_rule_finds_five_rigid_modes(self, rule):
        result = P.harmonic_analysis(dimer(), lj(), P.PhononSettings(sum_rule=rule))
        assert result.kinds.count("rigid-body") == 5
        removed = {"none": 0, "translational": 3, "translational+rotational": 5}[rule]
        assert result.diagnostics["rigid_body_modes_removed"] == removed


class TestSumRule:
    def test_projection_is_symmetric_and_exact(self):
        s = box([18] * 3, [[5, 5, 5], [5, 5, 5 + R0], [5, 5 + R0, 5.3]])
        s = _relaxed(s)
        result = P.harmonic_analysis(s, lj())
        phi = result.force_constants_eV_A2
        assert np.array_equal(phi, phi.T)
        sums = phi.reshape(9, 3, 3).sum(axis=1)
        assert np.abs(sums).max() < 1e-12 * np.abs(phi).max()

    def test_einstein_solid_refuses_the_sum_rule_and_is_exact_without_it(self):
        s = box([18] * 4, np.random.default_rng(0).uniform(0, 8, (4, 3)), size=8.0,
                pbc=(True, True, True))
        model = Harmonic(s.positions.copy(), k_eV_A2=2.0)
        with pytest.raises(P.PhononRefused, match="not invariant under translation"):
            P.harmonic_analysis(s, model)
        result = P.harmonic_analysis(s, model, P.PhononSettings(sum_rule="none"))
        assert result.kinds == ["real"] * 12
        assert np.allclose(result.frequencies_THz, thz(2.0 / s.masses()[0]), rtol=1e-9)

    def test_rotations_need_an_isolated_system(self):
        s = box([18, 18], [[5, 5, 5], [5, 5, 5 + R0]], pbc=(True, True, True))
        with pytest.raises(P.PhononRefused, match="isolated"):
            P.harmonic_analysis(s, lj(), P.PhononSettings(sum_rule="translational+rotational"))

    def test_rigid_basis_rank(self):
        free = np.ones(3, dtype=bool)
        linear = np.array([[0, 0, 0], [0, 0, 1.0], [0, 0, 2.0]])
        bent = np.array([[0, 0, 0], [0, 0, 1.0], [0, 1.0, 1.0]])
        assert P.rigid_body_basis(linear, free, True).shape[1] == 5
        assert P.rigid_body_basis(bent, free, True).shape[1] == 6
        assert P.rigid_body_basis(bent, free, False).shape[1] == 3


def _relaxed(s):
    from materia.solvers.classical import ClassicalSolver
    return ClassicalSolver(lj()).relax(s, fmax_eV_A=1e-7, max_steps=20000).structure


class TestImaginaryModes:
    def test_saddle_reports_one_imaginary_mode(self):
        s = box([18], [[5, 5, 5]])
        result = P.harmonic_analysis(s, saddle(), P.PhononSettings(sum_rule="none"),
                                     name="analytic saddle", fidelity=Fidelity.NON_PHYSICAL)
        mass = s.masses()[0]
        expected = sorted([thz(1.0 / mass), -thz(0.5 / mass), thz(2.0 / mass)])
        assert np.allclose(result.frequencies_THz, expected, rtol=1e-9)
        assert result.kinds == ["imaginary", "real", "real"]
        assert result.imaginary_modes() == [0]
        assert result.zero_point_energy_eV is None
        assert "not at a minimum" in result.zero_point_note
        zpe = result.results()["zero_point_energy"]
        assert not zpe.supported and "imaginary" in zpe.unsupported_reason
        thermo = result.thermodynamics_results([300.0])
        assert all(not record.supported for record in thermo.values())
        assert all("imaginary" in record.unsupported_reason for record in thermo.values())
        assert np.abs(result.displacements[0, 0]) == pytest.approx([0, 1, 0], abs=1e-9)

    def test_plain_callbacks_must_name_themselves(self):
        with pytest.raises(P.PhononError, match="model name and a fidelity"):
            P.harmonic_analysis(box([18], [[5, 5, 5]]), saddle(),
                                P.PhononSettings(sum_rule="none"))


class TestRefusals:
    def test_nonstationary_is_refused_unless_allowed(self):
        stretched = dimer(R0 + 0.3)
        with pytest.raises(P.PhononRefused, match="stationary point"):
            P.harmonic_analysis(stretched, lj())
        estimate = P.harmonic_analysis(stretched, lj(),
                                       P.PhononSettings(allow_nonstationary=True))
        assert estimate.provenance.origin is Origin.ESTIMATED
        assert not estimate.convergence.converged
        assert estimate.zero_point_energy_eV is None
        assert any("not stationary" in a for a in estimate.provenance.approximations)
        with pytest.raises(P.PhononRefused, match="stationary point"):
            P.harmonic_analysis(stretched, lj(), P.PhononSettings(
                allow_nonstationary=True, sum_rule="translational+rotational"))

    def test_fixed_atoms_give_a_partial_hessian_without_sum_rule(self):
        s = dimer()
        s.fixed[0] = True
        with pytest.raises(P.PhononRefused, match="fixed"):
            P.harmonic_analysis(s, lj())
        result = P.harmonic_analysis(s, lj(), P.PhononSettings(sum_rule="none", stencil=4))
        assert result.free_atom_ids == [int(s.ids[1])]
        k = 36 * 2 ** (2 / 3) * EPS / SIG ** 2
        assert result.frequencies_THz[-1] == pytest.approx(thz(k / s.masses()[1]), rel=1e-7)
        assert any("infinitely heavy" in a for a in result.provenance.approximations)

    def test_every_atom_fixed(self):
        s = dimer()
        s.fixed[:] = True
        with pytest.raises(P.PhononRefused, match="nothing to vibrate"):
            P.harmonic_analysis(s, lj())

    @pytest.mark.parametrize("bad", [lambda x: np.full_like(x, np.nan),
                                     lambda x: np.zeros((1, 3))])
    def test_invalid_force_output(self, bad):
        with pytest.raises(P.PhononError, match="invalid force array"):
            P.harmonic_analysis(box([18, 18], [[5, 5, 5], [5, 5, 9]]), bad,
                                P.PhononSettings(sum_rule="none"), name="bad",
                                fidelity=Fidelity.NON_PHYSICAL)

    def test_unknown_source(self):
        with pytest.raises(P.PhononError, match="force source"):
            P.harmonic_analysis(dimer(), object())


class TestProgressAndRecords:
    def test_progress_and_cancellation(self):
        seen = []
        P.harmonic_analysis(dimer(), lj(), progress=lambda f, m: seen.append(f))
        assert seen[0] == 0.0 and seen[-1] == 1.0 and seen == sorted(seen)
        with pytest.raises(P.PhononCancelled):
            P.harmonic_analysis(dimer(), lj(), progress=lambda f, m: f < 0.5)
        with pytest.raises(P.PhononCancelled):
            P.harmonic_analysis(dimer(), lj(), cancelled=lambda: True)

    def test_force_calls_follow_the_stencil(self):
        two = P.harmonic_analysis(dimer(), lj())
        four = P.harmonic_analysis(dimer(), lj(), P.PhononSettings(stencil=4))
        assert two.diagnostics["force_calls"] == 1 + 2 * 6
        assert four.diagnostics["force_calls"] == 1 + 4 * 6

    def test_provenance_units_and_digest(self):
        result = P.harmonic_analysis(dimer(), lj())
        prov = result.provenance
        assert prov.model == "phonons/finite-displacement[classical/lennard-jones]"
        assert prov.fidelity is Fidelity.TIER1_CLASSICAL
        assert prov.origin is Origin.CALCULATED
        assert prov.parameters["settings"]["displacement_A"] == 0.01
        assert any("Parlinski" in r for r in prov.references)
        other = P.harmonic_analysis(dimer(), lj(), P.PhononSettings(displacement_A=0.02))
        assert other.provenance.inputs_digest != prov.inputs_digest
        results = result.results()
        assert results["frequencies"].unit == "THz"
        assert results["force_constants"].unit == "eV/A^2"
        assert results["zero_point_energy"].unit == "eV"
        modes = results["frequencies"].extra["modes"]
        assert modes[-1]["frequency_cm1"] == pytest.approx(
            modes[-1]["frequency_THz"] * P.THZ_TO_CM1)
        assert P.THZ_TO_CM1 == pytest.approx(33.35641, rel=1e-6)
        assert P.THZ_TO_MEV == pytest.approx(4.135667696, rel=1e-9)
        assert results["frequencies"].uncertainty_kind == "bound"

    def test_stored_arrays_and_dict(self):
        result = P.harmonic_analysis(dimer(), lj())
        arrays = result.stored_arrays()
        assert set(arrays) == {"force_constants", "frequencies", "eigenvectors"}
        assert all(isinstance(a, StoredArray) for a in arrays.values())
        assert arrays["frequencies"].unit == "THz"
        assert arrays["frequencies"].meta["kinds"] == result.kinds
        record = result.as_dict()
        assert len(record["modes"]) == 6 and record["provenance"]["origin"] == "calculated"

    def test_isolated_record_has_no_periodic_limitations(self):
        result = P.harmonic_analysis(dimer(), lj())
        assert result.diagnostics["geometry"] == "isolated"
        assert not any("Gamma point" in a for a in result.provenance.approximations)
