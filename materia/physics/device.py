"""Semiconductor device simulation through DEVSIM: a one-dimensional p-n junction.

DEVSIM (Apache-2.0) solves the Poisson and drift-diffusion equations by the
finite-volume method with Scharfetter-Gummel fluxes.  Materia builds the mesh,
doping and equations with DEVSIM's own ``simple_physics`` model builders, but
replaces every material parameter with a stated value:

====================  ===========================  ====================================
Quantity              Value                        Source
====================  ===========================  ====================================
elementary charge     1.602176634e-19 C            SI definition (exact)
Boltzmann constant    1.380649e-23 J/K             SI definition (exact)
vacuum permittivity   8.8541878128e-14 F/cm        CODATA 2018
silicon permittivity  11.7                         S. M. Sze and K. K. Ng, Physics of
                                                   Semiconductor Devices, 3rd ed. (2007)
intrinsic density     9.65e9 cm^-3 at 300 K        Sze and Ng (2007)
electron mobility     1417 cm^2/Vs                 Sze and Ng (2007), lightly doped Si
hole mobility         470 cm^2/Vs                  Sze and Ng (2007), lightly doped Si
====================  ===========================  ====================================

Mobilities are constant (no doping or field dependence), recombination is
Shockley-Read-Hall through a midgap trap with the given lifetimes, there is
no band-gap narrowing, no Auger recombination and no incomplete ionisation,
and the contacts are ideal ohmic.  Nondegenerate (Boltzmann) statistics.

Outputs use cm for lengths in device units, as is conventional, and report
micrometres where stated.
"""

from __future__ import annotations

import importlib
import itertools
import math
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..provenance import Fidelity, Origin, Provenance, Result

Q_C = 1.602176634e-19
K_J_K = 1.380649e-23
EPS0_F_CM = 8.8541878128e-14
SILICON = {"permittivity": 11.7, "n_i_cm3": 9.65e9, "mu_n_cm2_Vs": 1417.0,
           "mu_p_cm2_Vs": 470.0}
SOURCE = "S. M. Sze and K. K. Ng, Physics of Semiconductor Devices, 3rd ed. (Wiley, 2007)"
REFERENCES = [
    "J. E. Sanchez, DEVSIM, https://devsim.org (Apache-2.0)",
    "D. L. Scharfetter and H. K. Gummel, IEEE Trans. Electron Devices 16 (1969) 64",
    SOURCE,
]
_counter = itertools.count()


class DeviceError(ValueError):
    """The device request is invalid or the solver did not converge."""


def _devsim():
    try:
        devsim = importlib.import_module("devsim")
        physics = importlib.import_module("devsim.python_packages.simple_physics")
        models = importlib.import_module("devsim.python_packages.model_create")
    except ImportError:
        raise DeviceError("DEVSIM is not installed: pip install devsim") from None
    except RuntimeError as exc:
        raise DeviceError(f"DEVSIM is installed but could not start ({exc}). It needs a "
                          "BLAS and LAPACK library; on Windows and Linux install one with "
                          "pip install mkl.") from None
    return devsim, physics, models


def analytic_junction(acceptors_cm3: float, donors_cm3: float, temperature_K: float = 300.0,
                      bias_V: float = 0.0) -> Dict[str, float]:
    """Built-in potential, depletion widths and peak field of an abrupt junction.

    Depletion approximation: ``V_bi = (kT/q) ln(N_A N_D / n_i^2)``,
    ``W = sqrt(2 eps (V_bi - V) (1/N_A + 1/N_D) / q)``, ``E_max = q N_A x_p / eps``.
    """
    vt = K_J_K * temperature_K / Q_C
    eps = SILICON["permittivity"] * EPS0_F_CM
    ni = SILICON["n_i_cm3"]
    vbi = vt * math.log(acceptors_cm3 * donors_cm3 / ni ** 2)
    width = math.sqrt(2 * eps * (vbi - bias_V) * (1 / acceptors_cm3 + 1 / donors_cm3) / Q_C)
    xp = width * donors_cm3 / (acceptors_cm3 + donors_cm3)
    return {"built_in_V": vbi, "depletion_width_um": width * 1e4, "x_p_um": xp * 1e4,
            "x_n_um": (width - xp) * 1e4, "peak_field_V_cm": Q_C * acceptors_cm3 * xp / eps,
            "thermal_voltage_V": vt}


def short_diode_saturation_A_cm2(acceptors_cm3: float, donors_cm3: float, p_width_um: float,
                                 n_width_um: float, temperature_K: float = 300.0) -> float:
    """Shockley saturation current of a short-base diode with ohmic contacts.

    ``J0 = q n_i^2 (D_n / (W_p N_A) + D_p / (W_n N_D))`` with ``D = mu kT/q`` and
    the neutral widths ``W`` in place of diffusion lengths.
    """
    vt = K_J_K * temperature_K / Q_C
    ni = SILICON["n_i_cm3"]
    dn, dp = SILICON["mu_n_cm2_Vs"] * vt, SILICON["mu_p_cm2_Vs"] * vt
    return Q_C * ni ** 2 * (dn / (p_width_um * 1e-4 * acceptors_cm3)
                            + dp / (n_width_um * 1e-4 * donors_cm3))


def pn_junction(acceptors_cm3: float = 1e17, donors_cm3: float = 1e16,
                length_um: float = 2.0, junction_um: Optional[float] = None,
                biases_V: Sequence[float] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5),
                temperature_K: float = 300.0, lifetime_s: float = 1e-6,
                spacing_nm: float = 1.0, generation_cm3_s: float = 0.0,
                photon_flux_cm2_s: float = 0.0,
                absorption_per_cm: Optional[float] = None) -> Dict[str, Result]:
    """Equilibrium profiles and the current-voltage curve of an abrupt silicon junction.

    The p side (acceptors) is at the anode, the n side at the grounded
    cathode; positive bias is forward.  Returns potential, field, carrier and
    charge profiles at equilibrium, and the current density at every bias.

    Optical generation adds to both continuity equations: a uniform rate
    ``generation_cm3_s``, and/or Beer-Lambert absorption of a photon flux
    entering at the anode, ``G(x) = Phi alpha exp(-alpha x)``, one pair per
    absorbed photon and no reflection.  The equilibrium profile is always the
    dark one.
    """
    devsim, physics, models = _devsim()
    if not (1e13 <= acceptors_cm3 <= 1e19 and 1e13 <= donors_cm3 <= 1e19):
        raise DeviceError("Dopings must lie between 1e13 and 1e19 cm^-3; above that the "
                          "nondegenerate statistics used here fail.")
    if temperature_K != 300.0:
        raise DeviceError("Only 300 K is parameterised: n_i and mobilities are 300 K values.")
    junction_um = length_um / 2 if junction_um is None else junction_um
    if not 0 < junction_um < length_um:
        raise DeviceError("The junction must lie inside the device.")
    if lifetime_s <= 0 or spacing_nm <= 0:
        raise DeviceError("lifetime_s and spacing_nm must be positive.")
    if generation_cm3_s < 0 or photon_flux_cm2_s < 0:
        raise DeviceError("Generation rates and photon fluxes cannot be negative.")
    if photon_flux_cm2_s > 0 and not (absorption_per_cm and absorption_per_cm > 0):
        raise DeviceError("A photon flux needs a positive absorption coefficient.")
    illuminated = generation_cm3_s > 0 or photon_flux_cm2_s > 0
    biases = [float(v) for v in biases_V]
    if any(v > 0.8 for v in biases):
        raise DeviceError("Forward bias above 0.8 V reaches high injection, outside the "
                          "low-level model described here.")
    name = f"materia_pn_{next(_counter)}"
    mesh, region = f"{name}_mesh", "bulk"
    length_cm, xj_cm = length_um * 1e-4, junction_um * 1e-4
    fine = spacing_nm * 1e-7
    coarse = min(length_cm / 50, 20 * fine)
    devsim.create_1d_mesh(mesh=mesh)
    devsim.add_1d_mesh_line(mesh=mesh, pos=0.0, ps=coarse, tag="anode")
    devsim.add_1d_mesh_line(mesh=mesh, pos=xj_cm, ps=fine)
    devsim.add_1d_mesh_line(mesh=mesh, pos=length_cm, ps=coarse, tag="cathode")
    devsim.add_1d_contact(mesh=mesh, name="anode", tag="anode", material="metal")
    devsim.add_1d_contact(mesh=mesh, name="cathode", tag="cathode", material="metal")
    devsim.add_1d_region(mesh=mesh, material="Si", region=region, tag1="anode",
                         tag2="cathode")
    devsim.finalize_mesh(mesh=mesh)
    devsim.create_device(mesh=mesh, device=name)
    try:
        physics.SetSiliconParameters(name, region, temperature_K)
        vt = K_J_K * temperature_K / Q_C
        for key, value in (("Permittivity", SILICON["permittivity"] * EPS0_F_CM),
                           ("ElectronCharge", Q_C), ("n_i", SILICON["n_i_cm3"]),
                           ("kT", K_J_K * temperature_K), ("V_t", vt),
                           ("mu_n", SILICON["mu_n_cm2_Vs"]), ("mu_p", SILICON["mu_p_cm2_Vs"]),
                           ("n1", SILICON["n_i_cm3"]), ("p1", SILICON["n_i_cm3"]),
                           ("taun", lifetime_s), ("taup", lifetime_s)):
            devsim.set_parameter(device=name, region=region, name=key, value=value)
        models.CreateNodeModel(name, region, "Acceptors", f"{acceptors_cm3:.10e}*step({xj_cm}-x)")
        models.CreateNodeModel(name, region, "Donors", f"{donors_cm3:.10e}*step(x-{xj_cm})")
        models.CreateNodeModel(name, region, "NetDoping", "Donors-Acceptors")
        physics.CreateSiliconPotentialOnly(name, region)
        for contact in ("anode", "cathode"):
            physics.CreateSiliconPotentialOnlyContact(name, region, contact)
            devsim.set_parameter(device=name, name=physics.GetContactBiasName(contact),
                                 value=0.0)
        _solve(devsim)
        x = np.array(devsim.get_node_model_values(device=name, region=region, name="x"))
        potential = np.array(devsim.get_node_model_values(device=name, region=region,
                                                          name="Potential"))
        physics.CreateSolution(name, region, "Electrons")
        physics.CreateSolution(name, region, "Holes")
        devsim.set_node_values(device=name, region=region, name="Electrons",
                               init_from="IntrinsicElectrons")
        devsim.set_node_values(device=name, region=region, name="Holes",
                               init_from="IntrinsicHoles")
        physics.CreateSiliconDriftDiffusion(name, region)
        for contact in ("anode", "cathode"):
            physics.CreateSiliconDriftDiffusionAtContact(name, region, contact)
        _solve(devsim)
        electrons = np.array(devsim.get_node_model_values(device=name, region=region,
                                                          name="Electrons"))
        holes = np.array(devsim.get_node_model_values(device=name, region=region, name="Holes"))
        doping = np.array(devsim.get_node_model_values(device=name, region=region,
                                                       name="NetDoping"))
        potential = np.array(devsim.get_node_model_values(device=name, region=region,
                                                          name="Potential"))
        field = -np.gradient(potential, x)
        dark_currents = None

        def terminal_current() -> float:
            electron = devsim.get_contact_current(device=name, contact="cathode",
                                                  equation=physics.ece_name)
            hole = devsim.get_contact_current(device=name, contact="cathode",
                                              equation=physics.hce_name)
            return -(electron + hole)

        resolution = abs(terminal_current())

        def sweep() -> List[float]:
            out = []
            for bias in biases:
                devsim.set_parameter(device=name, name=physics.GetContactBiasName("anode"),
                                     value=bias)
                _solve(devsim)
                out.append(terminal_current())
            devsim.set_parameter(device=name, name=physics.GetContactBiasName("anode"),
                                 value=0.0)
            _solve(devsim)
            return out

        currents = sweep()
        generated = 0.0
        if illuminated:
            dark_currents = currents
            terms = []
            if generation_cm3_s > 0:
                terms.append(f"{generation_cm3_s:.12e}")
            if photon_flux_cm2_s > 0:
                terms.append(f"{photon_flux_cm2_s * absorption_per_cm:.12e}"
                             f"*exp(-{absorption_per_cm:.12e}*x)")
            expression = " + ".join(terms)
            models.CreateNodeModel(name, region, "OpticalGeneration", f"0*({expression})")
            models.CreateNodeModel(name, region, "ElectronGeneration",
                                   "-ElectronCharge * (USRH - OpticalGeneration)")
            models.CreateNodeModel(name, region, "HoleGeneration",
                                   "+ElectronCharge * (USRH - OpticalGeneration)")
            for fraction in (0.01, 0.1, 0.3, 1.0):
                models.CreateNodeModel(name, region, "OpticalGeneration",
                                       f"{fraction:.6g}*({expression})")
                _solve(devsim)
            g = np.array(devsim.get_node_model_values(device=name, region=region,
                                                      name="OpticalGeneration"))
            generated = Q_C * float(np.trapezoid(g, x))
            currents = sweep()
    finally:
        devsim.delete_device(device=name)
        devsim.delete_mesh(mesh=mesh)
    analytic = analytic_junction(acceptors_cm3, donors_cm3, temperature_K)
    charge = Q_C * (holes - electrons + doping)
    params = {"acceptors_cm3": acceptors_cm3, "donors_cm3": donors_cm3,
              "length_um": length_um, "junction_um": junction_um,
              "temperature_K": temperature_K, "lifetime_s": lifetime_s,
              "spacing_nm": spacing_nm, "silicon": dict(SILICON),
              "generation_cm3_s": generation_cm3_s, "photon_flux_cm2_s": photon_flux_cm2_s,
              "absorption_per_cm": absorption_per_cm,
              "engine": "DEVSIM", "engine_version": getattr(devsim, "__version__", "unknown")}
    prov = Provenance(
        model="devsim/drift-diffusion-1d", fidelity=Fidelity.TIER2_SEMI_EMPIRICAL,
        origin=Origin.CALCULATED,
        approximations=["Poisson and drift-diffusion with Scharfetter-Gummel fluxes, solved by "
                        "DEVSIM; Boltzmann statistics; constant mobilities; midgap SRH "
                        "recombination; no band-gap narrowing, Auger recombination or "
                        "incomplete ionisation; ideal ohmic contacts; one dimension."],
        parameters=params, references=list(REFERENCES),
        boundary_conditions="ohmic anode (p side) and cathode (n side), cathode grounded")
    profile = {"x_um": (x * 1e4).tolist(), "potential_V": potential.tolist(),
               "field_V_cm": field.tolist(), "electrons_cm3": electrons.tolist(),
               "holes_cm3": holes.tolist(), "net_doping_cm3": doping.tolist(),
               "charge_C_cm3": charge.tolist()}
    built_in = float(potential[-1] - potential[0])
    return {
        "profile": Result("equilibrium_profile", profile, "mixed", prov,
                          extra={"units": {"x_um": "um", "potential_V": "V",
                                           "field_V_cm": "V/cm", "electrons_cm3": "cm^-3",
                                           "holes_cm3": "cm^-3", "charge_C_cm3": "C/cm^3"}}),
        "built_in_potential": Result("built_in_potential", built_in, "V", prov,
                                     extra={"analytic_V": analytic["built_in_V"]}),
        "peak_field": Result("peak_field", float(np.abs(field).max()), "V/cm", prov,
                             extra={"depletion_approximation_V_cm": analytic["peak_field_V_cm"]}),
        "iv": Result("current_voltage", {"bias_V": biases, "current_A_cm2": currents},
                     "mixed", prov,
                     extra={"resolution_A_cm2": resolution,
                            "dark_current_A_cm2": dark_currents,
                            "generated_current_A_cm2": generated,
                            "illuminated": illuminated,
                            "resolution_note": "The current at zero bias, which is zero in "
                            "exact arithmetic; it measures the cancellation error of drift "
                            "and diffusion terms. Currents below it are not resolved.",
                            "short_diode_J0_A_cm2": short_diode_saturation_A_cm2(
                         acceptors_cm3, donors_cm3, junction_um - analytic["x_p_um"],
                         length_um - junction_um - analytic["x_n_um"], temperature_K)}),
    }


def solar_metrics(iv: Result, incident_power_W_cm2: Optional[float] = None) -> Dict[str, float]:
    """Short-circuit current, open-circuit voltage, maximum power and fill factor.

    From an illuminated current-voltage result of :func:`pn_junction`.  The
    open-circuit voltage is interpolated between the two sampled biases that
    bracket zero current, linearly in ``ln(J + J_sc)``, which is exact for an
    exponential diode and a constant photocurrent; the maximum power is the largest
    sampled ``-J V``: sample densely near the knee for a precise fill factor.
    """
    bias = np.asarray(iv.value["bias_V"], dtype=float)
    current = np.asarray(iv.value["current_A_cm2"], dtype=float)
    if not iv.extra.get("illuminated"):
        raise DeviceError("The device was not illuminated.")
    if bias[0] != 0.0 or np.any(np.diff(bias) <= 0):
        raise DeviceError("The biases must start at 0 V and increase.")
    jsc = -float(current[0])
    crossing = np.flatnonzero((current[:-1] < 0) & (current[1:] >= 0))
    if jsc <= 0 or not len(crossing):
        raise DeviceError("The current does not change sign in the sampled biases; extend "
                          "them past the open-circuit voltage.")
    k = int(crossing[0])
    low, high = current[k] + jsc, current[k + 1] + jsc
    if low > 0 and high > low:
        step = (math.log(jsc) - math.log(low)) / (math.log(high) - math.log(low))
    else:
        step = -current[k] / (current[k + 1] - current[k])
    voc = float(bias[k] + step * (bias[k + 1] - bias[k]))
    power = -current * bias
    best = int(np.argmax(power))
    out = {"short_circuit_current_A_cm2": jsc, "open_circuit_voltage_V": voc,
           "max_power_W_cm2": float(power[best]), "max_power_bias_V": float(bias[best]),
           "fill_factor": float(power[best]) / (jsc * voc)}
    if incident_power_W_cm2:
        out["efficiency"] = float(power[best]) / incident_power_W_cm2
    return out


def _solve(devsim) -> None:
    try:
        devsim.solve(type="dc", absolute_error=1e10, relative_error=1e-10,
                     maximum_iterations=60)
    except devsim.error as exc:
        raise DeviceError(f"DEVSIM did not converge: {exc}") from None


SIO2_PERMITTIVITY = 3.9


def exact_surface_charge(band_bending_V: float, acceptors_cm3: float,
                         temperature_K: float = 300.0) -> float:
    """Charge per area in p-type silicon for a surface band bending, Boltzmann statistics.

    ``Q_s = -sign(phi) sqrt(2 eps kT p0) sqrt[(e^-bphi + bphi - 1) + (n0/p0)(e^bphi - bphi - 1)]``
    in C/cm^2, with ``b = q/kT`` (Sze and Ng 2007, chapter 4).
    """
    eps = SILICON["permittivity"] * EPS0_F_CM
    kt = K_J_K * temperature_K
    ni = SILICON["n_i_cm3"]
    p0 = 0.5 * (acceptors_cm3 + math.sqrt(acceptors_cm3 ** 2 + 4 * ni ** 2))
    n0 = ni ** 2 / p0
    b = band_bending_V * Q_C / kt
    inner = (math.exp(-b) + b - 1) + (n0 / p0) * (math.exp(b) - b - 1)
    return -math.copysign(1.0, b) * math.sqrt(2 * eps * kt * p0) * math.sqrt(max(inner, 0.0))


def exact_gate_voltage(band_bending_V: float, acceptors_cm3: float, oxide_nm: float,
                       temperature_K: float = 300.0) -> float:
    """Gate bias for a band bending, with the gate referenced to the intrinsic level.

    ``V_g = psi_bulk + phi_s - Q_s / C_ox``, ``psi_bulk = -(kT/q) ln(p0/n_i)``; this is the
    reference DEVSIM's oxide contact uses, so no work-function difference enters.
    """
    vt = K_J_K * temperature_K / Q_C
    ni = SILICON["n_i_cm3"]
    p0 = 0.5 * (acceptors_cm3 + math.sqrt(acceptors_cm3 ** 2 + 4 * ni ** 2))
    cox = SIO2_PERMITTIVITY * EPS0_F_CM / (oxide_nm * 1e-7)
    return -vt * math.log(p0 / ni) + band_bending_V - exact_surface_charge(
        band_bending_V, acceptors_cm3, temperature_K) / cox


def mos_capacitor(acceptors_cm3: float = 1e17, oxide_nm: float = 10.0,
                  gate_V: Sequence[float] = tuple(np.round(np.arange(-2.0, 2.01, 0.1), 3)),
                  silicon_um: float = 1.0, temperature_K: float = 300.0,
                  spacing_nm: float = 0.1) -> Dict[str, Result]:
    """Equilibrium (low-frequency) gate charge and capacitance of a p-type MOS capacitor.

    Silicon and the SiO2 layer (permittivity 3.9) are solved together with
    DEVSIM; carriers follow the bulk Fermi level, so the inversion layer
    responds and the result is the quasi-static capacitance.  The gate is
    referenced to the silicon intrinsic level (no work-function difference) and
    the oxide has no fixed or interface charge.
    """
    devsim, physics, models = _devsim()
    if not 1e14 <= acceptors_cm3 <= 1e18:
        raise DeviceError("Acceptor doping must lie between 1e14 and 1e18 cm^-3.")
    if temperature_K != 300.0:
        raise DeviceError("Only 300 K is parameterised.")
    if not 1.0 <= oxide_nm <= 200.0:
        raise DeviceError("oxide_nm must lie between 1 and 200 nm.")
    gates = [float(v) for v in gate_V]
    if len(gates) < 3 or any(b <= a for a, b in zip(gates, gates[1:])):
        raise DeviceError("gate_V must be at least three increasing values.")
    name = f"materia_mos_{next(_counter)}"
    mesh = f"{name}_mesh"
    tox, tsi, fine = oxide_nm * 1e-7, silicon_um * 1e-4, spacing_nm * 1e-7
    devsim.create_1d_mesh(mesh=mesh)
    devsim.add_1d_mesh_line(mesh=mesh, pos=-tox, ps=tox / 20, tag="gate")
    devsim.add_1d_mesh_line(mesh=mesh, pos=0.0, ps=fine, tag="interface")
    devsim.add_1d_mesh_line(mesh=mesh, pos=tsi, ps=tsi / 40, tag="body")
    devsim.add_1d_contact(mesh=mesh, name="gate", tag="gate", material="metal")
    devsim.add_1d_contact(mesh=mesh, name="body", tag="body", material="metal")
    devsim.add_1d_interface(mesh=mesh, name="interface", tag="interface")
    devsim.add_1d_region(mesh=mesh, material="Ox", region="oxide", tag1="gate",
                         tag2="interface")
    devsim.add_1d_region(mesh=mesh, material="Si", region="silicon", tag1="interface",
                         tag2="body")
    devsim.finalize_mesh(mesh=mesh)
    devsim.create_device(mesh=mesh, device=name)
    try:
        physics.SetOxideParameters(name, "oxide", temperature_K)
        devsim.set_parameter(device=name, region="oxide", name="Permittivity",
                             value=SIO2_PERMITTIVITY * EPS0_F_CM)
        devsim.set_parameter(device=name, region="oxide", name="ElectronCharge", value=Q_C)
        physics.SetSiliconParameters(name, "silicon", temperature_K)
        vt = K_J_K * temperature_K / Q_C
        for key, value in (("Permittivity", SILICON["permittivity"] * EPS0_F_CM),
                           ("ElectronCharge", Q_C), ("n_i", SILICON["n_i_cm3"]),
                           ("kT", K_J_K * temperature_K), ("V_t", vt)):
            devsim.set_parameter(device=name, region="silicon", name=key, value=value)
        models.CreateNodeModel(name, "silicon", "NetDoping", f"-{acceptors_cm3:.10e}")
        physics.CreateOxidePotentialOnly(name, "oxide", "log_damp")
        physics.CreateSiliconPotentialOnly(name, "silicon")
        physics.CreateOxideContact(name, "oxide", "gate")
        physics.CreateSiliconPotentialOnlyContact(name, "silicon", "body")
        physics.CreateSiliconOxideInterface(name, "interface")
        devsim.set_parameter(device=name, name=physics.GetContactBiasName("body"), value=0.0)
        start = min(gates, key=abs)
        results = {}
        sweep = [start] + [g for g in gates if g > start] + \
            [g for g in reversed(gates) if g < start]
        for gate in sweep:
            devsim.set_parameter(device=name, name=physics.GetContactBiasName("gate"),
                                 value=gate)
            _solve(devsim)
            charge = devsim.get_contact_charge(device=name, contact="gate",
                                               equation="PotentialEquation")
            psi = devsim.get_node_model_values(device=name, region="silicon",
                                               name="Potential")
            x = devsim.get_node_model_values(device=name, region="silicon", name="x")
            results[gate] = (charge, psi[int(np.argmin(np.abs(np.asarray(x))))], psi[-1])
            if gate == max(gates):
                devsim.set_parameter(device=name, name=physics.GetContactBiasName("gate"),
                                     value=start)
                _solve(devsim)
    finally:
        devsim.delete_device(device=name)
        devsim.delete_mesh(mesh=mesh)
    charge = np.array([results[g][0] for g in gates])
    band_bending = np.array([results[g][1] - results[g][2] for g in gates])
    capacitance = np.gradient(charge, np.array(gates))
    cox = SIO2_PERMITTIVITY * EPS0_F_CM / tox
    exact = []
    for gate in gates:
        exact.append(_exact_charge_at(gate, acceptors_cm3, oxide_nm, temperature_K))
    exact = np.array(exact)
    prov = Provenance(
        model="devsim/mos-capacitor-1d", fidelity=Fidelity.TIER2_SEMI_EMPIRICAL,
        origin=Origin.CALCULATED,
        approximations=["Poisson equation in silicon and SiO2 with equilibrium Boltzmann "
                        "carriers: the quasi-static (low-frequency) response, inversion "
                        "included.",
                        "Gate referenced to the silicon intrinsic level (no work-function "
                        "difference), no oxide or interface charge, no quantum confinement "
                        "or polysilicon depletion."],
        parameters={"acceptors_cm3": acceptors_cm3, "oxide_nm": oxide_nm,
                    "oxide_permittivity": SIO2_PERMITTIVITY, "silicon": dict(SILICON),
                    "temperature_K": temperature_K, "spacing_nm": spacing_nm,
                    "engine": "DEVSIM",
                    "engine_version": getattr(devsim, "__version__", "unknown")},
        references=list(REFERENCES))
    return {
        "cv": Result("capacitance_voltage",
                     {"gate_V": gates, "gate_charge_C_cm2": charge.tolist(),
                      "capacitance_F_cm2": capacitance.tolist(),
                      "band_bending_V": band_bending.tolist()}, "mixed", prov,
                     extra={"oxide_capacitance_F_cm2": cox,
                            "exact_gate_charge_C_cm2": (-exact).tolist(),
                            "flat_band_V": exact_gate_voltage(0.0, acceptors_cm3, oxide_nm,
                                                              temperature_K)}),
    }


def _exact_charge_at(gate_V: float, acceptors_cm3: float, oxide_nm: float,
                     temperature_K: float) -> float:
    from scipy.optimize import brentq

    phi = brentq(lambda p: exact_gate_voltage(p, acceptors_cm3, oxide_nm, temperature_K)
                 - gate_V, -3.0, 3.0, xtol=1e-14)
    return exact_surface_charge(phi, acceptors_cm3, temperature_K)
