# Semiconductor device simulation (DEVSIM)

Materia drives DEVSIM (Apache-2.0) to solve the Poisson and drift-diffusion
equations of a one-dimensional abrupt silicon p-n junction. DEVSIM performs
the finite-volume solution with Scharfetter-Gummel fluxes; Materia builds the
device and sets every material parameter from a stated source.

Code: `materia/physics/device.py`. Tests: `tests/unit/test_device.py`.

```python
from materia.physics.device import pn_junction
out = pn_junction(acceptors_cm3=1e17, donors_cm3=1e16, length_um=2.0,
                  biases_V=[0.0, 0.3, 0.5], lifetime_s=1e-6)
out["built_in_potential"].value       # V
out["profile"].value                  # potential, field, carriers, charge
out["iv"].value                       # bias and current density
```

## Parameters

DEVSIM's helper module uses rounded teaching constants (q = 1.6e-19 C,
permittivity 11.1, n_i = 1e10 cm^-3). Materia replaces all of them:
q and k exact SI values, permittivity 11.7, n_i = 9.65e9 cm^-3 at 300 K,
constant mobilities 1417 and 470 cm^2/Vs (Sze and Ng 2007), midgap SRH
recombination with the given lifetime.

## Checked

| Quantity | DEVSIM through Materia | Reference |
| --- | --- | --- |
| Built-in potential, 1e17 / 1e16 cm^-3 | 0.775686 V | (kT/q) ln(N_A N_D / n_i^2) = 0.775686 V |
| Peak field | 45.3 kV/cm | depletion approximation 46.7 kV/cm, which neglects Debye tails |
| Net charge integrated over the device | 3e-16 C/cm^2 | 0 |
| Ideality factor, 0.4 to 0.5 V | 1.010 | 1 for diffusion current |
| Current density at 0.5 V | 0.068 A/cm^2 | short-base Shockley 0.079 A/cm^2 with zero-bias neutral widths |
| Reverse current at -0.5 V | -1.1e-8 A/cm^2 | of the order of SRH generation q n_i W / 2 tau |

Every run reports a current resolution: the computed current at zero bias,
zero in exact arithmetic, here about 4e-10 A/cm^2 from the cancellation of
drift and diffusion terms. Currents below it are not resolved.

## MOS capacitor

```python
from materia.physics.device import mos_capacitor
cv = mos_capacitor(acceptors_cm3=1e17, oxide_nm=10.0)["cv"]
cv.value["gate_V"], cv.value["capacitance_F_cm2"], cv.value["band_bending_V"]
```

p-type silicon under a 10 nm SiO2 layer (permittivity 3.9) is solved with
equilibrium carriers: the quasi-static, low-frequency C-V curve with
inversion. The gate is referenced to the silicon intrinsic level, so the
flat-band voltage is the bulk potential, -0.418 V at 1e17 cm^-3; there is no
work-function difference, oxide charge, interface traps, quantum confinement
or polysilicon depletion.

| Check | Result |
| --- | --- |
| Gate charge against the exact one-dimensional Boltzmann solution Q_s(phi_s) mapped through V_g = psi_b + phi_s - Q_s / C_ox (Sze and Ng 2007, ch. 4) | within 7.3e-5 relative from -2 to +2 V |
| Capacitance | 0.96 C_ox in accumulation and strong inversion, minimum 0.26 C_ox |

## Illuminated junctions and solar-cell metrics

`pn_junction(..., generation_cm3_s=G)` adds a uniform optical generation rate
to both continuity equations. `photon_flux_cm2_s` with `absorption_per_cm`
adds Beer-Lambert absorption of light entering at the anode, one pair per
absorbed photon and no reflection. Generation is ramped in so Newton
converges. The dark curve is kept beside the illuminated one.
`solar_metrics(iv)` returns J_sc, V_oc, the maximum sampled power, the fill
factor and, given the incident power, the efficiency. V_oc is interpolated in
`ln(J + J_sc)`, which is exact for an exponential diode with a constant
photocurrent.

| Check (1e17/1e16 cm^-3, 2 um, lifetime 1 ms, G = 1e19 cm^-3 s^-1) | Result |
| --- | --- |
| Generated current against q G L | equal to 1e-6 |
| Photocurrent against the short-base result q G (L + W(V)) / 2 (ohmic contacts collect half of the neutral-region carriers) | within 0.7% from 0 to 0.45 V, including the drop as W shrinks with forward bias |
| Dark current at V_oc against the photocurrent there | equal within 5% |
| Beer-Lambert generation against q Phi (1 - exp(-alpha L)) | equal to 1e-3; collected current below it |

## Limitations and refusals

* One dimension, silicon, 300 K only.
* Boltzmann statistics: dopings above 1e19 cm^-3 are refused.
* Low-level injection: forward bias above 0.8 V is refused.
* Constant mobilities; no band-gap narrowing, Auger recombination,
  incomplete ionisation, tunnelling or impact ionisation.
* No high-frequency C-V, MOSFETs, heterojunctions or transient simulation yet.
* Illumination is monochromatic or uniform; no solar spectrum, reflection or
  optical interference.
