# X-ray diffraction and attenuation

`materia.physics.xray` uses XrayDB (M. Newville, MIT licence, `pip install
xraydb`) for reference data: Waasmaier-Kirfel form factors, Chantler anomalous
scattering factors, and Elam cross sections, edges and emission lines. Nothing
is copied into Materia; without the package every call is refused.

## Powder diffraction

`powder_pattern(crystal, source="CuKa1", two_theta_range=(10, 120))` returns
every reflection in range with its Miller indices, 2-theta, d spacing,
multiplicity, |F|^2, Lorentz-polarisation factor, integrated intensity and
intensity relative to the strongest peak. Options are anomalous dispersion at
the source energy (on by default), isotropic Debye-Waller B factors per
element, and site occupancies for disordered sites. Systematic absences are
counted and left out. `profile(peaks, grid, fwhm_deg, eta)` broadens the
peaks with area-preserving pseudo-Voigt functions.

The model is kinematic, from independent spherical atoms. It includes no
absorption, extinction, preferred orientation, instrument function or
monochromator polarisation.

## Attenuation, edges and lines

`attenuation(formula or crystal, energies_eV, density, thickness_um)` gives
the mass and linear attenuation coefficients (Elam total cross section), the
attenuation length and the transmission. A crystal supplies its own formula
and density. `edges_and_lines(element)` lists absorption edges with
fluorescence yields and jump ratios, and emission lines with relative
intensities.

## Validation

- Silicon: reflections, multiplicities and Bragg angles are exact, and the
  diamond-glide absences (200) and (222) are absent. Patterns are unchanged
  by an origin shift or a doubled cell. A Debye-Waller factor damps each
  reflection by exactly exp(-2 B s^2).
- Rock salt: (200) is the strongest line, and the odd reflections are weak
  (f_Na - f_Cl), as observed.
- Attenuation: water at 100 keV gives 0.1707 cm^2/g and lead at 100 keV
  gives 5.549 cm^2/g, equal to NIST XCOM.
- Cu K edge 8979 eV and Ka1 8047 eV.
- Parity with pymatgen 2026.9.24 `XRDCalculator` (MIT), run in its own
  environment, for Si, NaCl and wurtzite ZnO with Cu Ka1: the same reflections
  at the same angles (to 1e-6 degrees), and relative intensities within 1.5%
  (0.15 absolute for the weakest lines). The remaining difference comes from
  the two codes' form-factor fits. Anomalous dispersion is switched off for
  this comparison because pymatgen omits it.
