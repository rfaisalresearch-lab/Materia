# Ideal-gas thermochemistry

`materia.physics.thermochemistry.ideal_gas(molecule, electronic_energy_eV,
frequencies_cm1, temperature_K, pressure_Pa, spin_multiplicity)` returns the
enthalpy, entropy (with translational, rotational, vibrational and electronic
terms), Gibbs energy, zero-point energy and C_P of one molecule in the
rigid-rotor, harmonic-oscillator, ideal-gas approximation.

The rotational symmetry number comes from the point group PySCF detects
(`symmetry_number`), or can be given. Linear molecules are recognised from
their principal moments. Frequencies can come from
`molecular.vibrational_spectra` (PySCF) or from `phonons.harmonic_analysis`
with any Materia potential (xTB, MACE, OpenMM).

Refused: imaginary or zero frequencies (not a minimum), a number of
frequencies other than 3N - 6 (3N - 5 if linear), and periodic structures.

## Validation

- Parity with ASE `IdealGasThermo` for H2O, CO2 and CH4 at 500 K and 2 bar:
  entropy to 1e-6 relative, enthalpy and Gibbs energy to 1e-6 eV.
- B3LYP/def2-TZVP geometries and frequencies against NIST-JANAF at 298.15 K
  and 1 bar:

| Molecule | Point group, sigma | S (J/mol/K) | JANAF | H(298) - H(0) (kJ/mol) | JANAF |
| --- | --- | ---: | ---: | ---: | ---: |
| H2O | C2v, 2 | 188.77 | 188.835 | 9.924 | 9.905 |
| CO2 | Dinfh, 2 | 213.63 | 213.785 | 9.341 | 9.364 |
| NH3 | C3v, 3 | 192.47 | 192.774 | 10.016 | not checked |
| CH4 | Td, 12 | 186.16 | 186.251 | 10.011 | 10.016 |
| C2H2 | Dinfh, 2 | 200.44 | 200.927 | 9.930 | not checked |

## Limits

No anharmonicity, hindered rotors or low-frequency corrections; modes below
about 100 cm^-1 are counted and reported because their harmonic entropy is
unreliable. Only the electronic ground state is included.
