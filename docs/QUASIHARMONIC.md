# Quasi-harmonic thermodynamics

`materia.physics.quasiharmonic.quasiharmonic(structure, potential, volume_scales,
temperatures_K, supercell, dos_mesh)` computes, at each temperature, the
equilibrium volume, isothermal bulk modulus, Gibbs energy, entropy, C_V, C_P,
volumetric and linear thermal expansion and the thermodynamic Gruneisen
parameter of a crystal.

## Method

At each volume the cell is scaled isotropically (internal coordinates
optionally relaxed), the static energy is evaluated and periodic phonons are
computed by finite displacements in the given supercell. The harmonic phonon
free energy is summed over a uniform q mesh. At each temperature
`E(V) + F_vib(V, T) + PV` is fitted with the third-order Birch-Murnaghan form.
The expansion coefficient is `d ln V / dT` by finite differences over the
temperature grid, zero at 0 K. `C_P = C_V + alpha^2 B_T V T`.

Any Materia potential can be used, so the same calculation runs with native
EAM or Stillinger-Weber, LAMMPS, MACE or Quantum ESPRESSO forces.

## Refusals

- fewer than five volumes, scales outside 0.7 to 1.3, or temperatures not
  increasing;
- a structure that is not periodic in three dimensions;
- a volume where the crystal has imaginary modes away from the three acoustic
  modes at Gamma (the harmonic free energy is undefined there);
- a free-energy minimum outside the sampled volumes at any temperature;
- residual forces at a volume, unless internal relaxation is requested;
- a supercell narrower than twice the force cutoff (from the phonon module).

## Validation

Parity with phonopy 4.7.2 (A. Togo, BSD-3), driven by the same Materia
potential, for fcc Cu with the shipped Zhou 2004 EAM, an 8x8x8 supercell and
the same 16x16x16 Monkhorst-Pack mesh: at 100, 200 and 300 K the volume
agrees to 2e-6, the bulk modulus and expansion coefficient to 3e-4 and C_P to
1e-3.

Supercell convergence matters near an instability. In a 6x6x6 supercell at
7% expansion, the ASE force constants Materia uses give a free energy 0.1 meV
per atom away from phonopy's. Phonopy averages equidistant periodic images and
converges faster. In 8x8x8 the two agree to 1e-6 eV, and the 9% expanded cell
that looks stable in 6x6x6 shows imaginary modes. Check the supercell before
trusting volumes close to the stability limit.

Against experiment, with the Y. Mishin et al. Cu potential (Phys. Rev. B 63
(2001) 224106, `Cu_mishin1.eam.alloy` from the LAMMPS repository): linear
expansion 14.1e-6 /K at 300 K against 16.5e-6 /K measured, Gruneisen
parameter 1.75 against about 2.0. The LAMMPS copy of that file gives atomic
number 1 for Cu in its header, which Materia refuses; the test corrects that
one field in a temporary copy and leaves the tables untouched.

The shipped Zhou 2004 Cu potential gives 43e-6 /K at 300 K and a Gruneisen
parameter near 4, two and a half times the measured expansion, and becomes
dynamically unstable at 12% expansion. That is a property of the potential,
confirmed by phonopy; it is not fitted to anharmonic properties and should
not be used for thermal expansion.

## Limits

The quasi-harmonic approximation neglects anharmonicity at fixed volume. It is
usually reliable to about half the melting temperature; above that, use
molecular dynamics. There is no electronic free energy.
