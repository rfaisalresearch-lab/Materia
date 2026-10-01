# Extended defects

Code: `materia/structure_builder/extended_defects.py`. Tests:
`tests/unit/test_extended_defects.py`.

## Stacking faults (Materia-native)

`generalized_stacking_fault(potential, symbol, a)` tilts the third vector of an
fcc cell oriented x = [1-10], y = [11-2], z = [111] by a fraction of the
Shockley partial Burgers vector a/6[11-2], so periodicity holds exactly one
fault and no surfaces. It returns gamma(s) in mJ/m^2, the intrinsic fault
energy (s = 1) and the unstable fault energy (the maximum), rigid or with
atoms relaxed along [111] only (Vitek 1968).

Cu, Zhou 2004 EAM, 12 layers: intrinsic 40.1 mJ/m^2 rigid and 26.6 relaxed;
unstable 178.6 rigid and 120.2 relaxed. The hcp-fcc energy difference of
the same potential gives 33.4 mJ/m^2 by the axial nearest-neighbour Ising
estimate. The measured value for copper is about 45 mJ/m^2, so this EAM
underestimates it; that is a property of the potential.

## Dislocations (built by matscipy, relaxed by Materia)

`dislocation(potential, kind, symbol, a, C11, C12, C44)` builds an fcc edge or
screw, or a bcc edge or screw, dislocation with matscipy (LGPL-2.1) from the
anisotropic elastic field of the given elastic constants, in a cylinder
periodic along the line with a fixed outer shell, and relaxes the core with
the potential. `centrosymmetry` (Kelchner, Plimpton and Hamilton 1998) marks
atoms out of a perfect fcc environment; `dissociation_width` measures the
stacking-fault ribbon between Shockley partials.

Cu edge dislocation, Zhou 2004 EAM, elastic constants from
`materia.physics.elasticity`, 70 A radius (5,761 atoms): relaxes in about
16 s, |b| = a/sqrt(2), and dissociates into a planar ribbon 31 A wide along
[1-10] and less than 2 A thick normal to (111). Isotropic elasticity
(Hirth and Lothe 1982) gives 30 to 90 A depending on which shear modulus
represents copper's strongly anisotropic crystal, so it does not test the
width more closely than that.

## Grain boundaries (Materia-native)

`symmetric_tilt_boundary(potential, symbol, a, plane=(3, 1, 0))` builds a
symmetric [001] tilt boundary between mirror-image grains with (h k 0)
boundary planes, two boundaries per periodic bicrystal, samples rigid
translations of one grain within the boundary plane, merges atoms closer than
1.6 A, relaxes every candidate and keeps the lowest energy,
E_gb = (E - N e_bulk) / 2A.

Cu, Zhou 2004 EAM, Sigma5(310)[001], 36.87 degrees: 793 mJ/m^2 with three
grain repeats and 788 mJ/m^2 with five (0.6 percent). Reported as a property
of the potential; no literature value for this potential is asserted.

## Limitations

* A screw dislocation relaxed from the perfect elastic solution stays compact
  here: the dissociated screw is not reached by local minimisation from that
  start, and a pre-dissociated starting configuration is not built yet.
* Single straight dislocations in cylinders with fixed boundaries; no
  dipoles or quadrupoles in periodic cells, no Peierls stress or kink work.
* Grain boundaries: [001] symmetric tilt in fcc only; the translation search is a
  grid and atom density at the boundary is not optimised, so it is a local
  search for the boundary structure.
