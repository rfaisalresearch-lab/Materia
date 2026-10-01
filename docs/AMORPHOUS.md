# Amorphous structures by melt and quench

`materia.structure_builder.melt_quench.melt_quench(potential, crystal, melt_K)`
melts a crystal with Langevin dynamics, cools it in steps at a stated rate,
holds it, relaxes it to a local minimum, all at fixed volume, and reports the
radial distribution function, coordination, bond-angle distribution, energy
above the crystal and a crystalline order parameter. If the crystal has not
lost its long-range order after the melt stage the run is refused: quenching a
superheated solid would not give a glass.

## Amorphous silicon

| Potential | Mean coordination | Fourfold | rms bond angle | Order left |
| --- | --- | --- | --- | --- |
| Stillinger-Weber (1985), 3500 K melt | 4.38 | 59% | 19.7 deg | 0.007 |
| Stillinger-Weber refit by Vink et al. (2001), 6000 K melt | 3.97 | 93% | 10.9 deg | 0.02 |
| Measured (annealed a-Si) | about 3.9 | | about 10 deg | |

216 atoms, crystal density, cooled at 50 K/ps. The original parameterisation
gives the known over-coordinated, liquid-like glass (Luedtke and Landman
1989); the Vink refit (`stillinger-weber-si-vink2001`, lambda = 31.5,
epsilon = 1.64833 eV) gives a realistic continuous random network. It also
melts far above 3500 K, which the melt check caught.

## Limitations

* Fixed volume at the crystal density; no constant-pressure dynamics here,
  so the measured 1 to 2 percent lower density of a-Si is not reproduced.
* Cooling rates of 1e13 to 1e14 K/s, far faster than any laboratory quench.
* Amorphous silica is not provided: BKS collapses at melt temperatures
  without an added repulsive wall, which Materia does not add.
