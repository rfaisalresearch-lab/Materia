# Machine-learned interatomic potentials (MACE-MP-0)

Materia drives the MACE-MP-0 foundation model (Batatia et al.,
arXiv:2401.00096; MACE and weights MIT-licensed), a universal machine-learned
surrogate of PBE and PBE+U trained on the Materials Project trajectory data.
PyTorch and MACE live in their own conda environment (`materia-ml`), never in
the Materia process: `materia.solvers.ml_driver.MLPotential` keeps one
persistent worker and exchanges JSON lines with it, with a timeout on every
call. Registered models: `ml/mace-mp-0-small`, `ml/mace-mp-0-medium`. Every
result records the model file's SHA-256 and the MACE and PyTorch versions.

```bash
conda create -n materia-ml -c conda-forge python=3.11 pip
conda run -n materia-ml pip install mace-torch
```

As a Materia potential it works with relaxation, molecular dynamics,
finite-displacement phonons, NEB, elastic constants and melt-quench.

## Checked against Quantum ESPRESSO PBE (silicon)

| Quantity | MACE-MP-0 small | MACE-MP-0 medium | QE PBE |
| --- | --- | --- | --- |
| Lattice constant | 5.4647 A | 5.4555 A | 5.4707 A |
| Gamma optical phonon | 11.46 THz | 11.18 THz | 15.13 THz (DFPT) |
| Forces against energy differences | 2e-8 eV/A | | |

The lattice constant agrees with PBE to 0.1 to 0.3 percent, but the optical
phonon is about 25 percent too soft in both sizes: universal machine-learned
potentials are known to underestimate curvature of the energy surface. Use
MACE-MP-0 for structures and screening, and DFT (or a fitted potential) for
vibrational properties.

## Limitations

* Inherits PBE's errors and is least reliable far from the training crystals.
* No charges, magnetic moments or electronic structure.
* Slabs must be padded in fully periodic cells; molecules work as isolated.
