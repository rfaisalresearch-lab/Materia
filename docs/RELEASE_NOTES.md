# Materia 0.2.0

This is the first public release of Materia, an open source workbench for atoms, molecules, materials and devices. It brings its own models together with well known open source engines, runs them for you, and checks their answers against the engines' own output and against published data. Every result says which model produced it, what was assumed and whether it converged. When a model cannot answer a question, Materia says so instead of guessing.

## Getting it

Download the zip for your computer from the assets below.

On Windows 10 or 11, unzip `Materia-0.2.0-windows.zip` and double-click `Install Materia.cmd`. When it finishes, open Materia from the Start Menu or the Desktop shortcut. If you do not have Python yet, the installer offers to install Python 3.12 for you.

On macOS 12 or later, unzip `Materia-0.2.0-macos.zip`, right-click `Install Materia.command` and choose Open. The app ends up in `~/Applications/Materia.app`. You need Python 3.10 to 3.13 from python.org or Homebrew.

Both installers need an internet connection, and neither package is code signed yet, so your system will ask you to confirm the first time.

## What it can do

Everything below has been tested, and the numbers in brackets are what the tests found.

### Building structures

You can build crystals from a library of materials, cut surfaces of any orientation with stable terminations, reconstruct Si(100) into dimer rows, and add vacancies, dopants, adsorbates, alloys and partially occupied sites. Extended defects are supported too: stacking faults, dislocations that split into partials, and tilt grain boundaries (copper's stacking fault comes out at 26.6 mJ/m²). Amorphous solids come from a melt and quench that first checks the crystal really melted (amorphous silicon ends up with an average coordination of 3.97).

### Classical simulation

Materia has Stillinger-Weber, Lennard-Jones, EAM, Tersoff and rigid ion potentials, and can use LAMMPS and the MACE machine learned potentials. With any of them you can relax structures, run molecular dynamics, compute phonons and phonon dispersions, find reaction paths with the nudged elastic band method, and get rates from transition state theory. The Tersoff forces match LAMMPS to 2e-10 eV/Å.

It also computes elastic constants, thermal expansion and heat capacity in the quasi-harmonic approximation (matching phonopy to a few parts in a million in volume), vacancy formation energies and surface energies. With the Mishin copper potential it reproduces the published lattice constant, cohesive energy, vacancy energy and surface energies to the digits given in the original paper.

### Electronic structure

GPAW and Quantum ESPRESSO handle density functional calculations: total energies, forces, relaxation, band structures, densities of states, orbital character of bands, and phonons from perturbation theory, including Born charges, dielectric constants and LO-TO splitting in polar crystals (AlAs Born charge 2.16 against 2.18 measured). Results from Quantum ESPRESSO run through Materia match pw.x run by hand to 2e-6 eV, and the two codes agree with all-electron reference data for silicon to within half a percent.

### Molecules and biomolecules

Materia drives RDKit, xTB, PySCF, Psi4 and OpenMM, from force fields up to coupled cluster, with implicit or explicit solvent and QM/MM embedding. Each matches its own engine run directly, and PySCF and Psi4 agree with each other to below a microelectronvolt for the same calculation.

For spectroscopy it gives IR and Raman spectra, UV-Vis absorption from TDDFT, NMR shieldings (methane carbon within 2 ppm of experiment), and XPS core level energies (within 0.2 eV of experiment for carbon and nitrogen). It also gives dipole moments and polarizabilities, excited states, and gas phase thermochemistry (entropies within 0.5 J/mol/K of the NIST tables).

### Microscopy and X-rays

You can simulate STM images and spectra, including CO-tip contrast and dI/dV maps, and contact and non-contact AFM. On the X-ray side there are powder diffraction patterns (same peaks as pymatgen, intensities within 1.5%), attenuation through any material (matching NIST XCOM), and absorption edges.

### Devices

Through DEVSIM, Materia simulates p-n junctions, MOS capacitors and illuminated solar cells, and reports short circuit current, open circuit voltage and fill factor. The built-in potential is exact, and the photocurrent agrees with the textbook result to 0.7%.

### Nuclear physics

Atomic masses and binding energies from AME2020, decay chains, particle data, and proton stopping in water within 0.6% of the NIST tables.

### Analysis and the app itself

There are tools for dimensional analysis and for finding simple laws in data, with checks on held out data. The app has its own window on both systems, saves projects with full history, and can be driven entirely from Python.

## Which engines run where

The core of Materia, ASE, phonopy, XrayDB, RDKit, OpenMM and DEVSIM install directly on both Windows and macOS. PySCF and xTB install directly on macOS; on Windows they run through the Windows Subsystem for Linux. Psi4, LAMMPS and MACE install into their own conda environments on either system. GPAW and Quantum ESPRESSO need conda on macOS and WSL on Windows. If an engine is missing, Materia tells you and skips that calculation.

## Known limitations

The CP2K connection is not finished and is not part of this release. Windows support is new; it is tested automatically on a Windows machine for every change, but it has had less day to day use than the Mac version. Materia prepares, runs, checks and records calculations from these engines, but very large production runs are still best done in the engines directly.

The licence and citation for every engine Materia uses are listed in `docs/THIRD_PARTY.md`.
