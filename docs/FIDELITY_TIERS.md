# Fidelity tiers

Materia is layered. Each tier answers a different class of question, and each
declares what it cannot answer. Every result carries its tier.

## Why there are tiers at all

A general exact solution of the many-body Schrödinger equation for an arbitrary
condensed-matter system is not computationally possible at useful scales. The
cost of an exact treatment grows exponentially with the number of interacting
electrons. Every practical method is an approximation, and the useful question
is never "is this exact" but "which approximation, and is it valid here".

Materia makes that question answerable by attaching the answer to every number.

---

## Tier 0, structural

**What it is.** Crystallography and geometry. Lattice generation from space
group parameters, primitive-lattice detection, Miller-indexed surface
construction, periodic boundary conditions, neighbour lists, bond perception,
coordinate transforms.

**Exactness.** Exact within its own definitions. A nearest-neighbour distance
computed from a lattice constant is exact arithmetic, not a model prediction.

**What it cannot give.** Any energy, any force, anything electronic. Bond
*perception* is a distance heuristic and is labelled as such; a bond *order*
requires Tier 2 or above.

**Where the approximation enters.** Only in the input: the lattice constant is
a measured value at a stated temperature, and the ideal lattice is defect-free
unless you put a defect in it.

---

## Tier 1, classical

**What it is.** Empirical interatomic potentials with analytic forces, energy
minimisation and molecular dynamics.

| Model | Covers | Reference |
| --- | --- | --- |
| Stillinger-Weber | Si, Ge, and substitutional impurities on those hosts | Phys. Rev. B 31 (1985) 5262 |
| Lennard-Jones 12-6 | Any element with a tabulated cohesive energy | Proc. R. Soc. A 106 (1924) 463 |
| Harmonic | Testing minimisers and thermostats |, |
| Embedded-atom method (setfl) | Cu, Au, W shipped (Zhou 2004); any user-supplied setfl file | Phys. Rev. B 29 (1984) 6443; Phys. Rev. B 69 (2004) 144113 |
| Point-charge electrostatics | Any structure with an explicit charge model; bulk, slab or cluster | Ewald 1921; Yeh and Berkowitz 1999 |

**What it gives.** Total energies, forces, relaxed geometries, molecular
dynamics in the microcanonical and canonical ensembles, local strain. For
point charges supplied by the user, the long-range Coulomb energy, forces,
site potentials and site fields (`docs/ELECTROSTATICS.md`); these charges are
fixed and do not respond to the field.

**What it cannot give.** Anything involving electrons: no density of states, no
charge transfer, no optical response, no bond order, no tunnelling current. A
classical potential contains no electrons, so asking it for a density of states
is a category error and is reported as one.

**Where the approximation enters.**
* The functional form is fitted, not derived. Stillinger-Weber was fitted to
  the diamond structure and to bulk melting; its transferability to surfaces,
  defects and other coordinations is approximate.
* Lennard-Jones is pairwise additive. Metallic bonding is not pairwise, so the
  metal parameters derived here are for interactive geometry only. Cu, Au and W
  have embedded-atom potentials instead (`docs/EAM.md`), which capture the
  many-body density dependence of metallic bonding but still contain no
  electrons, no angular forces and no magnetism.
* Substitutional impurities are handled by rescaling the host sigma with the
  covalent-radius ratio. This gives the sign and rough size of the local
  bond-length change and carries no electronic information whatsoever.
* Nuclei are classical. There is no zero-point energy, and below the Debye
  temperature the equipartition temperature is not the thermodynamic
  temperature of the real solid.

---

## Tier 2, semi-empirical

**What it is.** Orthogonal nearest-neighbour tight binding.

| Model | Covers | Reference |
| --- | --- | --- |
| sp3s* | Si, Ge, GaAs | Vogl, Hjalmarson and Dow, J. Phys. Chem. Solids 44 (1983) 365 |
| pz | Graphene and sp2 carbon | Castro Neto et al., Rev. Mod. Phys. 81 (2009) 109 |

Substitutional impurities are added as an on-site energy shift taken from
Harrison free-atom term values.

**What it gives.** Single-particle eigenvalues and eigenvectors, band
structures along a k-path, total and site-projected densities of states,
Mulliken-like partial charges, and the vacuum local density of states that the
Tersoff-Hamann scanning tunnelling model needs.

**What it cannot give.**

* **Self-consistency.** The on-site energies are fixed empirical parameters, so
  there is no charge density to iterate. No charge transfer, no band bending, no
  response to an applied field or a net charge. Requesting a self-consistent
  solution returns an explicit refusal naming DFTB and DFT solvers.
* **Charged systems.** Refused, with the reason.
* **Spin.** The model is spin-restricted. A magnetic moment set on an atom is
  stored, reported and ignored by the Hamiltonian, and the solver says so.
* **Excited states, optical spectra, phonons.**
* **Quantitative surface-state energies.** The parameters were fitted to bulk
  bands. Dangling-bond states appear at the right character but not at
  quantitative energies, and every run warns about this when a surface is
  present.
* **Shallow-donor binding energies.** The 45 meV binding energy of phosphorus
  in silicon comes from the long-range Coulomb tail of the ionised donor and a
  central-cell correction. A short-range on-site shift in a finite
  non-self-consistent supercell does not contain either.

**What it does reproduce.** The values it was fitted to: silicon's 12.5 eV
valence bandwidth and 1.17 eV indirect gap near 0.85 Γ-X, GaAs's 1.55 eV direct
gap, graphene's gapless Dirac point and 6|t| bandwidth. These are checked on
every test run, and they are a statement about the fit, not a prediction.

---

## Tier 3, external first principles

**What it is.** Adapters for installed open-source packages. Materia bundles
none of them.

| Adapter | Current Materia support |
| --- | --- |
| `external:ase` | Energy and relaxation with a user-supplied ASE calculator |
| `external:gpaw` | Legacy single-point energy and forces; the DFT experiment system adds ground states, fixed and variable-cell relaxation, DOS and PDOS |
| `external:quantum-espresso` | Detected and declared, not driven |
| `external:lammps` | EAM (eam/alloy) energy, forces, stress, fixed-cell relaxation and dynamics, driven out of process; Tier 1 because the potential is classical |
| `external:gromacs`, `external:orca`, `external:ams` | Detected and declared, not driven |
| `external:cp2k` | Detected and declared, not driven |
| `external:pyscf`, `external:psi4` | Detected and declared, not driven |
| `external:wannier90` | Detected and declared, not driven |
| `external:openmx` | Detected and declared, not driven |

Each adapter registers whether or not its package is present. Detection is not
execution. An undriven or absent solver request is refused, and the request is
preserved so it can be implemented or rerun later without inventing a result.

**Licensing.** These packages carry their own licences, several of them GPL.
Materia communicates through public Python interfaces or input and output files
and does not vendor, link against or redistribute any of their code, so their
licences apply to your installation of them, not to Materia.

---

## Distinctions the program maintains

These are different things, and Materia never conflates them:

| Quantity | What it is | Where it comes from |
| --- | --- | --- |
| Classical trajectory | Positions of point nuclei over time | Tier 1 dynamics |
| Quantum probability density | \|ψ\|² for a single-particle state | Tier 2 eigenvector, or Tier 3 |
| Electron density | Σ over occupied states of \|ψ\|², a real observable | Tier 3; Tier 2 gives a projected approximation |
| Orbital amplitude | ψ itself, including its sign | Tier 2 eigenvector |
| Spin density | n↑ − n↓ | Tier 3 only |
| Electrostatic potential | Solution of Poisson's equation for the charge density | Tier 3 only |
| Scanning-probe observable | Tunnelling current or tip-sample force, an instrument response | Tier 2 for STM, Tier 1 for AFM |

The educational electron-shell diagram belongs to none of these. It is labelled
illustrative, and the panel states that electrons do not follow those paths and
that its only physical content is the shell occupancy.
