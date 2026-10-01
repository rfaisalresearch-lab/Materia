# Scientific models and approximations

Every model Materia implements, with its equations, its parameters, its
references, and, at the same level of detail, what it cannot do.

Internal units throughout: length in ångström, energy in electronvolts, mass in
unified atomic mass units, time in femtoseconds, force in eV/Å, charge in
elementary charges, temperature in kelvin. Constants are CODATA 2018.

---

## 1. Crystallography (Tier 0)

### 1.1 Cells and lattices

A cell is a 3×3 matrix whose rows are the lattice vectors, so
`cartesian = fractional · matrix`, the row-vector convention of ASE, VASP and
CIF tooling. The reciprocal lattice includes the 2π factor:
`B = 2π (A⁻¹)ᵀ`, so `B · Aᵀ = 2π I`, which is checked on every test run.

### 1.2 Minimum image

Wrapping fractional displacements by rounding is **not** sufficient for
non-orthogonal cells: for a 60° hexagonal surface cell it can return an image
one lattice vector away from the true minimum. Materia rounds and then searches
the 27 neighbouring images, taking the shortest. Neighbour lists do not rely on
this at all; they expand images explicitly.

### 1.3 Primitive-lattice detection

Material definitions are written in the conventional cell. For a centred
lattice the conventional cell contains several primitive cells, and enumerating
only its integer lattice vectors would give a Si(111) surface cell of 7.68 Å
instead of the correct 3.84 Å.

Materia recovers the real translation lattice from the structure itself: a
fractional vector **t** is a lattice translation when shifting every atom by
**t** maps the decorated lattice onto itself with matching species. The valid
translations are combined with the integer lattice, the three shortest
independent vectors are taken, and the basis is reduced by a greedy
Minkowski-style algorithm. No space-group table is consulted, so it works for
user-supplied materials, and it correctly finds *no* extra translations in a
supercell containing a defect.

Verified multiplicities: face-centred 4, body-centred 2, rhombohedral 3,
hexagonal and trigonal 1.

### 1.4 Surfaces

Miller indices are always given with respect to the **conventional** cell, the
convention an experimentalist uses when they say "a Si(100) wafer".

1. The plane normal comes from the conventional reciprocal lattice,
   **n** ∝ (h k l)·B.
2. Short integer vectors of the *primitive* lattice are enumerated and their
   projections on **n** computed. Those projections form a one-dimensional
   lattice whose spacing `d_min` is the physical interplanar spacing of the
   real crystal for that orientation, which is `a/2` for Si(100), not the
   conventional `a`.
3. Vectors with zero projection span the surface plane; the pair with the
   smallest area after Lagrange-Gauss reduction becomes the in-plane cell. The
   shortest vector with projection `d_min` is the stacking vector.
4. The supercell is filled, repeated, rotated so the normal is +z, cut open in
   z, wrapped back into the two-dimensional cell, and padded with vacuum.

Checked against analytic values: Si(111) in-plane 3.8403 Å at 60°, Si(100)
3.8403 Å at 90°, cell areas a²√3/4, a²/2 and a²/√2 for (111), (100) and (110).

### 1.5 Termination selection

An ideal truncation can be cut at several inequivalent planes. Cutting through
the short bonds of a Si(111) bilayer leaves three dangling bonds per surface
atom, which is not the plane a real crystal cleaves along. Materia tries every
cut, counts broken bonds against a per-element reference coordination taken
from the most-coordinated atom of that species, and keeps the best. Si(111)
selects the cut with one dangling bond per surface atom; Si(100) has two on
every cut, as it must.

The chosen offset and the full candidate table are recorded in
`structure.info['surface']` so the choice is inspectable rather than hidden.

### 1.6 Surface reconstructions

A reconstruction is split into a *generator*, which fixes topology and
periodicity, and a *relaxation*, which fixes geometry. The generator derives
directions from the slab rather than assuming a cube axis: for a tetrahedral
surface site holding two back-bonds **b₁** and **b₂**, the two missing orbitals
span the plane of −(**b₁**+**b₂**) and **b₁**×**b₂**, so the in-plane
projection of **b₁**×**b₂** is the axis along which two such sites can pair.
The Si(100) dimer generator groups surface sites into rows perpendicular to
that axis and pairs them along it, which is why it refuses a slab whose
in-plane repeat along the axis is odd.

The generator imposes the pairing and a starting separation; nothing else. The
bond length is the minimum of whatever interatomic model runs next, at a
tightened force criterion of 0.005 eV/Å, and is independent of the starting
separation over 2.2-3.4 Å to better than 10⁻³ Å. The result is checked against
the 1×1 and 2×1 translations before it is returned.

Published structural parameters live in the material file and are used only for
the comparison Materia prints; they never place an atom. Where the model cannot
produce a feature of the real structure, Stillinger-Weber and the Jahn-Teller
buckling of the Si(100) dimer, the deviation is reported and the ordered
buckled phases are refused. See [RECONSTRUCTIONS.md](RECONSTRUCTIONS.md).

### 1.7 Bond perception

`bonded(i, j) ⇔ d_ij ≤ 1.15 · (r_cov(i) + r_cov(j))`, with Cordero covalent
radii. Every bond produced this way is tagged `origin="distance-heuristic"` and
carries order 1.0. A real bond order needs Tier 2 or above.

---

## 2. Classical potentials (Tier 1)

### 2.1 Stillinger-Weber

    E = Σ_{i<j} φ₂(r_ij) + Σ_{i, j<k} φ₃(r_ij, r_ik, θ_jik)

    φ₂(r) = A ε [B (σ/r)^p − (σ/r)^q] exp[σ/(r − aσ)],        r < aσ
    φ₃    = λ ε (cos θ − cos θ₀)² exp[γσ/(r_ij − aσ)] exp[γσ/(r_ik − aσ)]

| Parameter | Si | Ge |
| --- | --- | --- |
| ε (eV) | 2.1683 | 1.93 |
| σ (Å) | 2.0951 | 2.181 |
| a | 1.80 | 1.80 |
| λ | 21.0 | 31.0 |
| γ | 1.20 | 1.20 |
| A | 7.049556277 | 7.049556277 |
| B | 0.6022245584 | 0.6022245584 |
| p, q | 4, 0 | 4, 0 |
| cos θ₀ | −1/3 | −1/3 |

Silicon: Stillinger and Weber, Phys. Rev. B 31 (1985) 5262.
Germanium: Ding and Andersen, Phys. Rev. B 34 (1986) 6987.

Forces are analytic and agree with numerical gradients to better than
10⁻⁶ eV/Å. By construction the ideal diamond lattice has exactly −2ε per atom,
which the validation suite checks to one part in 10⁷.

**Substitutional impurities.** With `impurities=("P", ...)` an impurity keeps
the host functional form and host ε, and its pair σ is rescaled by the ratio of
covalent radii. This is a *geometric* approximation. It gives the sign and
rough magnitude of the local bond-length change, a P-Si bond contracts by
about 1 % relative to Si-Si, in the direction experiment shows, and it carries
no information at all about the impurity's valence, level or charge state. The
provenance record says exactly this.

**Not modelled:** bond breaking beyond the cutoff, charge, polarisation,
dispersion, or any electronic property.

### 2.2 Lennard-Jones

    E = Σ_{i<j} 4 ε_ij [(σ_ij/r)¹² − (σ_ij/r)⁶]

with Lorentz-Berthelot mixing, shifted to zero at the cutoff. For a metal the
parameters are *derived*, not fitted, from the fcc 12-6 lattice sums:

    ε = E_coh / 8.6102,    σ = r_nn / 1.09026

A pair potential cannot represent metallic many-body bonding. These parameters
are for interactive geometry only, and the model says so in every result.
Quantitative metal energetics need an embedded-atom potential through the
LAMMPS adapter. With a 10 Å cutoff the lattice sum is truncated, which is why
the gold cohesive energy comes out at −3.66 eV rather than the −3.81 eV the
parameters were derived from.

### 2.3 Minimisation

FIRE, Bitzek et al., Phys. Rev. Lett. 97 (2006) 170201, with the standard
parameters (N_min = 5, f_inc = 1.1, f_dec = 0.5, α₀ = 0.1, f_α = 0.99). Atoms
flagged fixed are held. The convergence criterion is the maximum force on free
atoms. The cell is not relaxed. The result is the nearest local minimum, not a
global one, and the record says so.

### 2.4 Dynamics

Velocity Verlet with BAOAB splitting when a Langevin thermostat is active
(Leimkuhler and Matthews, Appl. Math. Res. Express 2013, 34). The random number
generator is seeded explicitly, so a run is bit-reproducible from the project
file. Nosé-Hoover and Berendsen are not implemented and are refused by name.

Validated: microcanonical energy drift below 5×10⁻⁴ eV per atom over 500 fs;
Langevin mean temperature within 12 % of target.

Classical nuclei: no zero-point energy, no quantum heat capacity. Below the
Debye temperature the equipartition temperature is not the thermodynamic
temperature of the real solid.

---

### 2.5 Embedded-atom metals

`E = sum_i F_a(rho_i) + 1/2 sum_{i != j} phi_ab(r_ij)`, `rho_i = sum_j f_b(r_ij)`,
with `F`, `f` and `r phi` read from a setfl file and interpolated with the
LAMMPS cubic Hermite scheme; forces are its exact gradient and the virial
stress is `sigma_ab = (1/2V) sum_directed (dE/dr) D_a D_b / r`. Shipped: the
Zhou, Johnson and Wadley (2004) potentials for Cu, Au and W, NIST
retabulation, public domain. Relaxation is FIRE at fixed cell; dynamics is
velocity Verlet with an optional BAOAB Langevin thermostat. Refusals: elements
the file does not list, malformed or altered files, alloys across
single-element files. Full treatment and validation in [EAM.md](EAM.md).

### 2.6 Point-charge electrostatics

The Coulomb energy of explicit point charges `q_i` (in e), with
`k_e = 14.3996454784 eV A`. Three methods, chosen by the cell's periodicity:

* **Crystal (3 periodic directions):** Ewald summation,
  `E = 1/2 sum' q_i q_j erfc(alpha r)/r + (2 pi/V) sum_{k!=0} exp(-k^2/4alpha^2)/k^2 |S(k)|^2 - (alpha/sqrt(pi)) sum q_i^2`,
  conducting (tin-foil) surrounding by default; optional vacuum surrounding
  `+ (2 pi/3V)|M|^2` for neutral cells; optional uniform neutralising
  background `- pi Q^2/(2 V alpha^2)` for charged cells, which must be
  requested explicitly.
* **Slab (2 periodic directions):** the same sum in an internally enlarged
  cell with gap `max(-ln(eps)/G_min, 2h, 10 A)` and the Yeh-Berkowitz
  correction `(2 pi/V') M_z^2`. Neutral slabs only.
* **Cluster:** the direct pair sum.

Wires are refused. Parameters: `r_c` from a measured cost balance, then
`alpha = s/r_c` and `k_c = 2 alpha s` with `s = sqrt(-ln eps)`; every choice
is recorded. Convergence is established by an independent second split
(`alpha` scaled by 0.8, slab gap by 1.3) and reported as the energy
difference.

Charges come only from an explicit charge model: per element or per atom with
a stated source, or the formal-point-ion model (formal charges as rigid point
charges, the full ionic limit). No charge is inferred.

Approximations recorded with every result: point charges in vacuum, no charge
penetration, no polarisation, no dielectric screening, fixed charges. Full
treatment in [ELECTROSTATICS.md](ELECTROSTATICS.md).

**Rigid ions.** `phi(r) = k_e q_i q_j / r + A exp(-b r) - C / r^6`: the Coulomb
sum above plus Buckingham terms truncated and shifted at 10 A. Shipped: BKS
silica (B. W. H. van Beest, G. J. Kramer and R. A. van Santen, Phys. Rev. Lett.
64 (1990) 1955). Pairs inside the inner maximum of their interaction (Si-O
1.194 A, O-O 1.439 A) are refused rather than allowed to collapse. Fixed cell.
Full treatment in [RIGID_ION.md](RIGID_ION.md).

References: P. P. Ewald, Ann. Phys. 369 (1921) 253; S. W. de Leeuw,
J. W. Perram and E. R. Smith, Proc. R. Soc. Lond. A 373 (1980) 27;
D. Fincham, Mol. Simul. 13 (1994) 1; I.-C. Yeh and M. L. Berkowitz,
J. Chem. Phys. 111 (1999) 3155; G. Makov and M. C. Payne, Phys. Rev. B 51
(1995) 4014.

## 3. Tight binding (Tier 2)

### 3.1 The model

Orthogonal, two-centre, nearest-neighbour. The published Vogl parameters
`V_ss`, `V_xx`, `V_xy`, `V_sapc` are four-neighbour sums; Materia converts them
back to Slater-Koster two-centre integrals so the Hamiltonian can be built for
*arbitrary* geometries, surfaces, defects, relaxed structures, not only the
ideal crystal:

    (ssσ) = V_ss / 4
    (spσ) = √3 · V_sapc / 4
    (ppσ) = (V_xx + 2 V_xy) / 4
    (ppπ) = (V_xx −   V_xy) / 4

Distance dependence follows Harrison's d⁻² scaling. This is a rule of thumb:
the parameters were fitted at the equilibrium bond length only.

Parameters for Si, Ge and GaAs from Vogl, Hjalmarson and Dow,
J. Phys. Chem. Solids 44 (1983) 365. Graphene uses the standard single-orbital
π model with t = −2.7 eV.

### 3.2 Substitutional impurities

An impurity is represented by an on-site energy shift equal to the difference
of free-atom term values (Harrison, *Electronic Structure and the Properties of
Solids*, Table 2-1), with the host's hopping integrals. For P in Si the shift
is −4.43 eV on s and −1.95 eV on p.

This reproduces the *direction* and rough size of the impurity potential: a
donor is an attractive site, an acceptor a repulsive one, and the perturbation
is visible in the local density of states and the partial charges. It does
**not** reproduce shallow-donor binding energies, which come from the
long-range Coulomb tail of the ionised donor and a central-cell correction,
neither of which a short-range on-site shift in a finite non-self-consistent
supercell contains. Every result carrying an impurity says so.

### 3.3 What it refuses

Self-consistency, net charge, spin polarisation, excited states, optical
spectra and phonons, each with a reason and the solvers that could.

---

## 4. Scanning tunnelling microscopy

### 4.1 Tersoff-Hamann

Tersoff and Hamann, Phys. Rev. B 31 (1985) 805. For a structureless s-wave tip
apex at **r**_t and bias V,

    I(r_t, V) ∝ ∫_{E_F}^{E_F + eV} ρ_s(r_t, E) dE

with ρ_s the *sample* local density of states at the centre of curvature of the
apex. Materia evaluates ρ_s from the tight-binding eigenstates,

    ψ_n(r) = Σ_{i,α} c_{n,i,α} χ_α(r − R_i)

with atom-centred exponential vacuum tails,

    χ_s(r) = exp(−κ r),   χ_p(r) = (û · r̂) exp(−κ r)

and the decay constant from the effective barrier,

    κ = √(2 m φ_eff) / ħ,   φ_eff = (φ_tip + φ_sample)/2 − |eV|/2

For φ_eff = 4 eV, κ ≈ 1.02 Å⁻¹, so the current falls by roughly one decade per
ångström, the standard experimental rule of thumb. The validation suite fits
the computed decay and checks it against 2κ.

### 4.2 What the numbers mean

* **The current is in arbitrary units.** The Tersoff-Hamann prefactor contains
  the tip density of states and the apex geometry, neither of which is known.
  Materia calibrates the constant-current setpoint against the computed signal
  at a nominal height, exactly what an experimental feedback loop does, and
  labels the current axis as calibrated.
* **A bright feature is a maximum of the vacuum local density of states.** It
  need not coincide with a nucleus. Dangling bonds, adsorbates and antibonding
  states all produce maxima that can be displaced from, or absent at, atomic
  positions. Every hover result states which it believes it has found, and with
  what confidence.

### 4.3 Numerical approximations, each recorded

| Approximation | Why it is acceptable | Where recorded |
| --- | --- | --- |
| Electronic structure on the topmost ~9 Å | Deeper atoms are suppressed by exp(−2κd); at 9 Å that is ~10⁻⁸ in intensity | `approximations`, with the depth and atom count |
| Vacuum sum over the topmost 5 Å | Same argument | `tolerances.vacuum_depth_cutoff_A` |
| Lateral images within 6.5 Å of the window | exp(−2κ·6.5) ≈ 10⁻⁶ | `parameters.replicated_atoms` |
| States with negligible surface weight dropped | They cannot reach the vacuum | `tolerances.state_amplitude_threshold` |
| Surface Brillouin zone on a small grid | Converged for the corrugation to <2 % between 1×1 and 3×3 | `parameters.n_kpoints` |
| Constant-current height by log-linear feedback | Residual reported per scan, typically 10⁻⁴ | `convergence.residual` |

The truncation creates its own lower surface, whose states are excluded from
the vacuum sum. That is stated in the provenance rather than glossed over.

### 4.4 Not modelled

Tip-induced band bending, inelastic channels, the bias dependence of the tip
density of states, current-induced forces, spin-polarised tunnelling, and any
many-body effect.

---

## 5. Atomic force microscopy

### 5.1 Forces

Short range: a 12-6 Lennard-Jones interaction between the apex atom and every
sample atom, with generic parameters derived from cohesive energies. This is an
estimate of dispersion and Pauli repulsion, **not** a chemical-bonding
calculation. True atomic contrast on semiconductors is dominated by covalent
tip-sample bonding, which needs an electronic-structure force calculation.

Long range: the Hamaker sphere-plane result, F(z) = −H R / (6 z²), with H the
Hamaker constant and R the apex radius. It carries no atomic contrast and is
responsible for the large background force and frequency shift.

### 5.2 Frequency shift

Giessibl, Phys. Rev. B 56 (1997) 16010:

    Δf = − f₀ / (k A²) · (1/π) ∫_{−A}^{A} F_ts(z + A + q) · q / √(A² − q²) dq

evaluated with Chebyshev-Gauss quadrature of the **first** kind, whose weight
matches the 1/√(A² − q²) kernel. (Using second-kind nodes with equal weights
under-estimates the integral by (n−1)/(n+1), a bug the validation suite
caught and now guards against.) As A → 0 this reduces to
Δf = −(f₀/2k)·∂F/∂z, which is checked to 0.1 %, and the quadrature is exact to
10⁻¹⁰ for a linear force at any amplitude.

### 5.3 Setpoint solving

Both the force and the frequency shift are **non-monotonic** in tip height. A
plain bisection over the whole bracket would lock onto the wrong branch, so
Materia scans inward from far away, brackets the first crossing, which is
where a real feedback loop stops, and bisects inside that bracket. Pixels that
never reach the setpoint are counted and reported; they are not valid
measurements and the scan says so.

### 5.4 Not modelled

Covalent tip-sample bonding, tip-apex relaxation and lateral flexing (which
dominate CO-tip imaging), dissipation, electrostatic force from contact
potential differences, and cantilever dynamics beyond the first harmonic.

---

## 6. Instrumental noise

Noise describes the **measurement chain**, not the sample. Every component is
individually switchable and driven by an explicit seed, so a scan is
bit-reproducible. The unfiltered physical signal is always kept as a separate
channel.

| Component | Model |
| --- | --- |
| White | Gaussian, uncorrelated between pixels (preamplifier Johnson noise, shot noise) |
| Pink | 1/f along the fast-scan direction, by frequency-domain filtering |
| Line offset and gain | Random-walk offset and gain error per scan line |
| Thermal drift | Linear ramp in z across the image |
| Piezo creep | Exponentially decaying offset after the scan starts |
| Vibration | Sinusoidal ripple at a chosen spatial frequency, random phase |
| Dropouts | Occasional single-line disturbances, off by default |

---

## 7. The procedural wafer

A 300 mm wafer holds of order 10²⁴ atoms and is never instantiated. Roughness
is band-limited value noise with the specified RMS and correlation length;
grains are a Voronoi tessellation of seeded points (disabled for single
crystals, because real single-crystal wafers have none); dopants and point
defects are a Poisson process at the specified densities. Everything is
deterministic in (seed, position), which is what lets a project store a wafer
and a coordinate instead of atoms.

All of it is **synthetic**. It is reproducible and statistically plausible; it
is not a measurement of any real wafer, and every result derived from it
carries `origin="estimated"`.

---

## 8. Atomic reference data

| Quantity | Source |
| --- | --- |
| Standard atomic weights | IUPAC 2021, Pure Appl. Chem. 94 (2022) 573 |
| Nuclide masses | AME 2020, Chinese Phys. C 45 (2021) 030003 |
| Isotopic compositions | Meija et al., Pure Appl. Chem. 88 (2016) 293 |
| Covalent radii | Cordero et al., Dalton Trans. (2008) 2832 |
| Van der Waals radii | Alvarez, Dalton Trans. 42 (2013) 8617 |
| Ionisation energies, electron affinities | CRC Handbook, 104th ed., Section 10 |
| Free-atom term values | Harrison (1980), Table 2-1 |

Electron configurations are generated from the Madelung (n+l, n) rule with an
explicit table of the twenty experimentally established exceptions. The
Madelung rule is empirical, not a theorem; beyond Z = 103 it is labelled
predicted with low confidence because relativistic effects are not modelled.

Orbital plots in the inspector are hydrogen-like functions evaluated with a
Slater effective nuclear charge (Slater, Phys. Rev. 36 (1930) 57). The number
of radial nodes and the shell ordering are correct; the amplitudes are an
approximation for a many-electron atom. They are tagged `estimated`.

The electron-shell diagram is tagged `illustrative` and the panel states that
electrons do not follow those paths and that its only physical content is the
shell occupancy.
