"""Periodic-table and isotope reference data.

Provenance
----------
All values in this module are literature reference data, not computed
results.  They are tagged in :mod:`materia.provenance` as
``origin="reference"``.

Sources
~~~~~~~
* Standard atomic weights: IUPAC Commission on Isotopic Abundances and
  Atomic Weights, "Atomic weights of the elements 2021",
  Pure Appl. Chem. 94(5) 573-600 (2022).  An element with no standard atomic
  weight carries the atomic mass (AME) of its conventional reference isotope,
  for example Tc-97 or Pu-244, so that it can be used as a dynamical mass.
  Lr, Sg, Rg, Mc and Ts keep the mass number of their reference isotope.
* Covalent radii: B. Cordero et al., Dalton Trans. 2008, 2832-2838.
* Van der Waals radii: S. Alvarez, Dalton Trans. 2013, 42, 8617.
* Pauling electronegativity: A. L. Allred, J. Inorg. Nucl. Chem. 17 (1961) 215,
  with the conventional Pauling scale values as tabulated by CRC.
* First ionisation energies and electron affinities: CRC Handbook of
  Chemistry and Physics, 104th ed. (2023), Section 10.
* Isotope masses / natural abundances: M. Wang et al., "The AME 2020 atomic
  mass evaluation", Chinese Phys. C 45 030003 (2021) and
  J. Meija et al., Pure Appl. Chem. 88 (2016) 293 (isotopic compositions).

A dash (``-``) in the tables below means "not evaluated / not applicable" and
is decoded as ``None``.  ``None`` must never be silently substituted by a
default; callers are expected to report the value as unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

_ELEMENT_TABLE = """
1   H   Hydrogen        1.008     1   0.31  1.20  2.20  13.598  0.754
2   He  Helium          4.002602 18   0.28  1.43  -     24.587  -
3   Li  Lithium         6.94      1   1.28  2.12  0.98   5.392  0.618
4   Be  Beryllium       9.0121831 2   0.96  1.98  1.57   9.323  -
5   B   Boron          10.81     13   0.84  1.91  2.04   8.298  0.280
6   C   Carbon         12.011    14   0.76  1.77  2.55  11.260  1.262
7   N   Nitrogen       14.007    15   0.71  1.66  3.04  14.534  -
8   O   Oxygen         15.999    16   0.66  1.50  3.44  13.618  1.461
9   F   Fluorine       18.998403 17   0.57  1.46  3.98  17.423  3.401
10  Ne  Neon           20.1797   18   0.58  1.58  -     21.565  -
11  Na  Sodium         22.98977   1   1.66  2.50  0.93   5.139  0.548
12  Mg  Magnesium      24.305     2   1.41  2.51  1.31   7.646  -
13  Al  Aluminium      26.981538 13   1.21  2.25  1.61   5.986  0.433
14  Si  Silicon        28.085    14   1.11  2.19  1.90   8.152  1.390
15  P   Phosphorus     30.973762 15   1.07  1.90  2.19  10.487  0.747
16  S   Sulfur         32.06     16   1.05  1.89  2.58  10.360  2.077
17  Cl  Chlorine       35.45     17   1.02  1.82  3.16  12.968  3.613
18  Ar  Argon          39.95     18   1.06  1.83  -     15.760  -
19  K   Potassium      39.0983    1   2.03  2.73  0.82   4.341  0.501
20  Ca  Calcium        40.078     2   1.76  2.62  1.00   6.113  0.025
21  Sc  Scandium       44.955907  3   1.70  2.58  1.36   6.561  0.188
22  Ti  Titanium       47.867     4   1.60  2.46  1.54   6.828  0.084
23  V   Vanadium       50.9415    5   1.53  2.42  1.63   6.746  0.525
24  Cr  Chromium       51.9961    6   1.39  2.45  1.66   6.767  0.666
25  Mn  Manganese      54.938043  7   1.39  2.45  1.55   7.434  -
26  Fe  Iron           55.845     8   1.32  2.44  1.83   7.902  0.153
27  Co  Cobalt         58.933194  9   1.26  2.40  1.88   7.881  0.662
28  Ni  Nickel         58.6934   10   1.24  2.40  1.91   7.640  1.156
29  Cu  Copper         63.546    11   1.32  2.38  1.90   7.726  1.235
30  Zn  Zinc           65.38     12   1.22  2.39  1.65   9.394  -
31  Ga  Gallium        69.723    13   1.22  2.32  1.81   5.999  0.430
32  Ge  Germanium      72.630    14   1.20  2.29  2.01   7.900  1.233
33  As  Arsenic        74.921595 15   1.19  1.88  2.18   9.789  0.804
34  Se  Selenium       78.971    16   1.20  1.82  2.55   9.752  2.021
35  Br  Bromine        79.904    17   1.20  1.86  2.96  11.814  3.364
36  Kr  Krypton        83.798    18   1.16  2.25  3.00  14.000  -
37  Rb  Rubidium       85.4678    1   2.20  3.21  0.82   4.177  0.486
38  Sr  Strontium      87.62      2   1.95  2.84  0.95   5.695  0.052
39  Y   Yttrium        88.905838  3   1.90  2.75  1.22   6.217  0.307
40  Zr  Zirconium      91.224     4   1.75  2.52  1.33   6.634  0.426
41  Nb  Niobium        92.90637   5   1.64  2.56  1.60   6.759  0.893
42  Mo  Molybdenum     95.95      6   1.54  2.45  2.16   7.092  0.746
43  Tc  Technetium     96.906365  7   1.47  2.44  1.90   7.280  0.550
44  Ru  Ruthenium     101.07      8   1.46  2.46  2.20   7.361  1.050
45  Rh  Rhodium       102.90549   9   1.42  2.44  2.28   7.459  1.137
46  Pd  Palladium     106.42     10   1.39  2.15  2.20   8.337  0.562
47  Ag  Silver        107.8682   11   1.45  2.53  1.93   7.576  1.302
48  Cd  Cadmium       112.414    12   1.44  2.49  1.69   8.994  -
49  In  Indium        114.818    13   1.42  2.43  1.78   5.786  0.383
50  Sn  Tin           118.710    14   1.39  2.42  1.96   7.344  1.112
51  Sb  Antimony      121.760    15   1.39  2.47  2.05   8.608  1.047
52  Te  Tellurium     127.60     16   1.38  1.99  2.10   9.010  1.971
53  I   Iodine        126.90447  17   1.39  2.04  2.66  10.451  3.059
54  Xe  Xenon         131.293    18   1.40  2.06  2.60  12.130  -
55  Cs  Caesium       132.905452  1   2.44  3.48  0.79   3.894  0.472
56  Ba  Barium        137.327     2   2.15  3.03  0.89   5.212  0.145
57  La  Lanthanum     138.90547   0   2.07  2.98  1.10   5.577  0.470
58  Ce  Cerium        140.116     0   2.04  2.88  1.12   5.539  0.650
59  Pr  Praseodymium  140.90766   0   2.03  2.92  1.13   5.473  0.962
60  Nd  Neodymium     144.242     0   2.01  2.95  1.14   5.525  1.916
61  Pm  Promethium    144.91276   0   1.99  -     1.13   5.582  0.129
62  Sm  Samarium      150.36      0   1.98  2.90  1.17   5.644  0.162
63  Eu  Europium      151.964     0   1.98  2.87  1.20   5.670  0.864
64  Gd  Gadolinium    157.25      0   1.96  2.83  1.20   6.150  0.137
65  Tb  Terbium       158.925354  0   1.94  2.79  1.20   5.864  1.165
66  Dy  Dysprosium    162.500     0   1.92  2.87  1.22   5.939  0.352
67  Ho  Holmium       164.930329  0   1.92  2.81  1.23   6.022  0.338
68  Er  Erbium        167.259     0   1.89  2.83  1.24   6.108  0.312
69  Tm  Thulium       168.934219  0   1.90  2.79  1.25   6.184  1.029
70  Yb  Ytterbium     173.045     0   1.87  2.80  1.10   6.254  -
71  Lu  Lutetium      174.9668    3   1.87  2.74  1.27   5.426  0.340
72  Hf  Hafnium       178.486     4   1.75  2.63  1.30   6.825  0.017
73  Ta  Tantalum      180.94788   5   1.70  2.53  1.50   7.550  0.322
74  W   Tungsten      183.84      6   1.62  2.57  2.36   7.864  0.815
75  Re  Rhenium       186.207     7   1.51  2.49  1.90   7.834  0.150
76  Os  Osmium        190.23      8   1.44  2.48  2.20   8.438  1.100
77  Ir  Iridium       192.217     9   1.41  2.41  2.20   8.967  1.565
78  Pt  Platinum      195.084    10   1.36  2.29  2.28   8.959  2.125
79  Au  Gold          196.96657  11   1.36  2.32  2.54   9.226  2.309
80  Hg  Mercury       200.592    12   1.32  2.45  2.00  10.438  -
81  Tl  Thallium      204.38     13   1.45  2.47  1.62   6.108  0.377
82  Pb  Lead          207.2      14   1.46  2.60  2.33   7.417  0.364
83  Bi  Bismuth       208.98040  15   1.48  2.54  2.02   7.286  0.942
84  Po  Polonium      208.98243  16   1.40  -     2.00   8.414  1.900
85  At  Astatine      209.98715  17   1.50  -     2.20   9.318  2.416
86  Rn  Radon         222.01758  18   1.50  -     -     10.749  -
87  Fr  Francium      223.01974   1   2.60  -     0.70   4.073  0.486
88  Ra  Radium        226.02541   2   2.21  -     0.90   5.278  0.100
89  Ac  Actinium      227.02775   0   2.15  -     1.10   5.380  0.350
90  Th  Thorium       232.0377    0   2.06  -     1.30   6.307  1.170
91  Pa  Protactinium  231.03588   0   2.00  -     1.50   5.890  0.550
92  U   Uranium       238.02891   0   1.96  -     1.38   6.194  0.315
93  Np  Neptunium     237.04817   0   1.90  -     1.36   6.266  0.480
94  Pu  Plutonium     244.06421   0   1.87  -     1.28   6.026  -
95  Am  Americium     243.06138   0   1.80  -     1.13   5.974  0.100
96  Cm  Curium        247.07035   0   1.69  -     1.28   5.991  0.280
97  Bk  Berkelium     247.07031   0   -     -     1.30   6.198  -
98  Cf  Californium   251.07959   0   -     -     1.30   6.282  -
99  Es  Einsteinium   252.083     0   -     -     1.30   6.368  -
100 Fm  Fermium       257.09511   0   -     -     1.30   6.500  -
101 Md  Mendelevium   258.09843   0   -     -     1.30   6.580  -
102 No  Nobelium      259.101     0   -     -     1.30   6.620  -
103 Lr  Lawrencium    266.0       3   -     -     1.30   4.960  -
104 Rf  Rutherfordium 267.122     4   -     -     -      6.020  -
105 Db  Dubnium       268.126     5   -     -     -      -      -
106 Sg  Seaborgium    269.0       6   -     -     -      -      -
107 Bh  Bohrium       270.133     7   -     -     -      -      -
108 Hs  Hassium       269.1338    8   -     -     -      -      -
109 Mt  Meitnerium    278.156     9   -     -     -      -      -
110 Ds  Darmstadtium  281.165    10   -     -     -      -      -
111 Rg  Roentgenium   282.0      11   -     -     -      -      -
112 Cn  Copernicium   285.177    12   -     -     -      -      -
113 Nh  Nihonium      286.182    13   -     -     -      -      -
114 Fl  Flerovium     289.19     14   -     -     -      -      -
115 Mc  Moscovium     290.0      15   -     -     -      -      -
116 Lv  Livermorium   293.204    16   -     -     -      -      -
117 Ts  Tennessine    294.0      17   -     -     -      -      -
118 Og  Oganesson     294.214    18   -     -     -      -      -
"""

_METALLOIDS = {"B", "Si", "Ge", "As", "Sb", "Te", "At"}
_NONMETALS = {"H", "C", "N", "O", "P", "S", "Se"}
_HALOGENS = {"F", "Cl", "Br", "I", "Ts"}
_NOBLE = {"He", "Ne", "Ar", "Kr", "Xe", "Rn", "Og"}
_LANTHANIDE_Z = range(57, 72)
_ACTINIDE_Z = range(89, 104)


def _f(token: str) -> Optional[float]:
    return None if token == "-" else float(token)


@dataclass(frozen=True)
class Isotope:
    """A single nuclide."""

    mass_number: int
    """Number of nucleons, A = Z + N."""
    atomic_mass_u: float
    """Nuclide mass in unified atomic mass units (Da)."""
    natural_abundance: Optional[float]
    """Mole fraction in a representative terrestrial sample (0-1), or None."""
    spin: Optional[float]
    """Nuclear spin quantum number I in units of hbar, or None if unevaluated."""
    half_life_s: Optional[float]
    """Half life in seconds. ``None`` means observationally stable."""
    atomic_number: int = 0
    """Number of protons, Z."""

    @property
    def is_stable(self) -> bool:
        return self.half_life_s is None

    @property
    def neutrons(self) -> int:
        """N = A - Z."""
        if self.atomic_number < 1:
            raise ValueError(f"Isotope of mass number {self.mass_number} carries no atomic "
                             "number, so its neutron count is unknown.")
        return self.mass_number - self.atomic_number


@dataclass(frozen=True)
class Element:
    """Immutable reference record for a chemical element."""

    number: int
    symbol: str
    name: str
    standard_atomic_weight: Optional[float]
    group: int
    period: int
    block: str
    category: str
    covalent_radius_A: Optional[float]
    vdw_radius_A: Optional[float]
    electronegativity_pauling: Optional[float]
    ionization_energy_eV: Optional[float]
    electron_affinity_eV: Optional[float]
    isotopes: Sequence[Isotope] = field(default_factory=tuple)

    @property
    def electron_configuration(self) -> str:
        from .configuration import configuration_string

        return configuration_string(self.number)

    def isotope(self, mass_number: int) -> Isotope:
        for iso in self.isotopes:
            if iso.mass_number == mass_number:
                return iso
        raise KeyError(
            f"No tabulated isotope {self.symbol}-{mass_number}. "
            f"Tabulated: {[i.mass_number for i in self.isotopes] or 'none'}"
        )

    @property
    def most_abundant_isotope(self) -> Optional[Isotope]:
        candidates = [i for i in self.isotopes if i.natural_abundance]
        if not candidates:
            return None
        return max(candidates, key=lambda i: i.natural_abundance or 0.0)


def _period_of(z: int) -> int:
    for period, limit in enumerate((2, 10, 18, 36, 54, 86, 118), start=1):
        if z <= limit:
            return period
    return 8


def _block_of(z: int, group: int) -> str:
    if group == 0:
        return "f"
    if group in (1, 2):
        return "s" if z != 2 else "s"
    if 3 <= group <= 12:
        return "d"
    return "p"


def _category_of(symbol: str, z: int, group: int) -> str:
    if symbol in _NOBLE:
        return "noble gas"
    if symbol in _HALOGENS:
        return "halogen"
    if symbol in _METALLOIDS:
        return "metalloid"
    if symbol in _NONMETALS:
        return "reactive nonmetal"
    if z in _LANTHANIDE_Z:
        return "lanthanide"
    if z in _ACTINIDE_Z:
        return "actinide"
    if group == 1:
        return "alkali metal"
    if group == 2:
        return "alkaline earth metal"
    if 3 <= group <= 12:
        return "transition metal"
    return "post-transition metal"


_ISOTOPE_TABLE = """
H:  1 1.00782503 0.999885 0.5 -
H:  2 2.01410178 0.000115 1.0 -
H:  3 3.01604928 -        0.5 3.888e8
He: 3 3.01602932 1.34e-6  0.5 -
He: 4 4.00260325 0.9999987 0.0 -
B:  10 10.0129370 0.199 3.0 -
B:  11 11.0093054 0.801 1.5 -
C:  12 12.0000000 0.9893 0.0 -
C:  13 13.00335484 0.0107 0.5 -
C:  14 14.0032420 -      0.0 1.807e11
N:  14 14.0030740 0.99636 1.0 -
N:  15 15.0001089 0.00364 0.5 -
O:  16 15.9949146 0.99757 0.0 -
O:  17 16.9991315 0.00038 2.5 -
O:  18 17.9991596 0.00205 0.0 -
F:  19 18.9984032 1.0 0.5 -
Ne: 20 19.9924402 0.9048 0.0 -
Na: 23 22.9897693 1.0 1.5 -
Mg: 24 23.9850417 0.7899 0.0 -
Mg: 25 24.9858370 0.1000 2.5 -
Mg: 26 25.9825930 0.1101 0.0 -
Al: 27 26.9815384 1.0 2.5 -
Si: 28 27.9769265 0.92223 0.0 -
Si: 29 28.9764947 0.04685 0.5 -
Si: 30 29.9737701 0.03092 0.0 -
P:  31 30.9737620 1.0 0.5 -
S:  32 31.9720712 0.9499 0.0 -
S:  33 32.9714589 0.0075 1.5 -
S:  34 33.9678670 0.0425 0.0 -
Cl: 35 34.96885268 0.7576 1.5 -
Cl: 37 36.96590260 0.2424 1.5 -
Ar: 40 39.9623831 0.996035 0.0 -
K:  39 38.9637065 0.932581 1.5 -
Ca: 40 39.9625909 0.96941 0.0 -
Ti: 46 45.9526277 0.0825 0.0 -
Ti: 47 46.9517588 0.0744 2.5 -
Ti: 48 47.9479420 0.7372 0.0 -
Ti: 49 48.9478657 0.0541 3.5 -
Ti: 50 49.9447869 0.0518 0.0 -
Cr: 52 51.9405062 0.83789 0.0 -
Fe: 56 55.9349363 0.91754 0.0 -
Ni: 58 57.9353424 0.68077 0.0 -
Ni: 60 59.9307859 0.26223 0.0 -
Ni: 61 60.9310556 0.011399 1.5 -
Ni: 62 61.9283454 0.036346 0.0 -
Ni: 64 63.9279668 0.009255 0.0 -
Cu: 63 62.9295977 0.6915 1.5 -
Cu: 65 64.9277897 0.3085 1.5 -
Zn: 64 63.9291420 0.4917 0.0 -
Ga: 69 68.9255735 0.60108 1.5 -
Ga: 71 70.9247026 0.39892 1.5 -
Ge: 70 69.9242499 0.2057 0.0 -
Ge: 72 71.9220758 0.2745 0.0 -
Ge: 73 72.9234590 0.0775 4.5 -
Ge: 74 73.9211778 0.3650 0.0 -
Ge: 76 75.9214027 0.0773 0.0 -
As: 75 74.9215946 1.0 1.5 -
Se: 80 79.9165218 0.4961 0.0 -
Mo: 92 91.9068080 0.1453 0.0 -
Mo: 95 94.9058388 0.1584 2.5 -
Mo: 96 95.9046761 0.1667 0.0 -
Mo: 98 97.9054048 0.2439 0.0 -
Ag: 107 106.9050916 0.51839 0.5 -
Ag: 109 108.9047553 0.48161 0.5 -
In: 113 112.9040618 0.0429 4.5 -
In: 115 114.9038788 0.9571 4.5 4.41e22
Sn: 120 119.9022016 0.3258 0.0 -
Sb: 121 120.9038120 0.5721 2.5 -
Te: 130 129.9062227 0.3408 0.0 2.5e28
Hf: 180 179.9465570 0.3508 0.0 -
Ta: 181 180.9479958 0.99988 3.5 -
W:  182 181.9482039 0.2650 0.0 -
W:  183 182.9502228 0.1431 0.5 -
W:  184 183.9509309 0.3064 0.0 -
W:  186 185.9543628 0.2843 0.0 -
Pt: 194 193.9626809 0.3286 0.0 -
Pt: 195 194.9647917 0.3378 0.5 -
Pt: 196 195.9649521 0.2521 0.0 -
Au: 197 196.9665688 1.0 1.5 -
Pb: 208 207.9766525 0.524 0.0 -
Bi: 209 208.9803991 1.0 4.5 6.0e26
Ne: 21 20.9938467 0.0027 1.5 -
Ne: 22 21.9913851 0.0925 0.0 -
S:  36 35.9670807 0.0001 0.0 -
Ar: 36 35.9675451 0.003336 0.0 -
Ar: 38 37.9627322 0.000629 0.0 -
K:  40 39.9639982 0.000117 4.0 3.938e16
K:  41 40.9618260 0.067302 1.5 -
Ca: 42 41.9586178 0.00647 0.0 -
Ca: 43 42.9587664 0.00135 3.5 -
Ca: 44 43.9554815 0.02086 0.0 -
Ca: 46 45.9536880 0.00004 0.0 -
Ca: 48 47.9525228 0.00187 0.0 2.0e27
Cr: 50 49.9460418 0.04345 0.0 -
Cr: 53 52.9406481 0.09501 1.5 -
Cr: 54 53.9388792 0.02365 0.0 -
Fe: 54 53.9396090 0.05845 0.0 -
Fe: 57 56.9353928 0.02119 0.5 -
Fe: 58 57.9332744 0.00282 0.0 -
Zn: 66 65.9260338 0.2773 0.0 -
Zn: 67 66.9271277 0.0404 2.5 -
Zn: 68 67.9248446 0.1845 0.0 -
Zn: 70 69.9253192 0.0061 0.0 -
Se: 74 73.9224759 0.0089 0.0 -
Se: 76 75.9192137 0.0937 0.0 -
Se: 77 76.9199142 0.0763 0.5 -
Se: 78 77.9173092 0.2377 0.0 -
Se: 82 81.9166995 0.0873 0.0 2.9e27
Mo: 94 93.9050849 0.0915 0.0 -
Mo: 97 96.9060181 0.0960 2.5 -
Mo: 100 99.9074718 0.0982 0.0 2.2e26
Sn: 112 111.9048239 0.0097 0.0 -
Sn: 114 113.9027827 0.0066 0.0 -
Sn: 115 114.9033447 0.0034 0.5 -
Sn: 116 115.9017428 0.1454 0.0 -
Sn: 117 116.9029540 0.0768 0.5 -
Sn: 118 117.9016066 0.2422 0.0 -
Sn: 119 118.9033112 0.0859 0.5 -
Sn: 122 121.9034438 0.0463 0.0 -
Sn: 124 123.9052766 0.0579 0.0 -
Sb: 123 122.9042132 0.4279 3.5 -
Te: 120 119.9040593 0.0009 0.0 -
Te: 122 121.9030435 0.0255 0.0 -
Te: 123 122.9042698 0.0089 0.5 -
Te: 124 123.9028171 0.0474 0.0 -
Te: 125 124.9044299 0.0707 0.5 -
Te: 126 125.9033109 0.1884 0.0 -
Te: 128 127.9044613 0.3174 0.0 6.9e31
Hf: 174 173.9400461 0.0016 0.0 6.3e22
Hf: 176 175.9414076 0.0526 0.0 -
Hf: 177 176.9432277 0.1860 3.5 -
Hf: 178 177.9437058 0.2728 0.0 -
Hf: 179 178.9458232 0.1362 4.5 -
W:  180 179.9467108 0.0012 0.0 5.7e25
Pt: 190 189.9599297 0.00012 0.0 2.0e19
Pt: 192 191.9610387 0.00782 0.0 -
Pt: 198 197.9678949 0.07356 0.0 -
Pb: 204 203.9730440 0.014 0.0 -
Pb: 206 205.9744657 0.241 0.0 -
Pb: 207 206.9758973 0.221 0.5 -
"""


def _parse_isotopes() -> dict:
    numbers = {line.split()[1]: int(line.split()[0])
               for line in _ELEMENT_TABLE.strip().splitlines() if line.strip()}
    out: dict = {}
    for raw in _ISOTOPE_TABLE.strip().splitlines():
        line = raw.strip()
        if not line:
            continue
        sym, rest = line.split(":", 1)
        a, mass, abundance, spin, half_life = rest.split()
        out.setdefault(sym.strip(), []).append(
            Isotope(
                mass_number=int(a),
                atomic_mass_u=float(mass),
                natural_abundance=_f(abundance),
                spin=_f(spin),
                half_life_s=_f(half_life),
                atomic_number=numbers[sym.strip()],
            )
        )
    return out


def _build() -> tuple:
    isotopes = _parse_isotopes()
    elements = []
    for raw in _ELEMENT_TABLE.strip().splitlines():
        parts = raw.split()
        if not parts:
            continue
        z = int(parts[0])
        symbol = parts[1]
        group = int(parts[4])
        elements.append(
            Element(
                number=z,
                symbol=symbol,
                name=parts[2],
                standard_atomic_weight=_f(parts[3]),
                group=group,
                period=_period_of(z),
                block=_block_of(z, group),
                category=_category_of(symbol, z, group),
                covalent_radius_A=_f(parts[5]),
                vdw_radius_A=_f(parts[6]),
                electronegativity_pauling=_f(parts[7]),
                ionization_energy_eV=_f(parts[8]),
                electron_affinity_eV=_f(parts[9]),
                isotopes=tuple(sorted(isotopes.get(symbol, ()), key=lambda i: i.mass_number)),
            )
        )
    return tuple(elements)


ELEMENTS: tuple = _build()
BY_SYMBOL = {e.symbol: e for e in ELEMENTS}
BY_NUMBER = {e.number: e for e in ELEMENTS}
BY_NAME = {e.name.lower(): e for e in ELEMENTS}
