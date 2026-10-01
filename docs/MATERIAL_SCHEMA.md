# Material definition schema

Version 1.0. A material is a plain JSON document. Users and plug-ins add
materials by dropping files into a search path; nothing in the program is
specialised to the shipped list.

## Search order

1. Paths passed explicitly to `MaterialLibrary`
2. `$MATERIA_MATERIALS` (os.pathsep-separated)
3. `~/.materia/materials`
4. The built-in library inside the package

First match wins for a given `id`, so a user file shadows a shipped one.

## Document

```json
{
  "schema_version": "1.0",
  "id": "silicon",
  "name": "Silicon",
  "formula": "Si",
  "aliases": ["Si", "c-Si"],
  "category": "elemental semiconductor",

  "structure": {
    "prototype": "diamond-cubic",
    "space_group": { "number": 227, "symbol": "Fd-3m" },
    "lattice": {
      "a": 5.4310205, "b": null, "c": null,
      "alpha": 90, "beta": 90, "gamma": 90,
      "unit": "A",
      "system": "cubic",
      "temperature_K": 295.7,
      "source": "CODATA-recommended silicon lattice parameter"
    },
    "basis": [
      { "element": "Si", "fractional": [0.0, 0.0, 0.0],
        "label": "Si(a)", "wyckoff": "8a", "occupancy": 1.0, "role": "bulk" }
    ]
  },

  "orientations": [[1,0,0], [1,1,0], [1,1,1]],

  "terminations": [
    { "id": "H-terminated-111", "orientation": [1,1,1], "passivation": "H",
      "description": "Monohydride Si(111):H-1x1.", "implemented": true }
  ],

  "reconstructions": [
    { "id": "7x7-DAS", "orientation": [1,1,1],
      "description": "Takayanagi dimer-adatom-stacking-fault reconstruction.",
      "implemented": false,
      "reference": "K. Takayanagi et al., Surf. Sci. 164 (1985) 367",
      "note": "Not implemented. Requesting it returns an explicit unsupported record." },

    { "id": "2x1-dimer", "orientation": [1,0,0],
      "description": "Rows of surface dimers on Si(100).",
      "implemented": true,
      "reference": "R. E. Schlier and H. E. Farnsworth, J. Chem. Phys. 30 (1959) 917",
      "method": "Generator establishes the pairing; geometry comes from relaxation.",
      "reference_geometry": {
        "bond_length_A": { "value": 2.24, "uncertainty": 0.08, "unit": "A",
          "method": "LEED structure analysis, room temperature",
          "source": "H. Over et al., Phys. Rev. B 55 (1997) 4731" }
      } }
  ],

  "properties": {
    "band_gap": { "value": 1.12, "unit": "eV",
                  "conditions": "300 K, indirect",
                  "source": "S. M. Sze, Physics of Semiconductor Devices",
                  "uncertainty": null, "note": "", "origin": "reference" }
  },

  "recommended_models": {
    "relax": "stillinger-weber",
    "electronic": "tight-binding-sp3s",
    "stm": "tersoff-hamann",
    "high-fidelity": "external:gpaw"
  },

  "references": ["..."],
  "license": "CC0-1.0",
  "provenance_note": "Compiled from the cited primary sources."
}
```

## Field reference

| Field | Required | Notes |
| --- | --- | --- |
| `schema_version` | yes | Major version must match the program's |
| `id` | yes | Unique, lowercase with underscores |
| `name`, `formula` | yes | Display name and chemical formula |
| `aliases` | no | Extra lookup keys |
| `category` | no | Free text, used for search |
| `structure.prototype` | yes | Free text; the builder reads the basis, not the prototype |
| `structure.space_group` | no | Recorded and displayed, not used to generate |
| `structure.lattice.a` | yes | Ångström. `unit` must be `"A"` |
| `structure.lattice.system` | no | One of cubic, tetragonal, orthorhombic, hexagonal, trigonal, monoclinic, triclinic, rhombohedral, amorphous, layered-2d. Constrains b, c and the angles |
| `structure.basis` | yes | Every site explicitly, fractional coordinates in [0, 1) |
| `structure.basis[].occupancy` | no | Finite value from 0 through 1. Values below 1 require an explicit `occupancy_seed` and a repeated finite cell; see `ALLOYS.md` |
| `orientations` | no | Offered in the interface; any (hkl) still works |
| `terminations` | no | Named cuts, each marked implemented or not |
| `reconstructions` | no | **Mark `implemented: false` for anything you have not generated**, with a reference |
| `reconstructions[].method` | no | One line on how the geometry is obtained, shown in the interface |
| `reconstructions[].reference_geometry` | no | Published structural parameters, **for comparison only**, never used to place atoms |
| `properties` | no | Each entry needs `value` and a `source` |
| `recommended_models` | no | Resolved against the solver registry; unknown names fall back to automatic selection |
| `license`, `provenance_note` | recommended | Required by the test suite for shipped materials |

## Rules the validator enforces

* Required fields present; `schema_version` major version compatible.
* `unit` is `"A"`; other length units are refused by name.
* Every basis element is a real element.
* Every basis site has three fractional components.
* Every occupancy lies from zero through one.
* `a > 0`; the lattice system is one of the recognised set.
* Every `reference_geometry` entry has a numeric `value` and a non-empty
  `source`. A published number without a citation is refused.

A malformed file is **reported, not raised**: the library records the error
against the file path and keeps loading the others, and the interface shows the
message in the warnings panel.

## Rules the test suite enforces for shipped materials

* Every property has a non-empty `source`.
* Every numeric property has a unit, except explicitly dimensionless ones.
* Fractional coordinates lie in [0, 1).
* `license` and `provenance_note` are present.
* The generated cell reproduces cited geometry, bond lengths, coordination
  numbers, density, within a stated tolerance.
* `implemented: true` on a reconstruction is resolved against the generator
  registry, not trusted. A file cannot claim a reconstruction the build has no
  generator for; the interface shows it as unavailable with the reason.

That last rule is what catches a wrong space-group setting. The α-quartz entry
in this release was wrong on the first attempt and the validation case is what
found it.

## Writing your own

```bash
mkdir -p ~/.materia/materials
cp materia/materials/library/silicon.json ~/.materia/materials/my_material.json
# edit id, name, lattice and basis
materia materials --search my_material
```

Then check it:

```python
from materia.materials import default_library
from materia.structure_builder import bulk
from materia.physics.neighbors import neighbor_list

crystal = bulk(default_library().get("my_material"))
nl = neighbor_list(crystal.positions, crystal.cell, 3.0)
print(crystal.formula(), nl.counts(), nl.d.min())
```

If the nearest-neighbour distance is not what your source says it should be,
the basis is wrong.
