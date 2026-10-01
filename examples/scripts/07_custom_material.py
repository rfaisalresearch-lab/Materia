"""Add a material at runtime and build a surface from it.

The same JSON a file on disk would contain. Drop it in ~/.materia/materials to
make it permanent.
"""

import numpy as np

definition = {
    "schema_version": "1.0",
    "id": "example_rocksalt_mgo",
    "name": "Magnesium oxide (example)",
    "formula": "MgO",
    "aliases": ["MgO", "periclase"],
    "category": "oxide insulator",
    "structure": {
        "prototype": "rocksalt",
        "space_group": {"number": 225, "symbol": "Fm-3m"},
        "lattice": {
            "a": 4.2117, "unit": "A", "system": "cubic", "temperature_K": 300,
            "source": "CRC Handbook of Chemistry and Physics, 104th ed.",
        },
        "basis": [
            {"element": "Mg", "fractional": [0.0, 0.0, 0.0], "label": "Mg", "wyckoff": "4a"},
            {"element": "Mg", "fractional": [0.0, 0.5, 0.5], "label": "Mg", "wyckoff": "4a"},
            {"element": "Mg", "fractional": [0.5, 0.0, 0.5], "label": "Mg", "wyckoff": "4a"},
            {"element": "Mg", "fractional": [0.5, 0.5, 0.0], "label": "Mg", "wyckoff": "4a"},
            {"element": "O", "fractional": [0.5, 0.0, 0.0], "label": "O", "wyckoff": "4b"},
            {"element": "O", "fractional": [0.0, 0.5, 0.0], "label": "O", "wyckoff": "4b"},
            {"element": "O", "fractional": [0.0, 0.0, 0.5], "label": "O", "wyckoff": "4b"},
            {"element": "O", "fractional": [0.5, 0.5, 0.5], "label": "O", "wyckoff": "4b"},
        ],
    },
    "orientations": [[1, 0, 0], [1, 1, 0], [1, 1, 1]],
    "properties": {
        "density": {"value": 3.58, "unit": "g/cm^3", "conditions": "300 K",
                    "source": "CRC Handbook of Chemistry and Physics, 104th ed."},
        "band_gap": {"value": 7.8, "unit": "eV", "conditions": "300 K, direct",
                     "source": "D. M. Roessler and W. C. Walker, Phys. Rev. 159 (1967) 733"},
        "band_gap_type": {"value": "direct", "unit": "",
                          "source": "classification following the band-gap reference"},
    },
    "recommended_models": {"relax": "unsupported", "electronic": "unsupported",
                           "stm": "unsupported", "high-fidelity": "external:gpaw"},
    "references": ["D. M. Roessler and W. C. Walker, Phys. Rev. 159 (1967) 733"],
    "license": "CC0-1.0",
    "provenance_note": "Example runtime material. Verify the generated cell before use.",
}

from materia.materials import default_library

registered = default_library().register_raw(definition, "<example script>", overwrite=True)
print("registered:", registered.id)

magnesia = materials.load("example_rocksalt_mgo", orientation="100")
crystal = magnesia.bulk()
print(f"{len(crystal)} atoms, {crystal.structure.formula()}")

neighbours = measure.neighbour_list(2.5, crystal)
print(f"coordination {set(neighbours.counts())}, "
      f"nearest neighbour {float(neighbours.d.min()):.4f} A "
      f"(expected a/2 = {4.2117 / 2:.4f} A)")

surface = magnesia.create_surface(size=(2, 2, 3), vacuum_angstrom=12)
print(f"MgO(100) slab: {len(surface)} atoms")

print("\nThis material has no shipped potential, so energetics are refused:")
try:
    surface.energy()
except Exception as exc:
    print("   ", exc)
