"""Band structures of the parameterised materials.

Reproduces the values the tight-binding parameters were fitted to. These are a
statement about the fit, not a prediction.
"""

import numpy as np

from materia.structure_builder.lattice import bulk
from materia.structure_builder.primitive import primitive_structure

for material_id, path, labels, expectation in (
    ("silicon", [(0.5, 0.5, 0.5), (0, 0, 0), (0.5, 0, 0.5)], ["L", "G", "X"],
     "valence bandwidth 12.5 eV, indirect gap near 0.85 Gamma-X"),
    ("gallium_arsenide", [(0.5, 0.5, 0.5), (0, 0, 0), (0.5, 0, 0.5)], ["L", "G", "X"],
     "direct gap 1.55 eV at Gamma"),
    ("graphene", [(0, 0, 0), (1 / 3, 1 / 3, 0), (0.5, 0, 0)], ["G", "K", "M"],
     "gapless at K, bandwidth 6|t| = 16.2 eV"),
):
    definition = materials.load(material_id).definition
    cell = primitive_structure(bulk(definition))
    out = lab.band_structure(cell, path=path, labels=labels, n_per_segment=60)

    bands = out["band_structure"].value
    gap = out["band_gap"]
    values = np.asarray(bands["bands_eV"])

    print(f"\n{definition.name}  ({len(cell)} atoms in the primitive cell)")
    print(f"   expected: {expectation}")
    print(f"   gap along the path: {gap.value:.4f} eV "
          f"({'direct' if gap.extra['direct'] else 'indirect'})")
    print(f"   total bandwidth   : {values.max() - values.min():.4f} eV")
    print(f"   model             : {out.solver}")
    print(f"   caveat            : {gap.extra['note']}")

    view.plot(bands["k_coord"], values[:, 0].tolist(),
              title=f"{definition.name}: lowest band",
              xlabel="k path", ylabel="energy / eV")
