"""Rigid-ion alpha-quartz: relax, inspect the Coulomb check, meet the collapse refusal.

Run with:
    materia run examples/scripts/19_rigid_ion_silica.py
or paste into the embedded Python console.

The BKS potential (van Beest, Kramer and van Santen 1990) gives Si +2.4 e and
O -1.2 e and adds Buckingham terms for Si-O and O-O. The cell is held at the
measured quartz cell; only the atoms move.
"""

quartz = materials.load("silicon_dioxide").bulk()
relaxed = quartz.relax(model="recommended", fmax=1e-3, steps=2000)
print(relaxed.convergence.message)
energy = project.results["relax::energy"]
checks = energy.extra["potential_checks"]
print(f"model {energy.provenance.model}: {energy.value / 3:.4f} eV per SiO2")
print(f"  short range {checks['components_eV']['short_range']:.4f} eV, "
      f"Coulomb {checks['components_eV']['coulomb']:.4f} eV")
print("  " + checks["ewald_message"])
for pair, distance in checks["closest_pair_A"].items():
    print(f"  closest {pair}: {distance:.4f} A, "
          f"{checks['barrier_margin_A'][pair]:.4f} A outside its barrier")

squeezed = materials.load("silicon_dioxide").bulk(activate=False)
s = squeezed.structure
positions = s.positions.copy()
bond = positions[3] - positions[0]
positions[3] = positions[0] + bond / (bond @ bond) ** 0.5 * 1.1
s.positions = positions
refused = squeezed.relax(model="recommended")
print("supported:", refused.supported)
print("refused:", refused.unsupported_reason)
