"""Build a Si(111) surface, relax it and image it.

Run with:
    materia run examples/scripts/01_silicon_surface.py
or paste into the embedded Python console.
"""

import numpy as np

silicon = materials.load("silicon", orientation="111")
print(silicon)
print("band gap:", silicon.property("band_gap"))

surface = silicon.create_surface(size=(4, 4, 4), vacuum_angstrom=15)
info = surface.info()
print(f"{len(surface)} atoms")
print(f"in-plane cell  {info['in_plane_a_A']:.4f} x {info['in_plane_b_A']:.4f} A"
      f" at {info['in_plane_angle_deg']:.1f} deg")
print(f"layer spacing  {info['interplanar_spacing_A']:.4f} A")
print(f"termination    {info['termination']} ({info['termination_mode']})")

energy = surface.energy()
print(energy)
print("model:", energy.provenance.model)
for approximation in energy.provenance.approximations:
    print("   -", approximation)

relaxed = surface.relax(model="recommended", fmax=0.02, steps=300)
print("relaxation:", relaxed.convergence.message)
print(f"largest displacement {relaxed.extra['max_displacement_A']:.4f} A")

scan = microscope.stm_scan(
    surface,
    bias_volts=1.0,
    current_nA=0.5,
    resolution=(256, 256),
    noise="realistic",
    seed=0,
)
topography = scan.channel("topography")
print(f"corrugation {float(np.ptp(topography)):.4f} A")
print("feedback:", scan.convergence.message)

features = scan.detect_features()
kinds = {}
for feature in features:
    kinds[feature.kind] = kinds.get(feature.kind, 0) + 1
print("features:", kinds)

view.display(scan, palette="silver", title="Si(111) constant current, +1.0 V")
