"""Image the same surface at several biases.

Positive bias probes empty states, negative bias probes filled states. The
electronic structure is cached between scans, so only the first pays for the
eigensolve.
"""

import numpy as np

silicon = materials.load("silicon", orientation="111")
surface = silicon.create_surface(size=(4, 4, 4), vacuum_angstrom=15)

print(f"{'bias / V':>9} {'corrugation / A':>16} {'states':>8} {'cached':>7}")
for bias in (-1.5, -0.8, 0.8, 1.5):
    scan = microscope.stm_scan(surface, bias_volts=bias, resolution=(160, 160), seed=7)
    corrugation = float(np.ptp(scan.channel("topography")))
    parameters = scan.provenance.parameters
    print(f"{bias:>9.1f} {corrugation:>16.4f} "
          f"{parameters['electronic_states_in_window']:>8} "
          f"{str(parameters['electronic_from_cache']):>7}")
    name = f"si111_bias_{'m' if bias < 0 else 'p'}{abs(bias):.1f}.png"
    io.write_image(name, scan, palette="silver")

print("\nThe images differ because the bias window selects different states,")
print("not because anything about the atoms changed.")
