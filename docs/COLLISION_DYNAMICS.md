# Collision dynamics

Materia can prepare two atomistic structures on a collision course, run their
classical dynamics, save every sampled coordinate and velocity, replay the run,
pause on any frame, inspect fragments, and reopen the trajectory from a project.

## Python setup

```python
projectile = materials.load("copper").bulk(repeat=(2, 2, 2), activate=False)
target = materials.load("copper").bulk(repeat=(3, 3, 3), activate=False)

impact = collisions.prepare(
    projectile,
    target,
    relative_speed_A_fs=0.08,
    direction=(1, 0, 0),
    impact_parameter_A=2.0,
    gap_A=3.0,
    padding_A=10.0,
)

run = collisions.run(
    impact,
    potential="Cu-Zhou04",
    steps=800,
    dt_fs=0.1,
    sample_every=4,
)
print(run["energy"].extra["run_id"])
```

`collisions.prepare` gives the two bodies zero total momentum. The relative
speed is split by total projectile and target mass, so unequal bodies move at
different laboratory-frame speeds. The recorded centre-of-mass kinetic energy
is calculated from the reduced mass.

`collisions.run` preserves those velocities and runs NVE velocity-Verlet
dynamics. It uses the selected EAM potential, so the shipped path currently
covers Cu, Au and W. Other elements require a compatible user-supplied setfl
potential or a future external backend.

## Playback and fragments

The EAM panel exposes play, pause, frame scrubbing, playback speed and a final
structure reset. A frame reports positions, velocities, physical time, energy,
temperature and contact-connected fragments. Fragment colours use the Materia
blue palette. Bonds are not drawn during playback because a static bond list
would be misleading when contacts break and form.

Frame positions and velocities are stored as checksummed chunked arrays in the
project, not embedded in result JSON. The interface fetches one frame at a time.
This keeps browser memory bounded and detects damaged trajectory chunks when a
project is reopened.

Fragment membership is a geometric diagnostic. Two atoms are connected when
their separation is within 1.35 times the sum of their covalent radii. It is not
a quantum bond order and must not be interpreted as one.

## Physical scope

This is classical nuclear motion under an interatomic potential. It is suited
to questions such as low-energy cluster impact, sputtering within the fitted
range of a potential, defect generation and nanostructure collision dynamics.

It does not model quarks, hadrons, nuclear reactions, relativistic particle
showers or Higgs bosons. Those require a separate high-energy stack such as an
event generator and detector-transport code. They cannot be made accurate by
zooming further into a classical atom rendering.

At impact energies or local environments far outside the potential's training
domain, the calculation can remain numerically stable while being physically
wrong. A reactive force field, machine-learned potential or first-principles
dynamics backend must be selected when bond breaking or charge transfer matters.

