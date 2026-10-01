"""Example Materia plug-in.

Demonstrates every extension point the plug-in API offers:

* a new material definition (a hypothetical strained-silicon variant)
* a new solver (an Einstein-solid harmonic model)
* a new analysis routine (surface coordination histogram)
* a new colour palette
* a new exporter

Copy this directory to ``~/.materia/plugins/`` and edit it.
"""

from __future__ import annotations

import json
import os

PLUGIN_API_VERSION = "1.0"

PLUGIN = {
    "id": "example_plugin",
    "name": "Materia example plug-in",
    "version": "0.1.0",
    "description": "Reference plug-in showing every extension point.",
    "author": "Materia contributors",
    "license": "MIT",
    "api_version": "1.0",
}

HERE = os.path.dirname(os.path.abspath(__file__))


def _harmonic_solver():
    """An Einstein-solid solver, registered as an example of a custom model."""
    import numpy as np

    from materia.physics.potentials import Harmonic
    from materia.solvers.classical import ClassicalSolver

    class EinsteinSolver(ClassicalSolver):
        name = "example/einstein-solid"

        def __init__(self, k_eV_A2: float = 2.0):
            super().__init__(Harmonic(np.zeros((0, 3)), k_eV_A2))
            self.k = k_eV_A2
            self.name = "example/einstein-solid"

        def single_point(self, structure):
            self.potential = Harmonic(structure.positions.copy(), self.k)
            return super().single_point(structure)

    return EinsteinSolver()


def _coordination_histogram(structure, cutoff_A: float = 3.0):
    """Analysis routine: coordination-number histogram of a structure."""
    import numpy as np

    from materia.physics.neighbors import neighbor_list
    from materia.provenance import Fidelity, Origin, Provenance, Result

    nl = neighbor_list(structure.positions, structure.cell, cutoff_A)
    counts = nl.counts()
    values, occurrences = np.unique(counts, return_counts=True)
    return Result(
        name="coordination_histogram",
        value={"coordination": values.tolist(), "atoms": occurrences.tolist()},
        unit="count",
        provenance=Provenance(
            model="example_plugin/coordination-histogram",
            fidelity=Fidelity.TIER0_STRUCTURAL,
            origin=Origin.CALCULATED,
            approximations=[f"Neighbours counted within a fixed {cutoff_A} A cutoff."],
            parameters={"cutoff_A": cutoff_A},
        ),
    )


def _write_summary(path: str, structure) -> str:
    """Exporter: a one-line plain-text summary of a structure."""
    with open(path, "w") as fh:
        fh.write(f"{structure.formula()}\t{len(structure)} atoms\t"
                 f"cell {structure.cell.lengths.tolist()}\n")
    return path


def register(ctx):
    ctx.add_material_path(os.path.join(HERE, "materials"))
    ctx.add_solver("example/einstein-solid", _harmonic_solver,
                   meta={"tier": "non-physical",
                         "note": "Springs to the current positions; for testing only."})
    ctx.add_analysis("coordination-histogram", _coordination_histogram)
    ctx.add_exporter("summary", _write_summary)

    from materia.visualization.palette import Palette
    ctx.add_palette(Palette(
        id="example-teal",
        name="Example teal",
        stops=((0.0, (4, 18, 22)), (0.5, (30, 120, 130)), (1.0, (222, 250, 252))),
        perceptual="monotonic luminance",
        note="Contributed by the example plug-in.",
    ))
    ctx.add_panel({"id": "example-panel", "title": "Example plug-in",
                   "placement": "right", "kind": "info",
                   "body": "This panel was contributed by example_plugin."})
