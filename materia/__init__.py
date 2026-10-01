"""Materia: atomic-scale wafer exploration and scanning-probe simulation.

Materia lets you start from a macroscopic semiconductor wafer, descend
through progressively smaller scales, and inspect or manipulate individual
surface atoms through a simulated scanning probe microscope.

It is organised in explicit fidelity tiers (see ``docs/FIDELITY_TIERS.md``).
Every number it reports carries a provenance record naming the model that
produced it, the approximations that model makes, its convergence state and
whether the value is calculated, estimated, illustrative or imported.  When a
model cannot answer a question, Materia says so and names the solvers that
could, rather than producing a plausible-looking number.

Quick start
-----------
>>> import materia as asc                                   # doctest: +SKIP
>>> lab = asc.Lab()
>>> si = lab.materials.load("silicon", orientation="111")
>>> surface = si.create_surface(size=(4, 4, 4), vacuum_angstrom=14)
>>> scan = lab.microscope.stm_scan(surface, bias_volts=1.0, resolution=(256, 256))
>>> lab.io.write_image("si111.png", scan)
"""

from .version import MATERIAL_SCHEMA_VERSION, PLUGIN_API_VERSION, PROJECT_SCHEMA_VERSION, __version__

from . import core_model, elements, materials, provenance
from .core_model import Cell, Selection, Structure
from .materials import default_library, load as load_material
from .provenance import Fidelity, Origin, Provenance, Result

from . import solvers as _solvers
from .python_api import Lab, ScriptRunner
from .project_format import Project

__all__ = [
    "__version__", "PROJECT_SCHEMA_VERSION", "MATERIAL_SCHEMA_VERSION",
    "PLUGIN_API_VERSION",
    "Lab", "Project", "ScriptRunner",
    "Structure", "Cell", "Selection",
    "Result", "Provenance", "Origin", "Fidelity",
    "load_material", "default_library",
    "core_model", "elements", "materials", "provenance",
]
