"""Solvers: the layered physics engines and their registry.

Importing this package registers the built-in solvers.  Plug-ins register
additional ones through :func:`materia.solvers.registry.register`.
"""

from .base import (
    ALL_CAPABILITIES,
    CAPABILITY_DOC,
    Capability,
    Solver,
    SolverResult,
    SupportReport,
)
from .classical import ClassicalSolver, lennard_jones, stillinger_weber
from .electrostatics import ElectrostaticsSolver
from .registry import (
    available,
    create,
    describe_all,
    register,
    resolve_recommended,
    solvers_providing,
    unregister,
)
from .gpaw_driver import GPAWSolver
from .tight_binding import MODELS as TIGHT_BINDING_MODELS
from .tight_binding import TightBinding


def _register_builtin() -> None:
    from ..physics.potentials import SW_PARAMETERS

    for element in sorted(SW_PARAMETERS):
        register(
            f"stillinger-weber-{element.lower()}",
            (lambda el=element: stillinger_weber(el)),
            overwrite=True,
            meta={"tier": "tier1-classical", "element": element},
        )
    from ..physics.potentials import SW_VARIANTS, StillingerWeber

    for element, variant in sorted(SW_VARIANTS):
        register(f"stillinger-weber-{element.lower()}-{variant}",
                 (lambda e=element, v=variant: ClassicalSolver(StillingerWeber(e, variant=v))),
                 overwrite=True,
                 meta={"tier": "tier1-classical", "element": element, "variant": variant,
                       "note": SW_VARIANTS[(element, variant)].source})
    register("stillinger-weber", lambda: stillinger_weber("Si"), overwrite=True,
             meta={"tier": "tier1-classical", "element": "Si",
                   "note": "Defaults to the silicon parameterisation."})

    def _lj(material=None, params=None, cutoff_A: float = 10.0):
        if material is None and params is None:
            params = {"Si": (0.5378, 2.1571)}
        return lennard_jones(material=material, params=params, cutoff_A=cutoff_A)

    register("lennard-jones", _lj, overwrite=True,
             meta={"tier": "tier1-classical",
                   "note": "Generic pair potential; parameters must be supplied or "
                           "derived from a material."})

    for name in sorted(TIGHT_BINDING_MODELS):
        register(f"tight-binding/{name}", (lambda n=name: TightBinding(n)),
                 overwrite=True, meta={"tier": "tier2-semi-empirical", "model": name})

    from ..physics import eam as eam_module

    for entry in eam_module.catalog():
        register(f"eam/{entry['id']}",
                 (lambda pid=entry["id"]: ClassicalSolver(eam_module.load_shipped(pid))),
                 overwrite=True,
                 meta={"tier": "tier1-classical", "elements": list(entry["elements"]),
                       "family": entry["family"], "license": entry["license"],
                       "sha256": entry["sha256"],
                       "note": "Embedded-atom potential read from a checksummed setfl file."})

    from ..physics import rigid_ion

    for entry in rigid_ion.catalog():
        register(f"rigid-ion/{entry['id']}",
                 (lambda pid=entry["id"]: ClassicalSolver(
                     rigid_ion.RigidIon(rigid_ion.SHIPPED[pid]))),
                 overwrite=True,
                 meta={"tier": "tier1-classical", "elements": entry["elements"],
                       "parameter_digest": entry["digest"],
                       "note": "Fixed point charges with Ewald summation plus Buckingham "
                               "short-range terms; fixed-cell relaxation and dynamics."})

    from . import molecular

    for model, (engine, _) in molecular.MODELS.items():
        register(model, (lambda m=model: molecular.create(m)), overwrite=True,
                 meta={"tier": "molecular-engine", "engine": engine,
                       "available": molecular.available(engine),
                       "license_note": molecular.ENGINE_LICENCES[engine],
                       "install_hint": molecular.mol.INSTALL[engine],
                       "note": "External open-source engine driven in process; results "
                               "record the engine and its version."})

    from ..physics.tersoff import Tersoff

    for key, (filename, elements) in {"tersoff/si": ("Si.tersoff", ["Si"]),
                                      "tersoff/sic": ("SiC.tersoff", ["C", "Si"]),
                                      "tersoff/sicge": ("SiCGe.tersoff", ["C", "Ge", "Si"]),
                                      "tersoff/bnc": ("BNC.tersoff", ["B", "C", "N"])}.items():
        register(key, (lambda f=filename, e=elements: ClassicalSolver(Tersoff(f, elements=e))),
                 overwrite=True,
                 meta={"tier": "tier1-classical", "elements": elements, "file": filename,
                       "note": "Tersoff bond-order potential in the LAMMPS form; the LAMMPS "
                               "parameter file is downloaded into ~/.cache/materia on first "
                               "use (GPL, not shipped)."})

    from .ml_driver import MLPotential, find_python as _ml_python

    for size in ("small", "medium"):
        register(f"ml/mace-mp-0-{size}", (lambda s=size: ClassicalSolver(MLPotential(s))),
                 overwrite=True,
                 meta={"tier": "machine-learned", "engine": "MACE",
                       "available": _ml_python() is not None,
                       "license_note": "MACE and the MACE-MP-0 weights are MIT licensed; "
                                       "they run in a separate interpreter.",
                       "note": "Universal machine-learned surrogate of PBE(+U); phonons "
                               "measured about 25 percent soft for silicon."})

    register("electrostatics/ewald", ElectrostaticsSolver, overwrite=True,
             meta={"tier": "tier1-classical",
                   "note": "Point-charge electrostatics; needs an explicit charge model."})

    from . import external
    external.register_adapters()

    from .gpaw_driver import GPAWSolver

    register("external:gpaw", GPAWSolver, overwrite=True,
             meta={"tier": "tier3-external-first-principles", "external": True,
                   "install_hint": ("conda create -n materia-gpaw -c conda-forge "
                                    "python=3.13 gpaw gpaw-data, then point "
                                    "MATERIA_GPAW_PYTHON at that interpreter."),
                   "license_note": ("GPAW is LGPL-3.0-or-later and runs as a separate "
                                    "program; the PAW datasets carry their own "
                                    "licence."),
                   "note": "Single-point energy and forces only."})


_register_builtin()

__all__ = [
    "Solver", "SolverResult", "SupportReport", "Capability", "ALL_CAPABILITIES",
    "CAPABILITY_DOC", "ClassicalSolver", "stillinger_weber", "lennard_jones",
    "TightBinding", "TIGHT_BINDING_MODELS", "GPAWSolver", "ElectrostaticsSolver",
    "register", "unregister", "available", "create", "describe_all",
    "solvers_providing", "resolve_recommended",
]
