# Plug-in API

Version 1.0. A plug-in is a Python package or module that exposes
`register(ctx)`. It can add materials, solvers, importers, exporters, analyses,
colour maps and interface panels, through the same calls the built-ins use.

## Discovery

1. Entry points in the `materia.plugins` group (installed packages)
2. `$MATERIA_PLUGINS` (os.pathsep-separated directories)
3. `~/.materia/plugins`
4. The `plugins/` directory of the repository, for development

## Contract

```python
PLUGIN_API_VERSION = "1.0"

PLUGIN = {
    "id": "my_plugin",
    "name": "My plug-in",
    "version": "0.1.0",
    "description": "What it adds.",
    "author": "You",
    "license": "MIT",
    "api_version": "1.0",
}

def register(ctx):
    ctx.add_material_path("/path/to/materials")
    ctx.add_solver("my-solver", MySolverFactory)
    ctx.add_analysis("my-analysis", run_my_analysis)
    ctx.add_exporter("myfmt", write_myfmt)
    ctx.add_palette(my_palette)
    ctx.add_panel({"id": "my-panel", "title": "My panel", "placement": "right"})
```

A plug-in whose major `api_version` does not match is **not loaded**, and the
reason is shown. A plug-in whose `register` raises is reported with its
traceback; it does not stop the others.

## The context

| Method | Adds |
| --- | --- |
| `add_material_path(path)` | A directory of material JSON files |
| `add_material(raw, source)` | An in-memory definition (validated) |
| `add_solver(name, factory, meta)` | A solver, by registry name |
| `add_importer(fmt, fn)` | A reader |
| `add_exporter(fmt, fn)` | A writer |
| `add_analysis(name, fn)` | A routine that returns a `Result` |
| `add_palette(palette)` | A colour map |
| `add_panel(spec)` | An interface panel |

## Writing a solver

```python
from materia.solvers.base import Capability, Solver, SolverResult
from materia.provenance import Fidelity, Origin, Provenance, Result

class MySolver(Solver):
    name = "my-plugin/my-model"
    fidelity = Fidelity.TIER1_CLASSICAL
    capabilities = {Capability.ENERGY, Capability.FORCES}
    description = "One sentence. Say what it cannot do."

    def supports(self, structure):
        from materia.solvers.base import SupportReport
        missing = ...
        if missing:
            return SupportReport(False, [f"No parameters for {missing}."])
        return SupportReport(True, [], ["Any caveat the user should see."])

    def run(self, structure, task="energy", **kwargs):
        out = SolverResult(solver=self.name, structure=structure)
        energy = ...
        out.results["energy"] = Result(
            "energy", energy, "eV",
            Provenance(
                model=self.name,
                fidelity=self.fidelity,
                origin=Origin.CALCULATED,
                approximations=["State every one."],
                tolerances={"...": ...},
                boundary_conditions=str(structure.cell.pbc),
                references=["Author, Journal Vol (Year) Page"],
            ))
        return out
```

The capability set is a promise. If you declare `Capability.LOCAL_DOS`, the
interface will offer it and users will rely on it. Declare only what you
compute, and refuse the rest through `self.require(capability)`, which returns
a populated unsupported `Result` naming other providers.

## Writing an analysis

```python
def coordination_histogram(structure, cutoff_A=3.0):
    from materia.physics.neighbors import neighbor_list
    from materia.provenance import Fidelity, Origin, Provenance, Result
    import numpy as np

    counts = neighbor_list(structure.positions, structure.cell, cutoff_A).counts()
    values, occurrences = np.unique(counts, return_counts=True)
    return Result(
        name="coordination_histogram",
        value={"coordination": values.tolist(), "atoms": occurrences.tolist()},
        unit="count",
        provenance=Provenance(
            model="my_plugin/coordination-histogram",
            fidelity=Fidelity.TIER0_STRUCTURAL,
            origin=Origin.CALCULATED,
            approximations=[f"Fixed {cutoff_A} A neighbour cutoff."],
            parameters={"cutoff_A": cutoff_A},
        ))
```

## Rules

1. **Return `Result`s with full provenance.** A plug-in that returns a bare
   float defeats the purpose of the program.
2. **Declare capabilities honestly.**
3. **Refuse rather than approximate.** Use `unsupported(...)` with a reason and
   alternatives.
4. **Cite your sources.** Material properties need `source`; models need
   `references`.
5. **Mark unimplemented reconstructions** `"implemented": false`. The flag is
   resolved against the generator registry in any case, so a material cannot
   make the interface offer something no generator can build. A plug-in may
   register its own generator; see
   [RECONSTRUCTIONS.md](RECONSTRUCTIONS.md) for the interface.
6. **Give reference geometry only where it applies.** A
   `reconstructions[].reference_geometry` entry needs a value and a citation,
   and it must be a measurement *of the system in the file*. The shipped
   strained-silicon plug-in material declares the Si(100) dimer generator and
   deliberately carries no reference geometry, because the LEED determination
   for unstrained Si(001) is not a measurement of a strained surface. Materia
   then reports the computed geometry with no comparison, which is the correct
   answer.
7. **Do not vendor GPL code.** Drive external packages through their public
   interfaces or their files, and record the licence.

## Security

Loading a plug-in **executes its code** with the privileges of the application.
Materia only searches directories you configured, and shows what it loaded and
what each plug-in contributed in the solver panel. It does not sandbox
plug-ins. Install plug-ins the way you install any other software.

## Reference implementation

`plugins/example_plugin/` exercises every extension point: a material path, a
solver, an analysis, an exporter, a palette and a panel. Copy it to
`~/.materia/plugins/` and edit.

Note how its material declares itself hypothetical: a deliberate 1 % scaling of
silicon, with `"note": "Copied from unstrained silicon; strain-induced
band-gap changes are NOT applied."` on the band gap. A plug-in that ships
fabricated data without saying so is a bug.
