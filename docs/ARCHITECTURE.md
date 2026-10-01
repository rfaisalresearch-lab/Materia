# Architecture

## Mission, in one paragraph

Let a user start at a macroscopic wafer, descend through scale until individual
atoms are resolved, image them with a virtual scanning probe, inspect and edit
them, and drive all of it from real Python, with every number traceable to the
model that produced it, and with honest refusal wherever the model cannot
answer.

## The stack, and why

| Layer | Choice | Reason |
| --- | --- | --- |
| Window | Native OS web view via pywebview (WKWebView / WebView2 / WebKitGTK) | A real application window with a native menu bar, no browser, no bundled runtime. Same architecture as Tauri and Electron, without needing a Rust or Node toolchain to build. |
| Interface | ES modules, WebGL2, Canvas2D, no build step | The source you read is the source that runs. No transpiler, no lockfile drift, no `node_modules`. WebGL2 is available everywhere the native web views are. |
| Core | Python with NumPy and SciPy | The scientific ecosystem this field already uses. Every external solver worth adapting has a Python interface. |
| Transport | JSON over a loopback HTTP endpoint | One boundary, easy to test, and the same boundary the headless `materia serve` mode uses. |
| Project file | ZIP with a versioned manifest | Inspectable with standard tools. A user can read the provenance report without the application. |

### Why not Tauri

The original specification preferred Tauri. Tauri requires the Rust toolchain
to build, which is a heavy prerequisite for a scientific tool whose users
already have Python. The architecture here is deliberately Tauri-shaped, a
native shell around a web view plus a separate computational core, so moving
to Tauri is a shell swap, not a rewrite: point a Tauri window at the same
loopback endpoint and the interface is unchanged. `docs/LIMITATIONS.md` records
that the Tauri shell is not built or tested here.

### Why WebGL2 and not WebGPU

WebGPU is not yet uniformly available in the system web views that host this
application. The renderer uses instanced sphere impostors, which WebGL2
supports fully and which carry a hundred thousand atoms interactively. The
renderer is isolated in `atoms-view.js`; a WebGPU backend would replace that
one module.

## Modules

```
materia/
  core_model/        cells, structures, atoms, selections, units
  elements/          periodic table, isotopes, electron configurations
  materials/         definition schema, validation, extensible registry
  potentials/        shipped parameter files with checksums and licences (data)
  structure_builder/ bulk crystals, primitive reduction, surfaces, defects
  multiscale/        procedural wafer, region-of-interest extraction
  physics/           neighbour lists, bond perception, potentials, EAM, strain, electrostatics
  solvers/           registry, capabilities, classical, electrostatics, tight binding, external
  experiments/       versioned experiment specifications run through external solvers (DFT ground state)
  microscopy/        tips, instrumental noise, STM, AFM, scan records
  visualization/     colour maps, PNG and TIFF writers
  provenance/        provenance, fidelity, origin, result records, claim classifications
  project_format/    project state, history, checkpoints, chunked arrays, migrations
  python_api/        the public API and the script runner
  plugin_system/     discovery, loading, registration context
  dataio/            structure and data import and export
  desktop_ui/        service layer, job queue, HTTP server, native shell, interface
  benchmarks.py      measured benchmark scenes
  cli.py             command line
```

Dependencies run downward only. `core_model` imports nothing from the project
except `elements` and `provenance`. `desktop_ui` imports everything and is
imported by nothing.

## Data flow

```
      wafer specification (parameters + seed)
                  │
                  ▼  procedural sampling at a coordinate
        local microstructure description
                  │
                  ▼  region extraction, seeded
              Structure  ──────────────► solvers ──► Result + Provenance
             (ids, arrays)                              │
                  │                                     ▼
                  ├──────────────► microscopy ────► ScanResult (channels,
                  │                                  noise record, provenance)
                  └──────────────► renderer payload ──► viewport
```

`Structure` is a structure-of-arrays container. Numeric data lives in
contiguous NumPy arrays; every atom also carries a stable integer id that
survives insertion, deletion and reordering. Selections, undo records, bonds
and provenance all reference ids, never array indices.

## The multiscale representation

A 300 mm silicon wafer holds of order 10²⁴ atoms. Materia never instantiates
it. The wafer is a parameter set plus a seed; roughness, grains, terraces,
doping and defect densities are deterministic functions of `(seed, position)`.
Asking for a region runs those functions at that coordinate, builds a slab from
the material's crystallography, and places dopants and defects by a Poisson
process seeded from the same pair. The same coordinate always regenerates the
same region, which is what lets a project file store a wafer and a coordinate
instead of atoms.

Scale levels, with the span at which each becomes the useful description:

| Level | Span | What it shows |
| --- | --- | --- |
| wafer | ≥ 10 mm | Diameter, flat or notch, layer stack, device regions, orientation |
| microstructure | ≥ 10 µm | Grains, boundaries, roughness, patterned regions |
| nanoscale | ≥ 100 nm | Steps, terraces, islands, voids, local strain |
| lattice | ≥ 1.5 nm | Unit cells, lattice sites, dopants, vacancies |
| atomic | < 1.5 nm | Individual atoms, bonds, orbitals, probe signals |

## The provenance contract

Every quantity that crosses a module boundary is a `Result`:

```python
Result(
    name="band_gap",
    value=1.17,
    unit="eV",
    provenance=Provenance(
        model="tight-binding/sp3s*-Si",
        fidelity=Fidelity.TIER2_SEMI_EMPIRICAL,
        origin=Origin.CALCULATED,
        approximations=[...],
        tolerances={...},
        boundary_conditions="3D periodic",
        references=[...],
        seed=None,
        software_version="0.1.0",
    ),
    convergence=Convergence(converged=True, ...),
)
```

`value` is `None` exactly when `origin` is `UNSUPPORTED`, and then
`unsupported_reason` and `suggested_models` are populated. There is no state in
which a number appears without a model attached to it.

## The service layer

`materia.desktop_ui.service.Service` is the only thing the interface talks to.
Every graphical action has a method there, and the same methods are what the
integration tests drive. This is what keeps simulation out of the views: a view
can render a `Result`, but it cannot compute one.

Long operations go through `materia.desktop_ui.jobs.JobQueue`: a worker thread
with progress, a message, a cancellation flag and a retained result.

## Performance strategy

* **Region-of-interest generation.** Atoms exist only where the user looks.
* **Level of detail.** The renderer payload thins above a configurable atom
  budget and says so in the viewport.
* **Spatial indexing.** Neighbour lists use a k-d tree over explicitly expanded
  periodic images, which is correct for cells thinner than twice the cutoff.
* **Truncated electronic slab.** The scanning-probe electronic structure is
  computed on the topmost few nanometres, because deeper atoms cannot reach the
  vacuum. Recorded as an approximation.
* **Windowed eigensolves.** Only states inside the bias window are extracted.
* **Deterministic caching.** The electronic structure is memoised on a content
  digest of the structure and the settings, so re-imaging at a different
  resolution, palette or noise seed is immediate and any edit invalidates it.
* **Cancellable background jobs** with incremental progress.

Measured numbers are in [PERFORMANCE.md](PERFORMANCE.md). None are estimated.

## Testing strategy

| Suite | Question it answers |
| --- | --- |
| `tests/unit` | Does the code do what the model says? |
| `tests/integration` | Do the acceptance scenarios work end to end, through the same service the interface uses? |
| `tests/validation` | Does the model reproduce an analytic result, a published value or a conservation law? |
| `tests/performance` | Does the scaling and the level-of-detail behaviour hold? |

The validation suite is marked separately because a passing software test is
not scientific validation, and conflating them would be the central dishonesty
this project exists to avoid.
