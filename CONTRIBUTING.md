# Contributing to Materia

## The standard this project holds itself to

Materia is a scientific instrument. A contribution is judged on whether a user
can trust what the program tells them.

1. **No unlabelled numbers.** Anything a user can see carries a
   `Provenance` record: the model, its approximations, its tolerances, its
   boundary conditions, its convergence state, and whether the value is
   calculated, interpolated, estimated, illustrative, reference, imported or
   measured.
2. **Refuse rather than approximate silently.** If a model cannot answer a
   question, return `unsupported(...)` with the reason and the solvers that
   could, and preserve the request. Never substitute a different model without
   saying so.
3. **Cite reference data.** Every physical constant in a material definition
   needs a `source` naming the primary literature. "From memory" is not a
   source.
4. **Separate implementation tests from physical validation.** A unit test
   shows the code does what the model says. A validation case shows the model
   reproduces an analytic result, a published value or a conservation law. Mark
   validation cases with `@pytest.mark.validation`.
5. **Keep simulation out of the interface.** Views call
   `materia.desktop_ui.service.Service`; the service calls the core. No physics
   in a view, and no rendering in the core.

## Getting set up

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[all]"
pip install pywebview
pytest
materia
```

## Adding a material

Write a JSON file following [docs/MATERIAL_SCHEMA.md](docs/MATERIAL_SCHEMA.md)
and drop it in `materia/materials/library/`. Then:

* add its id to `EXPECTED` in `tests/unit/test_materials.py`;
* add a validation case in `tests/validation/` that checks a geometric
  quantity you can cite, a bond length, a coordination number, the density;
* mark every reconstruction you have *not* implemented with
  `"implemented": false` and a reference.

A definition whose generated cell does not reproduce its own cited bond lengths
is a bug, and the validation suite is there to catch it.

## Adding a solver

Subclass `materia.solvers.base.Solver`, declare `capabilities` honestly, and
register it. The capability set is a promise: if you declare
`Capability.LOCAL_DOS`, the interface will offer it and users will rely on it.
Implement `supports()` so that unusable inputs are refused with a reason that
names the offending species, charge or periodicity.

## Adding an external adapter

Add an `AdapterSpec` to `materia/solvers/external.py`. Do not vendor the
package, do not link against it, and record its licence in `license_note`.
The adapter must register even when the package is absent and must then report
itself unavailable with an installation hint.

## Style

* Python: 4 spaces, type hints on public functions, NumPy-style docstrings that
  state the approximations. Keep lines under 96 characters.
* JavaScript: ES modules, no build step, JSDoc types on exported functions.
* Comments explain *why*, never *what*. The code says what it does.
* Units in identifiers (`length_A`, `energy_eV`, `bias_V`) wherever ambiguity
  is possible.

## Pull requests

Include: what changed, which tests cover it, any new approximation and where it
is recorded, and, for anything touching physics, the validation case that
demonstrates it is right. Run `pytest` before opening the request.
