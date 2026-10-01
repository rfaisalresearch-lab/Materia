## What this changes

## Why

## How it was verified

- [ ] `pytest` passes
- [ ] New behaviour has a test
- [ ] Anything touching physics has a **validation** case comparing against an
      analytic result, a published value or a conservation law

## Provenance

- [ ] Every new result carries a `Provenance` record
- [ ] Every new approximation is listed in `approximations`
- [ ] Anything the model cannot do returns `unsupported(...)` with a reason and
      alternatives, rather than an approximation
- [ ] New reference data cites primary literature
- [ ] New material definitions mark unimplemented reconstructions as such

## Documentation

- [ ] `docs/SCIENTIFIC_MODELS.md` updated for a new model
- [ ] `docs/LIMITATIONS.md` updated for a new limitation
- [ ] `CHANGELOG.md` updated
