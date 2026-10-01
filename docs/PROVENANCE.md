# Provenance

Every number Materia shows is attached to the model that produced it. This
document defines the vocabulary and the guarantees.

## The record

```python
@dataclass
class Provenance:
    model: str                    # "tight-binding/sp3s*-Si"
    fidelity: Fidelity            # which tier
    origin: Origin                # how it should be read
    approximations: list[str]     # what the model assumes
    tolerances: dict              # numerical settings that affect the answer
    boundary_conditions: str      # periodicity, vacuum, fixed atoms
    parameters: dict              # everything the model was given
    references: list[str]         # primary literature
    dataset: str                  # which material file
    dataset_license: str
    seed: int | None              # for anything stochastic
    software_version: str
    schema_version: str
    platform: str
    created_unix: float
    inputs_digest: str
    notes: str
```

A `Result` carries a value, a unit, a `Provenance`, an optional uncertainty
with its kind, an optional confidence, and an optional `Convergence`.

## Origins

| Origin | Meaning |
| --- | --- |
| `calculated` | Produced by a solver in this session from the current state |
| `interpolated` | Interpolated or extrapolated from calculated samples |
| `estimated` | From a heuristic or empirical rule, not a variational solution |
| `illustrative` | Teaching visualisation. Not a physical prediction |
| `reference` | Literature value shipped with the program |
| `imported` | Read from a file or an external solver |
| `measured` | Experimental data supplied by the user |
| `unsupported` | The active model cannot produce this. `value` is `None` |

`value is None` **if and only if** `origin is UNSUPPORTED`. There is no state in
which a number appears without a model attached.

## Fidelity

`tier0-structural`, `tier1-classical`, `tier2-semi-empirical`,
`tier3-external-first-principles`, `non-physical`. See
[FIDELITY_TIERS.md](FIDELITY_TIERS.md).

`non-physical` is used for models that exist to test the machinery, such as the
harmonic Einstein solid. It is never used for anything a user would mistake for
a prediction.

## Convergence

```python
Convergence(converged, iterations, residual, residual_metric, tolerance, message)
```

`residual_metric` is a sentence, not a symbol: "max |F| on free atoms",
"max |I/I_setpoint − 1|". A non-converged result is still returned, with
`converged=False` and a message saying what was not reached: it is not
silently discarded and not silently accepted. A ground-state DFT calculation
that does not reach self-consistency is stricter still: it keeps no value at
all, only its record with origin `unsupported`, because a half-solved
Kohn-Sham problem is not an energy.

## Refusal

```python
unsupported(
    name="self_consistent_solution",
    model="tight-binding/sp3s*-Si",
    reason="This tight-binding model is non-self-consistent by construction: "
           "the on-site energies are fixed empirical parameters, so there is no "
           "charge density to iterate. ...",
    suggested_models=["external:gpaw", "external:cp2k", "external:quantum-espresso"],
)
```

Three rules:

1. **Never substitute.** If the requested model cannot answer, do not quietly
   use a different one.
2. **Always name an alternative** when one exists, by registry name.
3. **Preserve the request.** External-solver refusals keep the task and its
   keyword arguments in `extra["preserved_request"]` so the same calculation
   can be re-run unchanged once the solver is installed.

## Classifications

A result says how it was produced; a **classification** says what kind of
statement it supports. `materia/provenance/classification.py` defines seven,
each with an evidence rule checked when a claim is stored and again when it is
read from a project:

| Classification | Minimum evidence |
| --- | --- |
| `observation` | one recorded value, computed or measured |
| `computational-prediction` | one converged calculated result |
| `candidate-relation` | converged results at two or more conditions |
| `empirical-invariant` | converged results at three or more conditions, and it holds only across them |
| `conjecture` | a statement; evidence optional |
| `independently-reproduced` | a result and a reproduction by a different model or different software |
| `experimentally-supported` | a result and a measurement or cited experiment |

There is no theorem classification, and a statement that speaks of a proof, a
theorem or a law is refused: a theorem needs a formal proof, and a
computation does not provide one however often it is repeated. Ground-state
DFT results are classified as `computational-prediction`. Numerical
convergence evidence from the convergence laboratory is recorded beside a
result; it is never turned into an uncertainty on the physical value.

## Seeds and reproducibility

Anything stochastic records its seed: instrumental noise, molecular dynamics,
procedural wafer microstructure, defect placement. Re-running with the same
seed reproduces the result bit for bit, and the integration tests assert this.

The procedural wafer is deterministic in `(seed, position)`, which is why a
project file can store a wafer specification and a coordinate instead of atoms
and still regenerate the identical region.

## In the interface

* Every inspector field shows an origin tag next to its section heading.
* The scan overlay names the model, the seed, the acquisition time and the
  feedback residual.
* The solver panel shows each model's capabilities *and* the capabilities it
  lacks.
* A refusal opens a dialog with the reason, the suggested solvers, and a note
  that nothing has been fabricated.
* **File ▸ Show provenance report** prints the whole project: wafer parameters
  and seed, every region, every result with its model and approximations, and
  the full operation history.

## In the project file

`PROVENANCE.txt` inside the `.materia` archive is plain text, readable with
`unzip -p project.materia PROVENANCE.txt` and no application at all. Each
result is stored in `results/<key>.json` with its complete record.
