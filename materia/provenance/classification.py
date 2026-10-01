"""What kind of statement a result supports, stated before anyone generalises it.

A number computed by a solver and a law of nature are different things, and
the vocabulary has to keep them apart.  Every claim Materia records carries
one of these classifications, each with an evidence rule that is checked
when the claim is stored:

========================== ==================================================
classification             minimum evidence
========================== ==================================================
observation                one recorded value, computed or measured
computational-prediction   one converged calculated result
candidate-relation         calculated results at two or more conditions
empirical-invariant        calculated results at three or more conditions
conjecture                 a statement; evidence optional
independently-reproduced   a result, and a reproduction by a different model
                           or different software
experimentally-supported   a result, and a measurement or cited experiment
========================== ==================================================

There is no "theorem" classification, and a statement that says it proves or
has discovered a theorem or law is refused.  A theorem requires a formal
proof; a computation, however many times it is repeated, does not provide
one.  Numerical convergence evidence says that a number is converged with
respect to its discretisation, not that the model producing it is right.

This module only defines the structures an equation-discovery engine will
later need.  No such engine exists in Materia.
"""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class Classification(str, Enum):
    OBSERVATION = "observation"
    COMPUTATIONAL_PREDICTION = "computational-prediction"
    CANDIDATE_RELATION = "candidate-relation"
    EMPIRICAL_INVARIANT = "empirical-invariant"
    CONJECTURE = "conjecture"
    INDEPENDENTLY_REPRODUCED = "independently-reproduced"
    EXPERIMENTALLY_SUPPORTED = "experimentally-supported"


DEFINITIONS: Dict[str, str] = {
    "observation": "A recorded value, computed or measured, with no claim beyond it.",
    "computational-prediction": "What a stated model predicts for a stated system. "
                                "Carries the model's systematic error, which is not "
                                "estimated by numerical convergence.",
    "candidate-relation": "A pattern seen across several computed conditions, "
                          "proposed for testing. Not established.",
    "empirical-invariant": "A quantity observed to stay fixed across the conditions "
                           "tested, within stated tolerances, and only there.",
    "conjecture": "A proposed general statement, not yet supported by the evidence "
                  "required for any stronger class.",
    "independently-reproduced": "Reproduced by a different model or different "
                                "software from the original result.",
    "experimentally-supported": "Consistent with a measurement or a cited experiment "
                                "within stated uncertainties.",
}

EVIDENCE_KINDS = ("result", "study", "reproduction", "measurement", "literature")

_PROOF_WORDS = re.compile(
    r"\b(theorem|theorems|proof|proofs|proven|proved|proves|law of nature|"
    r"laws of nature|universal law)\b", re.IGNORECASE)

NO_THEOREM = ("There is no theorem classification. A theorem requires a formal "
              "proof, and a computation does not provide one however often it is "
              "repeated. Classify the statement as a conjecture or a candidate "
              "relation and cite its evidence.")


class ClassificationError(ValueError):
    """A claim whose classification is not supported by its evidence."""


def parse(value: Any) -> Classification:
    """A classification from its name, refusing anything that implies proof."""
    if isinstance(value, Classification):
        return value
    text = str(value).strip().lower().replace("_", "-").replace(" ", "-")
    if "theorem" in text or "proof" in text or "proven" in text or text == "law":
        raise ClassificationError(NO_THEOREM)
    try:
        return Classification(text)
    except ValueError:
        raise ClassificationError(
            f"Unknown classification {value!r}. Available: "
            f"{', '.join(c.value for c in Classification)}.") from None


@dataclass(frozen=True)
class Evidence:
    """One piece of evidence a claim cites.

    ``reference`` is a project result key, a study id, or a citation for a
    measurement or publication.  ``model`` and ``software`` identify what
    produced it, which is what makes a reproduction independent or not.
    """

    kind: str
    reference: str
    model: str = ""
    software: str = ""
    digest: str = ""
    converged: Optional[bool] = None
    note: str = ""

    def as_dict(self) -> dict:
        return {"kind": self.kind, "reference": self.reference, "model": self.model,
                "software": self.software, "digest": self.digest,
                "converged": self.converged, "note": self.note}

    @staticmethod
    def from_dict(data: dict) -> "Evidence":
        return Evidence(kind=str(data.get("kind", "")),
                        reference=str(data.get("reference", "")),
                        model=str(data.get("model", "")),
                        software=str(data.get("software", "")),
                        digest=str(data.get("digest", "")),
                        converged=data.get("converged"),
                        note=str(data.get("note", "")))


@dataclass
class Claim:
    """A statement, its classification, and the evidence behind it."""

    statement: str
    classification: Classification
    evidence: List[Evidence] = field(default_factory=list)
    conditions: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    claim_id: str = field(default_factory=lambda: f"claim-{uuid.uuid4().hex[:10]}")
    created_unix: float = field(default_factory=time.time)

    def validate(self) -> None:
        """Raise :class:`ClassificationError` unless the evidence supports the class."""
        self.classification = parse(self.classification)
        statement = str(self.statement or "").strip()
        if not statement:
            raise ClassificationError("A claim needs a statement.")
        if _PROOF_WORDS.search(statement):
            raise ClassificationError(
                "The statement speaks of proof, a theorem or a law. " + NO_THEOREM)
        for item in self.evidence:
            if item.kind not in EVIDENCE_KINDS:
                raise ClassificationError(
                    f"Unknown evidence kind {item.kind!r}. Available: "
                    f"{', '.join(EVIDENCE_KINDS)}.")
            if not item.reference:
                raise ClassificationError("Every piece of evidence needs a reference.")
        results = [e for e in self.evidence if e.kind == "result"]
        calculated = [e for e in results if e.converged is not False]
        kind = self.classification
        if kind is Classification.OBSERVATION and not self.evidence:
            raise ClassificationError("An observation must cite the value it records.")
        if kind is Classification.COMPUTATIONAL_PREDICTION and not calculated:
            raise ClassificationError(
                "A computational prediction must cite at least one converged "
                "calculated result.")
        if kind is Classification.CANDIDATE_RELATION and len(calculated) < 2:
            raise ClassificationError(
                "A candidate relation must cite converged results at two or more "
                "conditions.")
        if kind is Classification.EMPIRICAL_INVARIANT and len(calculated) < 3:
            raise ClassificationError(
                "An empirical invariant must cite converged results at three or more "
                "conditions, and holds only across the conditions cited.")
        if kind is Classification.INDEPENDENTLY_REPRODUCED:
            reproductions = [e for e in self.evidence if e.kind == "reproduction"]
            if not results or not reproductions:
                raise ClassificationError(
                    "An independently reproduced claim must cite a result and a "
                    "reproduction of it.")
            originals = {(e.model, e.software) for e in results}
            if all((r.model, r.software) in originals for r in reproductions):
                raise ClassificationError(
                    "A reproduction by the same model and the same software is a "
                    "repeat, not an independent reproduction.")
        if kind is Classification.EXPERIMENTALLY_SUPPORTED:
            if not any(e.kind in ("measurement", "literature") for e in self.evidence):
                raise ClassificationError(
                    "An experimentally supported claim must cite a measurement or a "
                    "published experiment.")

    def as_dict(self) -> dict:
        return {"claim_id": self.claim_id, "statement": self.statement,
                "classification": parse(self.classification).value,
                "definition": DEFINITIONS[parse(self.classification).value],
                "evidence": [e.as_dict() for e in self.evidence],
                "conditions": dict(self.conditions), "notes": self.notes,
                "created_unix": self.created_unix}

    @staticmethod
    def from_dict(data: dict) -> "Claim":
        claim = Claim(statement=str(data.get("statement", "")),
                      classification=parse(data.get("classification", "")),
                      evidence=[Evidence.from_dict(e) for e in data.get("evidence", [])],
                      conditions=dict(data.get("conditions", {}) or {}),
                      notes=str(data.get("notes", "")),
                      claim_id=str(data.get("claim_id") or f"claim-{uuid.uuid4().hex[:10]}"),
                      created_unix=float(data.get("created_unix", time.time())))
        claim.validate()
        return claim


def describe() -> List[dict]:
    """Every classification with its definition, for the interface."""
    return [{"name": c.value, "definition": DEFINITIONS[c.value]} for c in Classification]
