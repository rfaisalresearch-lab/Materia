"""Virtual probe tip models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

TIP_WORK_FUNCTIONS: Dict[str, float] = {
    "W": 4.55,
    "PtIr": 5.5,
    "Pt": 5.65,
    "Ir": 5.27,
    "Au": 5.1,
    "Ag": 4.26,
    "Ni": 5.15,
    "CO": 5.0,
    "Si": 4.85,
}

APEX_STATES = ("s", "pz", "dz2")


@dataclass
class Tip:
    """A virtual scanning-probe tip.

    Attributes
    ----------
    material
        Apex material; sets the work function unless one is given explicitly.
    apex_state
        Electronic character of the apex orbital used by the Tersoff-Hamann
        model.  ``"s"`` is the only case the original theory covers.
    radius_A
        Macroscopic apex radius of curvature, used by the van der Waals
        (Hamaker sphere-plane) term in AFM, not by STM.
    functionalisation
        Optional adsorbate on the apex (e.g. ``"CO"`` for a CO-functionalised
        AFM tip).  Recorded in provenance; it changes ``stiffness_N_m`` only.
    """

    material: str = "W"
    apex_state: str = "s"
    radius_A: float = 50.0
    work_function_eV: Optional[float] = None
    functionalisation: str = ""
    lateral_stiffness_N_m: Optional[float] = None
    apex_atoms: int = 1
    hamaker_eV: float = 0.624

    def __post_init__(self) -> None:
        if self.apex_state not in APEX_STATES:
            raise ValueError(
                f"Unknown apex state {self.apex_state!r}; implemented: {APEX_STATES}"
            )
        if self.work_function_eV is None:
            if self.material not in TIP_WORK_FUNCTIONS:
                raise ValueError(
                    f"No tabulated work function for tip material {self.material!r}. "
                    f"Known: {sorted(TIP_WORK_FUNCTIONS)}. Pass work_function_eV explicitly."
                )
            self.work_function_eV = TIP_WORK_FUNCTIONS[self.material]

    def describe(self) -> dict:
        return {
            "material": self.material,
            "apex_state": self.apex_state,
            "radius_A": self.radius_A,
            "work_function_eV": self.work_function_eV,
            "functionalisation": self.functionalisation,
            "apex_atoms": self.apex_atoms,
            "hamaker_eV": self.hamaker_eV,
            "caveats": [
                "The tip is modelled as a structureless apex. Real tips have an "
                "unknown atomic structure that dominates the apparent corrugation.",
                ("Tersoff-Hamann assumes an s-wave apex; apex_state='" + self.apex_state
                 + "' is " + ("the case the theory covers."
                              if self.apex_state == "s"
                              else "outside the original derivation and is applied as a "
                                   "directional weighting only.")),
            ],
        }
