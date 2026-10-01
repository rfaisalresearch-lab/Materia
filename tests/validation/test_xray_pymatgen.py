"""Powder patterns against pymatgen's XRDCalculator, an independent implementation.

pymatgen (MIT) is run in whichever interpreter has it: this one, or a conda
environment such as ``materia-ml``.  Skipped with BLOCKED when none does.
Anomalous dispersion is switched off on the Materia side because pymatgen
omits it; the two codes use different form-factor parametrisations, which
limits agreement to about 1.5% of each line.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from materia.core_model.cell import Cell
from materia.core_model.structure import Structure
from materia.elements import periodic_table as pt
from materia.physics import xray as X

pytestmark = pytest.mark.validation
pytest.importorskip("xraydb", reason="BLOCKED: xraydb is not installed")
FCC = np.array([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])
SCRIPT = """
import json, sys
from pymatgen.core import Lattice, Structure
from pymatgen.analysis.diffraction.xrd import XRDCalculator
spec = json.loads(sys.stdin.read())
s = Structure(Lattice(spec["cell"]), spec["species"], spec["frac"])
p = XRDCalculator(wavelength=spec["wavelength"]).get_pattern(s, scaled=True,
                                                             two_theta_range=(20, 100))
print(json.dumps({"two_theta": [float(v) for v in p.x], "intensity": [float(v) for v in p.y]}))
"""


def interpreter():
    candidates = [Path(sys.executable)]
    for root in ("miniforge3", "miniconda3", "anaconda3"):
        envs = Path.home() / root / "envs"
        if envs.is_dir():
            candidates += sorted(envs.glob("*/bin/python"))
    for python in candidates:
        probe = subprocess.run([str(python), "-c", "import pymatgen.analysis.diffraction.xrd"],
                               capture_output=True)
        if probe.returncode == 0:
            return str(python)
    pytest.skip("BLOCKED: no interpreter with pymatgen")


def crystals():
    a = 5.431
    yield "Si", Structure([14] * 8, np.vstack([FCC, FCC + .25]) * a,
                          Cell(np.eye(3) * a, (True,) * 3))
    a = 5.6402
    yield "NaCl", Structure([11] * 4 + [17] * 4, np.vstack([FCC, FCC + [.5, 0, 0]]) * a,
                            Cell(np.eye(3) * a, (True,) * 3))
    a, c, u = 3.2495, 5.2069, 0.3819
    cell = np.array([[a, 0, 0], [-a / 2, a * np.sqrt(3) / 2, 0], [0, 0, c]])
    frac = np.array([[1 / 3, 2 / 3, 0], [2 / 3, 1 / 3, .5], [1 / 3, 2 / 3, u],
                     [2 / 3, 1 / 3, .5 + u]])
    yield "ZnO", Structure([30, 30, 8, 8], frac @ cell, Cell(cell, (True,) * 3))


@pytest.mark.parametrize("name,crystal", list(crystals()))
def test_powder_pattern_matches_pymatgen(name, crystal):
    python = interpreter()
    cell = np.asarray(crystal.cell.matrix)
    spec = {"cell": cell.tolist(), "species": [pt.symbol(int(z)) for z in crystal.numbers],
            "frac": (np.asarray(crystal.positions) @ np.linalg.inv(cell)).tolist(),
            "wavelength": 1.540593}
    done = subprocess.run([python, "-c", SCRIPT], input=json.dumps(spec), capture_output=True,
                          text=True, timeout=300, check=True)
    reference = json.loads(done.stdout)
    mine = X.powder_pattern(crystal, 1.540593, (20, 100), anomalous=False).value
    assert len(mine) == len(reference["two_theta"])
    for peak, angle, intensity in zip(mine, reference["two_theta"], reference["intensity"]):
        assert peak["two_theta_deg"] == pytest.approx(angle, abs=1e-6)
        assert peak["relative_intensity"] == pytest.approx(intensity, abs=0.15, rel=0.02)
