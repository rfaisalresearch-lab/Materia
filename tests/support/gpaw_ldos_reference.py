"""An independent LDOS reference, run under GPAW's own interpreter.

Given the job Materia would send, this script repeats the calculation with
GPAW directly and evaluates the LDOS with ASE's ``ase.dft.stm.STM`` class,
code that shares nothing with Materia's worker.  It also writes GPAW's pseudo
valence density of the self-consistent ground state and, for a slab, ASE's
constant-current heights and the linearly interpolated constant-height values.

usage: gpaw_ldos_reference.py <job.json> <out.npz> [isovalue z0 height]
"""

import json
import sys

import numpy as np
from ase import Atoms
from ase.dft.stm import STM
from gpaw import GPAW


def main(argv):
    job = json.load(open(argv[1]))
    structure = job["structure"]
    atoms = Atoms(numbers=structure["numbers"], positions=structure["positions"],
                  cell=structure["cell"], pbc=structure["pbc"])
    atoms.set_initial_magnetic_moments(structure["magnetic_moments"])
    calc = GPAW(txt=None, **job["parameters"])
    atoms.calc = calc
    atoms.get_potential_energy()
    density = calc.get_pseudo_density()
    nscf = calc.fixed_density(txt=None, **job["ldos"]["nscf"])
    target = nscf.atoms
    target.calc = nscf
    low, high = job["ldos"]["energy_min"], job["ldos"]["energy_max"]
    stm = STM(target)
    if low < 0.0 <= high and high == 0.0:
        stm.calculate_ldos(low)
        ldos = stm.ldos
    elif low >= 0.0:
        stm.calculate_ldos(high)
        ldos = stm.ldos
    else:
        raise SystemExit("the reference handles windows ending or starting at the Fermi level")
    out = {"ase_ldos": np.asarray(ldos), "pseudo_density": np.asarray(density),
           "n_spins": np.array(nscf.get_number_of_spins())}
    if len(argv) > 3:
        isovalue, z0, height = (float(v) for v in argv[3:6])
        out["ase_heights"] = stm.scan(stm.bias, isovalue, z0=z0)[2]
        nz = ldos.shape[2]
        index = height / atoms.cell[2, 2] * nz
        lower = int(np.floor(index))
        fraction = index - lower
        out["interpolated"] = (1 - fraction) * ldos[:, :, lower] + fraction * ldos[:, :,
                                                                                   lower + 1]
    np.savez(argv[2], **out)


if __name__ == "__main__":
    main(sys.argv)
