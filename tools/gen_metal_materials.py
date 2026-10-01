#!/usr/bin/env python3
"""Generate material-definition JSON for the elemental metals.

The elemental metals differ only in a handful of numbers, so their definition
files are generated from the table below to keep them consistent.  The
generated files are checked into the repository and are the authoritative
data; re-run this script only when the table changes.

Usage:  python tools/gen_metal_materials.py
"""

import json
import os

OUT = os.path.join(os.path.dirname(__file__), "..", "materia", "materials", "library")

FCC = [("Ax", [0.0, 0.0, 0.0]), ("Ax", [0.0, 0.5, 0.5]),
       ("Ax", [0.5, 0.0, 0.5]), ("Ax", [0.5, 0.5, 0.0])]
BCC = [("Ax", [0.0, 0.0, 0.0]), ("Ax", [0.5, 0.5, 0.5])]
HCP = [("Ax", [1 / 3, 2 / 3, 0.25]), ("Ax", [2 / 3, 1 / 3, 0.75])]

METALS = [
    ("copper", "Copper", "Cu", "fcc", 3.6149, None, 8.96, (4.65, "(111)"), 1357.77, 401,
     (168.3, 122.1, 75.7), 3.49, 0.0),
    ("silver", "Silver", "Ag", "fcc", 4.0853, None, 10.49, (4.74, "(111)"), 1234.93, 429,
     (124.0, 93.4, 46.1), 2.95, 0.0),
    ("platinum", "Platinum", "Pt", "fcc", 3.9242, None, 21.45, (5.65, "(111)"), 2041.4, 71.6,
     (346.7, 250.7, 76.5), 5.84, 0.0),
    ("nickel", "Nickel", "Ni", "fcc", 3.5240, None, 8.908, (5.15, "(111)"), 1728, 90.9,
     (246.5, 147.3, 124.7), 4.44, 0.606),
    ("tungsten", "Tungsten", "W", "bcc", 3.1652, None, 19.25, (4.55, "polycrystalline"), 3695, 173,
     (522.4, 204.4, 160.6), 8.90, 0.0),
    ("titanium", "Titanium", "Ti", "hcp", 2.9508, 4.6855, 4.506, (4.33, "polycrystalline"), 1941, 21.9,
     (162.4, 92.0, 46.7), 4.85, 0.0),
]

SOURCES = {
    "lattice": "CRC Handbook of Chemistry and Physics, 104th ed., crystallographic data table",
    "wf": "H. B. Michaelson, J. Appl. Phys. 48 (1977) 4729",
    "elastic": "G. Simmons and H. Wang, Single Crystal Elastic Constants and Calculated Aggregate Properties, 2nd ed. (MIT Press, 1971)",
    "cohesive": "C. Kittel, Introduction to Solid State Physics, 8th ed., Table 3.1",
    "crc": "CRC Handbook of Chemistry and Physics, 104th ed.",
}

PROTO = {
    "fcc": (FCC, "cubic", 225, "Fm-3m", [[1, 1, 1], [1, 0, 0], [1, 1, 0]], "4a"),
    "bcc": (BCC, "cubic", 229, "Im-3m", [[1, 1, 0], [1, 0, 0], [1, 1, 1]], "2a"),
    "hcp": (HCP, "hexagonal", 194, "P6_3/mmc", [[0, 0, 1], [1, 1, 0], [1, 0, 0]], "2c"),
}


def build(rec):
    (mid, name, sym, proto, a, c, dens, (wf, face), tm, kth, (c11, c12, c44), ecoh, mag) = rec
    basis_template, system, sgnum, sgsym, orients, wyck = PROTO[proto]
    basis = [
        {"element": sym, "fractional": [round(x, 7) for x in frac], "label": sym, "wyckoff": wyck}
        for _, frac in basis_template
    ]
    lattice = {"a": a, "unit": "A", "system": system, "temperature_K": 293,
               "source": SOURCES["lattice"]}
    if c is not None:
        lattice["c"] = c
    props = {
        "density": {"value": dens, "unit": "g/cm^3", "conditions": "293 K", "source": SOURCES["crc"]},
        "band_gap": {"value": 0.0, "unit": "eV", "conditions": "metal", "source": "definition"},
        "band_gap_type": {"value": "metal", "unit": ""},
        "work_function": {"value": wf, "unit": "eV", "conditions": face, "source": SOURCES["wf"]},
        "melting_point": {"value": tm, "unit": "K", "source": SOURCES["crc"]},
        "thermal_conductivity": {"value": kth, "unit": "W/(m K)", "conditions": "300 K",
                                 "source": SOURCES["crc"]},
        "elastic_C11": {"value": c11, "unit": "GPa", "conditions": "300 K", "source": SOURCES["elastic"]},
        "elastic_C12": {"value": c12, "unit": "GPa", "conditions": "300 K", "source": SOURCES["elastic"]},
        "elastic_C44": {"value": c44, "unit": "GPa", "conditions": "300 K", "source": SOURCES["elastic"]},
        "cohesive_energy": {"value": ecoh, "unit": "eV/atom", "source": SOURCES["cohesive"]},
    }
    if mag:
        props["magnetic_moment"] = {
            "value": mag, "unit": "mu_B/atom", "conditions": "ferromagnetic ground state, 0 K",
            "source": "C. Kittel, Introduction to Solid State Physics, 8th ed.",
        }
    doc = {
        "schema_version": "1.0",
        "id": mid,
        "name": name,
        "formula": sym,
        "aliases": [sym],
        "category": "elemental metal",
        "structure": {
            "prototype": proto,
            "space_group": {"number": sgnum, "symbol": sgsym},
            "lattice": lattice,
            "basis": basis,
        },
        "orientations": orients,
        "terminations": [{
            "id": "ideal-bulk", "orientation": orients[0],
            "description": f"Ideal bulk truncation of {sym}{tuple(orients[0])}.",
            "implemented": True,
        }],
        "reconstructions": [],
        "properties": props,
        "recommended_models": {
            "relax": "lennard-jones-fitted",
            "electronic": "unsupported",
            "stm": "tersoff-hamann-jellium",
            "high-fidelity": "external:lammps-eam",
        },
        "references": [SOURCES["wf"], SOURCES["elastic"], SOURCES["cohesive"]],
        "license": "CC0-1.0",
        "provenance_note": (
            "Generated by tools/gen_metal_materials.py from the cited primary sources. "
            "Metallic bonding is not pairwise; the Lennard-Jones fit offered as the default "
            "relaxation model is flagged as an interactive approximation only."
        ),
    }
    return doc


def main():
    for rec in METALS:
        doc = build(rec)
        path = os.path.join(OUT, f"{doc['id']}.json")
        with open(path, "w") as fh:
            json.dump(doc, fh, indent=2)
            fh.write("\n")
        print("wrote", os.path.relpath(path))


if __name__ == "__main__":
    main()
