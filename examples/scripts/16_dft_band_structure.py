"""Kohn-Sham band structure of diamond silicon through GPAW.

Run with:
    materia run examples/scripts/16_dft_band_structure.py
or paste into the embedded Python console.

The band structure is a frozen, versioned specification checked before GPAW
starts: the ground state it belongs to, the symmetry used for that ground
state, and the exact reciprocal-space path. The standard path of the lattice is
generated once from ASE's tables and stored as explicit fractional k-points
with their labels; GPAW computes exactly those points with symmetry off, and
the k-points, band count, labels and path distance it reports are verified
before anything is stored. A band structure observes a structure and never
changes it. Band edges are read along the sampled path only.
"""

import numpy as np

from materia.core_model import Cell, Structure

if not dft.available():
    print("GPAW is not usable here:", dft.status()["blocking_reason"])
    print(dft.status()["install_hint"])
else:
    a = 5.43
    si = Structure(np.array([14, 14]), np.array([[0.0, 0.0, 0.0], [a / 4] * 3]),
                   Cell(0.5 * a * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]),
                        (True, True, True)))
    project.add_structure(si)
    spec = dft.bands_spec(si, xc="PBE", cutoff_eV=300.0, kpoints=[4, 4, 4],
                          kpoints_gamma_centered=True, occupations="fixed", smearing_eV=0.0,
                          path="GXWKGL", sampling_density_per_invA=8.0, n_bands=8)
    print(f"Si band structure {spec.short_digest}: path {spec.path_origin.get('path_chosen')} "
          f"on the {spec.path_origin.get('lattice')} lattice, {spec.n_kpoints} k-points, "
          f"refusals: {dft.bands_check(spec)['blocking'] or 'none'}")
    run = dft.bands(spec=spec)
    summary = run["bands"].value
    edges = summary["band_edges"]
    print(f"Si band structure: Fermi level {summary['fermi_level_eV']:.4f} eV, "
          f"path length {summary['path_length_invA']:.4f} 1/A")
    if edges.get("gap_eV") is not None:
        print(f"  {edges['gap_kind']} gap along the path {edges['gap_eV']:.4f} eV: valence top "
              f"at {edges['vbm']['label'] or 'k-point ' + str(edges['vbm']['index'])}, "
              f"conduction bottom at k-point {edges['cbm']['index']} "
              f"{np.round(edges['cbm']['kpoint_frac'], 4).tolist()}")
    eigen = dft.bands_array(name="eigenvalues").data[0] - summary["fermi_level_eV"]
    for tick in summary["ticks"]:
        k = tick["indices"][0]
        print(f"  {tick['label']:>4} at {tick['distance_invA']:.4f} 1/A: bands 4 and 5 at "
              f"{eigen[k, 3]:+.4f} and {eigen[k, 4]:+.4f} eV from E_F")
    print("state:", dft.bands_state()["state"])
