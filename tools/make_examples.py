#!/usr/bin/env python3
"""Build the example projects and the documentation figures.

Everything here is produced by running the program, so the figures in the
documentation are the program's real output rather than illustrations.

Usage:  python tools/make_examples.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from materia.microscopy.afm import AFMSettings, AFMSimulator
from materia.microscopy.noise import NoiseModel
from materia.microscopy.stm import STMSettings, STMSimulator
from materia.microscopy.tip import Tip
from materia.multiscale import RegionSpec, WaferSpec
from materia.project_format import Project
from materia.solvers.tight_binding import TightBinding
from materia.structure_builder.defects import create_vacancy, substitute_atom
from materia.visualization.image import add_scale_bar, write_png
from materia.visualization.palette import apply_palette

PROJECTS = ROOT / "examples" / "projects"
FIGURES = ROOT / "docs" / "screenshots"


def render(scan, name: str, channel: str = None, palette: str = "silver",
           invert: bool = False) -> pathlib.Path:
    data = scan.channel(channel)
    rgb, meta = apply_palette(data, palette, mode="percentile", invert=invert)
    rgb, bar = add_scale_bar(rgb, scan.extent_A)
    path = FIGURES / name
    write_png(str(path), rgb, text={
        "software": "Materia",
        "model": scan.provenance.model,
        "channel": channel or scan.primary_channel,
        "provenance": scan.provenance.summary()[:800],
        "limits": f"{meta['vmin']:.6g}..{meta['vmax']:.6g}",
        "scale_bar": bar["label"],
    })
    print(f"  {path.name:<28} {data.shape[1]}x{data.shape[0]}  "
          f"range {np.ptp(data):.4f}  bar {bar['label']}")
    return path


def clean_silicon() -> Project:
    project = Project("Silicon (111) clean surface")
    project.create_wafer(WaferSpec(
        material_id="silicon", orientation=(1, 1, 1), diameter_mm=300,
        dopant="P", dopant_concentration_cm3=1e19, vacancy_density_cm2=0.0,
        seed=20260920))
    project.extract_region(RegionSpec(
        x_mm=0.0, y_mm=0.0, size_nm=(3.0, 3.0), depth_layers=5, vacuum_A=14.0,
        include_defects=False, include_dopants=False, max_atoms=8000))
    return project


def main() -> int:
    PROJECTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    print("clean Si(111)")
    project = clean_silicon()
    surface = project.structure
    simulator = STMSimulator(TightBinding("sp3s*-Si"), Tip("W"))

    scan = simulator.scan(surface, STMSettings(
        bias_V=1.0, resolution=(320, 320), kgrid=(2, 2),
        noise=NoiseModel.realistic(42)))
    project.add_scan(scan)
    render(scan, "si111_stm.png", "topography")
    render(scan, "si111_stm_current.png", "current", palette="gold")

    quiet = simulator.scan(surface, STMSettings(
        bias_V=1.0, resolution=(320, 320), kgrid=(2, 2),
        noise=NoiseModel.quiet(0)))
    render(quiet, "si111_stm_noiseless.png", "topography")

    filled = simulator.scan(surface, STMSettings(
        bias_V=-1.2, resolution=(320, 320), kgrid=(2, 2),
        noise=NoiseModel.realistic(42)))
    project.add_scan(filled)
    render(filled, "si111_stm_filled_states.png", "topography")

    afm = AFMSimulator(Tip("W", radius_A=40.0)).scan(surface, AFMSettings(
        mode="fm-afm", height_A=4.2, resolution=(260, 260),
        noise=NoiseModel.realistic(3)))
    project.add_scan(afm)
    render(afm, "si111_afm.png", "frequency_shift", invert=True)

    project.save_checkpoint("clean", "pristine ideal truncation")
    path = project.save(str(PROJECTS / "si111_clean"))
    print(f"  saved {pathlib.Path(path).name}")

    print("phosphorus donor")
    doped = clean_silicon()
    doped.name = "Silicon (111) with a phosphorus donor"
    structure = doped.structure
    z = structure.positions[:, 2]
    interior = [int(i) for i, zz, role in zip(structure.ids, z, structure.roles)
                if str(role) != "surface" and zz > z.min() + 3.0]
    target = interior[len(interior) // 2]
    doped.begin("Substitute phosphorus", "edit.substitute",
                {"atom_id": target, "element": "P"})
    substitute_atom(structure, target, "P")

    from materia.solvers.classical import stillinger_weber
    from materia.physics.potentials import StillingerWeber
    from materia.solvers.classical import ClassicalSolver

    solver = ClassicalSolver(StillingerWeber("Si", impurities=["P"]))
    result = solver.relax(structure, fmax_eV_A=0.02, max_steps=300, in_place=True)
    print(f"  relaxation: {result.convergence.message}")

    doped_simulator = STMSimulator(TightBinding("sp3s*-Si", impurities=["P"]), Tip("W"))
    doped_scan = doped_simulator.scan(structure, STMSettings(
        bias_V=1.0, resolution=(320, 320), kgrid=(2, 2),
        noise=NoiseModel.realistic(42)))
    doped.add_scan(doped_scan)
    render(doped_scan, "si111_phosphorus.png", "topography")
    doped.save_checkpoint("relaxed", "P substituted and relaxed")
    path = doped.save(str(PROJECTS / "si111_phosphorus_donor"))
    print(f"  saved {pathlib.Path(path).name}")

    print("surface vacancy")
    vacancy = clean_silicon()
    vacancy.name = "Silicon (111) with a surface vacancy"
    structure = vacancy.structure
    z = structure.positions[:, 2]
    surface_ids = [int(i) for i, zz in zip(structure.ids, z) if zz >= z.max() - 0.4]
    victim = surface_ids[len(surface_ids) // 2]
    vacancy.begin("Create vacancy", "edit.vacancy", {"atom_id": victim})
    create_vacancy(structure, victim)
    solver = ClassicalSolver(StillingerWeber("Si"))
    solver.relax(structure, fmax_eV_A=0.03, max_steps=300, in_place=True)
    vacancy_scan = simulator.scan(structure, STMSettings(
        bias_V=1.0, resolution=(320, 320), kgrid=(2, 2),
        noise=NoiseModel.realistic(42)))
    vacancy.add_scan(vacancy_scan)
    render(vacancy_scan, "si111_vacancy.png", "topography")
    path = vacancy.save(str(PROJECTS / "si111_vacancy"))
    print(f"  saved {pathlib.Path(path).name}")

    print("graphene")
    graphene = Project("Graphene monolayer")
    graphene.create_wafer(WaferSpec(material_id="graphene", orientation=(0, 0, 1),
                                    diameter_mm=100, vacancy_density_cm2=0.0,
                                    seed=7))
    graphene.extract_region(RegionSpec(size_nm=(2.5, 2.5), depth_layers=1,
                                       vacuum_A=14.0, include_defects=False,
                                       include_dopants=False, max_atoms=4000))
    sheet = graphene.structure
    graphene_scan = STMSimulator(TightBinding("pz-graphene"), Tip("W"),
                                 sample_work_function_eV=4.6).scan(
        sheet, STMSettings(bias_V=0.5, resolution=(320, 320), kgrid=(2, 2),
                           z_min_A=2.0, noise=NoiseModel.realistic(11)))
    if graphene_scan.supported:
        graphene.add_scan(graphene_scan)
        render(graphene_scan, "graphene_stm.png", "topography")
    else:
        print("  graphene scan unsupported:", graphene_scan.unsupported_reason)
    path = graphene.save(str(PROJECTS / "graphene_monolayer"))
    print(f"  saved {pathlib.Path(path).name}")

    print("silicon (100) 2x1 dimer reconstruction")
    dimer = Project("Silicon (100)-(2x1) dimer reconstruction")
    from materia.materials import load as load_material
    from materia.structure_builder import compare_to_reference, make_surface

    silicon = load_material("silicon")
    slab = make_surface(silicon, (1, 0, 0), size=(4, 4, 6), vacuum_A=14.0,
                        fix_bottom_layers=3, reconstruction="2x1-dimer")
    record = slab.info["reconstruction"]
    print(f"  {record['n_dimers']} dimers, bond "
          f"{record['measured']['bond_length_A']:.4f} A, "
          f"buckling {record['measured']['buckling_A']:.4f} A, "
          f"relaxed with {record['relaxation']['model']}")
    for row in compare_to_reference(slab, silicon)["rows"]:
        inside = "inside" if row["within_stated_uncertainty"] else "OUTSIDE"
        print(f"  {row['quantity']:<16} {row['measured']:.4f} vs "
              f"{row['reference']} +- {row['reference_uncertainty']} {row['unit']}"
              f"  ({inside} the quoted uncertainty)")
    dimer.add_structure(slab, activate=True)
    dimer_scan = STMSimulator(TightBinding("sp3s*-Si"), Tip("W")).scan(
        slab, STMSettings(bias_V=1.0, resolution=(320, 320), kgrid=(2, 2),
                          noise=NoiseModel.realistic(19)))
    if dimer_scan.supported:
        dimer.add_scan(dimer_scan)
        render(dimer_scan, "si100_2x1_stm.png", "topography")
    else:
        print("  Si(100) scan unsupported:", dimer_scan.unsupported_reason)
    dimer.save_checkpoint("reconstructed", "2x1 dimer rows, relaxed")
    path = dimer.save(str(PROJECTS / "si100_2x1_dimer"))
    print(f"  saved {pathlib.Path(path).name}")

    print("gallium arsenide (110)")
    gaas = Project("Gallium arsenide (110) cleavage surface")
    gaas.create_wafer(WaferSpec(material_id="gallium_arsenide", orientation=(1, 1, 0),
                                diameter_mm=150, vacancy_density_cm2=0.0, seed=3))
    gaas.extract_region(RegionSpec(size_nm=(2.5, 2.5), depth_layers=4, vacuum_A=14.0,
                                   include_defects=False, include_dopants=False,
                                   max_atoms=6000))
    gaas_scan = STMSimulator(TightBinding("sp3s*-GaAs"), Tip("W")).scan(
        gaas.structure, STMSettings(bias_V=-2.0, resolution=(320, 320), kgrid=(2, 2),
                                    noise=NoiseModel.realistic(5)))
    if gaas_scan.supported:
        gaas.add_scan(gaas_scan)
        render(gaas_scan, "gaas110_stm.png", "topography")
    else:
        print("  GaAs scan unsupported:", gaas_scan.unsupported_reason)
    path = gaas.save(str(PROJECTS / "gaas110_cleavage"))
    print(f"  saved {pathlib.Path(path).name}")

    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
