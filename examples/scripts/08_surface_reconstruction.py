"""Build the Si(100)-(2x1) dimer reconstruction and measure it.

The generator imposes the dimer pairing and nothing else; the geometry printed
below is the relaxed minimum of the Stillinger-Weber potential. The script
finishes by asking for a reconstruction Materia cannot generate, to show what a
refusal looks like.

Run with:
    materia run examples/scripts/08_surface_reconstruction.py
or paste into the embedded Python console.
"""

silicon = materials.load("silicon")

print("declared reconstructions")
for entry in silicon.reconstructions():
    flag = "available" if entry["supported"] else "not implemented"
    print(f"   {entry['id']:<22} ({''.join(str(v) for v in entry['orientation'])})  {flag}")
    if not entry["supported"]:
        print("      ", entry["reason"])

print("\nbuilding Si(100) and dimerising the upper face")
surface = silicon.create_surface(size=(4, 2, 8), orientation=(1, 0, 0),
                                 vacuum_angstrom=12, fix_bottom_layers=4,
                                 reconstruction="2x1-dimer")
record = surface.reconstruction()
relaxation = record["relaxation"]
measured = record["measured"]
check = record["periodicity_check"]

print(f"   {len(surface)} atoms, {record['n_dimers']} dimers in {record['n_rows']} rows")
print(f"   dimer axis      {record['dimer_axis']} (derived from the back-bond geometry)")
print(f"   1x1 spacing     {record['unreconstructed_spacing_A']:.4f} A")
print(f"   relaxed with    {relaxation['model']}")
print(f"   converged       {relaxation['converged']} in {relaxation['steps']} steps,"
      f" residual {relaxation['max_force_eV_A']:.4f} eV/A")
print(f"   geometry        {record['geometry_status']}")
print(f"   dimer bond      {measured['bond_length_A']:.4f} A"
      f"  (spread {measured['bond_length_spread_A']:.2e} A over the cell)")
print(f"   contraction     {measured['contraction_A']:.4f} A from the 1x1 spacing")
print(f"   buckling        {measured['buckling_A']:.4f} A")

print("\nsymmetry check")
print(f"   1x1 translation broken  {not check['invariant_under_1x1_shift']}")
print(f"   2x1 translation held    {check['invariant_under_2x1_shift']}"
      f"  (lattice vector: {check['2x1_shift_is_lattice_vector']})")

print("\nmeasured against published structural data")
comparison = surface.compare_reconstruction()
if not comparison["comparable"]:
    print("   NOT COMPARABLE:", comparison["blocked_reason"])
for row in comparison["rows"]:
    inside = ("n/a" if row["within_stated_uncertainty"] is None
              else "yes" if row["within_stated_uncertainty"] else "NO")
    print(f"   {row['quantity']:<16} computed {row['measured']:8.4f} {row['unit']}"
          f"   published {row['reference']} +- {row['reference_uncertainty']} {row['unit']}"
          f"   deviation {row['deviation']:+.4f}   inside uncertainty: {inside}")
print("  ", comparison["note"])
for source in {row["source"] for row in comparison["rows"]}:
    print("   source:", source)

print("\nprovenance of the reconstructed geometry")
provenance = record["provenance"]
print(f"   model    {provenance['model']}")
print(f"   origin   {provenance['origin']}  fidelity {provenance['fidelity']}")
for approximation in provenance["approximations"]:
    print("   -", approximation)

print("\nthe same reconstruction with a step budget too small to converge")
stalled = silicon.create_surface(size=(4, 2, 8), orientation=(1, 0, 0),
                                 vacuum_angstrom=12, fix_bottom_layers=4,
                                 reconstruction="2x1-dimer",
                                 reconstruction_max_steps=1, activate=False)
stalled_record = stalled.reconstruction()
stalled_comparison = stalled.compare_reconstruction()
print(f"   geometry        {stalled_record['geometry_status']}")
print(f"   relaxed         {stalled_record['relaxed']}"
      f"  (attempted: {stalled_record['relaxation_attempted']})")
print(f"   dimer bond      {stalled_record['measured']['bond_length_A']:.4f} A"
      "  <- not a prediction")
print(f"   comparable      {stalled_comparison['comparable']}")
print("   reason         ", stalled_comparison["blocked_reason"])

print("\nasking for a reconstruction this build cannot generate")
try:
    silicon.create_surface(size=(2, 2, 4), orientation=(1, 1, 1),
                           vacuum_angstrom=12, reconstruction="7x7-DAS")
except UnsupportedRequest as exc:
    refusal = exc.as_dict()
    print("   refused :", refusal["reason"])
    print("   asked   :", refusal["reference"])
    print("   could do:", ", ".join(refusal["suggested"]))
    print("   nothing approximate was returned and no structure was added.")
