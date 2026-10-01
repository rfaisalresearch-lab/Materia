"""What Materia does when it cannot answer.

Nothing here fails. Every refusal is a result object with a reason and a list
of solvers that could produce the quantity.
"""

silicon = materials.load("silicon", orientation="111")
surface = silicon.create_surface(size=(3, 3, 3), vacuum_angstrom=12)

print("1. asking a non-self-consistent model for self-consistency")
out = surface.solve(self_consistent=True)
refusal = out.get("self_consistency")
if refusal is not None and not refusal.supported:
    print("   refused:", refusal.unsupported_reason)
    print("   try    :", ", ".join(refusal.suggested_models))

print("\n2. asking a classical potential for a density of states")
from materia.solvers import Capability
from materia.solvers.classical import stillinger_weber

classical = stillinger_weber("Si")
blocked = classical.require(Capability.LOCAL_DOS)
print("   refused:", blocked.unsupported_reason)
print("   try    :", ", ".join(blocked.suggested_models))

print("\n3. asking for the electronic structure of gold")
gold = materials.load("gold", orientation="111").create_surface(size=(2, 2, 2))
try:
    gold.solve()
except Exception as exc:
    print("   refused:", exc)

print("\n4. asking for a calculation that needs an uninstalled solver")
from materia.solvers.external import ADAPTERS, ExternalSolver

spec = next(a for a in ADAPTERS if a.name == "external:quantum-espresso")
solver = ExternalSolver(spec)
if solver.installed:
    print("   Quantum ESPRESSO is installed here; nothing to demonstrate")
else:
    result = solver.run(surface.structure, task="band_structure", kpoints=12)
    blocked = result["band_structure"]
    print("   refused:", blocked.unsupported_reason)
    preserved = blocked.extra["preserved_request"]
    print("   kept   :", preserved["task"], preserved["kwargs"])
    print("   note   :", preserved["note"])

print("\n5. asking for a reconstruction that is declared but not implemented")
definition = materials.load("silicon").definition
for reconstruction in definition.reconstructions:
    status = "implemented" if reconstruction.implemented else "NOT implemented"
    print(f"   {reconstruction.id}: {status}")
    if not reconstruction.implemented:
        print("      ", reconstruction.note)
        print("      reference:", reconstruction.reference)
