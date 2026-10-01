status = lammps.status()
print("LAMMPS available:", status["available"])
if not status["available"]:
    print(status["install_hint"])
else:
    print("LAMMPS:", status["version"], "at", status["path"], "via", status["kind"])
    cu = materials.load("copper").bulk(repeat=(3, 3, 3))
    spec = lammps.spec(
        cu,
        "md",
        steps=200,
        timestep_fs=1.0,
        ensemble="langevin",
        temperature_K=300.0,
        damping_fs=100.0,
        seed=11,
        initial_velocities="create",
        sample_every=10,
    )
    report = lammps.check(spec)
    print("refusals:", report["blocking"])
    if report["ok"]:
        run = lammps.run(spec=spec)
        record = run["run"].value
        print("command:", record["command_line"])
        print("input SHA-256:", record["inputs"]["in.lammps"]["sha256"])
        print("final energy / eV:", run["energy"].value)
        print("frames:", run["trajectory"].value["frames"])
        print("state:", lammps.state()["state"])
