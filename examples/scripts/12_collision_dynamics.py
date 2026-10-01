projectile = materials.load("copper").bulk(repeat=(1, 1, 1), activate=False)
target = materials.load("copper").bulk(repeat=(1, 1, 1), activate=False)
impact = collisions.prepare(
    projectile,
    target,
    relative_speed_A_fs=0.06,
    impact_parameter_A=1.0,
    gap_A=3.0,
    padding_A=8.0,
)
run = collisions.run(
    impact,
    potential="Cu-Zhou04",
    steps=600,
    dt_fs=0.1,
    sample_every=3,
)
print("run:", run["energy"].extra["run_id"])
print("frames:", run["trajectory"].value["frames"])
print("scope:", impact.info["collision"]["model_scope"])
