"""NumPy archive export through the service, the HTTP layer and the console."""

from __future__ import annotations

import json
import threading
import urllib.request

import numpy as np
import pytest

from materia.desktop_ui.server import create_server, find_free_port
from materia.desktop_ui.service import Service
from materia.materials import default_library
from materia.structure_builder.lattice import bulk


@pytest.fixture
def svc(tmp_path):
    service = Service(recovery_dir=str(tmp_path / "recovery"))
    service.project.add_structure(bulk(default_library().get("silicon_dioxide")), activate=True)
    yield service
    service.shutdown(grace_s=1.0)


def test_service_export_after_a_relaxation(svc, tmp_path):
    relax = svc.solve(task="relax", model="rigid-ion/bks-silica", fmax=1e-3, steps=2000,
                      background=False)
    assert relax["converged"]
    history_before = len(svc.project.history.log)
    out = svc.export_npz(str(tmp_path / "quartz.npz"))
    assert out["ok"], out
    assert len(svc.project.history.log) == history_before
    with np.load(out["path"], allow_pickle=False) as archive:
        assert np.array_equal(archive["structure/positions_A"], svc.structure.positions)
        assert float(archive["results/relax::energy"]) == relax["energy_eV"]
        assert np.array_equal(archive["results/relax::relaxation_history/fmax_eV_A"],
                              np.asarray(relax["history"]["fmax_eV_A"]))
        manifest = json.loads(str(archive["__manifest__"]))
    assert manifest["entries"]["results/relax::energy"]["model"] == \
        "classical/rigid-ion/bks-silica"
    assert out["entries"] == len(manifest["entries"])


def test_refusal_is_reported_not_raised(svc, tmp_path):
    out = svc.export_npz(str(tmp_path / "wrong.csv"))
    assert out["ok"] is False and ".npz" in out["error"]
    assert any(".npz" in w["text"] for w in svc.state()["warnings"])


def test_http_route_and_console(svc, tmp_path):
    port = find_free_port()
    server = create_server(svc, port=port)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        body = json.dumps({"path": str(tmp_path / "h.npz"), "parts": ["structure"]}).encode()
        reply = json.loads(urllib.request.urlopen(urllib.request.Request(
            f"http://127.0.0.1:{port}/api/export/npz", data=body,
            headers={"Content-Type": "application/json"})).read())["data"]
    finally:
        server.shutdown()
        server.server_close()
    assert reply["ok"] and reply["entries"] >= 5
    path = str(tmp_path / "c.npz")
    run = svc.runner.run(f"out = io.write_npz({path!r}, ['structure'])\n"
                         "print(out['entries'], sorted(out['skipped']))")
    assert run.ok, run.error or run.traceback
    with np.load(path, allow_pickle=False) as archive:
        assert "structure/numbers" in archive.files
