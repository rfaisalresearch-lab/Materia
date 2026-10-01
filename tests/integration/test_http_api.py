"""The HTTP layer the desktop window talks to."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

from materia.desktop_ui.server import create_server, find_free_port


@pytest.fixture(scope="module")
def server():
    port = find_free_port(9731)
    httpd = create_server(port=port)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    httpd.shutdown()
    httpd.server_close()


def post(base, route, payload=None):
    request = urllib.request.Request(
        f"{base}/api/{route}", method="POST",
        data=json.dumps(payload or {}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read())


def test_server_binds_to_loopback_only(server):
    assert server.startswith("http://127.0.0.1:")


def test_static_files_are_served(server):
    for path in ("/", "/css/app.css", "/js/main.js", "/menus.json"):
        with urllib.request.urlopen(f"{server}{path}", timeout=30) as response:
            assert response.status == 200
            body = response.read()
            assert body
            if path == "/":
                assert b"<title>Materia</title>" in body


def test_directory_traversal_is_refused(server):
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        urllib.request.urlopen(f"{server}/../pyproject.toml", timeout=30)
    assert excinfo.value.code in (403, 404)


def test_state_endpoint_returns_a_full_snapshot(server):
    payload = post(server, "state")
    assert payload["ok"] is True
    data = payload["data"]
    for key in ("version", "project", "wafer", "structure", "selection", "scales"):
        assert key in data


def test_unknown_endpoint_lists_the_available_ones(server):
    request = urllib.request.Request(f"{server}/api/nope", method="POST",
                                     data=b"{}",
                                     headers={"Content-Type": "application/json"})
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        urllib.request.urlopen(request, timeout=30)
    body = json.loads(excinfo.value.read())
    assert "state" in body["available"]


def test_malformed_json_is_reported(server):
    request = urllib.request.Request(f"{server}/api/state", method="POST",
                                     data=b"{not json",
                                     headers={"Content-Type": "application/json"})
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        urllib.request.urlopen(request, timeout=30)
    assert excinfo.value.code == 400
    assert "Invalid JSON" in json.loads(excinfo.value.read())["error"]


def test_missing_parameters_are_named(server):
    request = urllib.request.Request(f"{server}/api/material", method="POST",
                                     data=b"{}",
                                     headers={"Content-Type": "application/json"})
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        urllib.request.urlopen(request, timeout=30)
    body = json.loads(excinfo.value.read())
    assert "Missing required parameter" in body["error"]


def test_full_workflow_over_http(server):
    post(server, "project/new", {"name": "HTTP test"})
    post(server, "wafer/create", {"material_id": "silicon", "orientation": [1, 1, 1],
                                  "diameter_mm": 200})
    probe = post(server, "wafer/probe", {"x_mm": 1.0, "y_mm": 1.0})["data"]
    assert probe["on_wafer"] is True

    region = post(server, "region/extract",
                  {"size_nm": [1.5, 1.5], "depth_layers": 3, "max_atoms": 3000})["data"]
    assert region["n_atoms"] > 20

    render = post(server, "structure/render")["data"]
    assert render["n_total"] == region["n_atoms"]

    scan = post(server, "scan", {"technique": "stm", "background": False,
                                 "settings": {"resolution": [48, 48], "kgrid": [1, 1],
                                              "noise": "quiet"}})["data"]
    assert scan["supported"] is True
    assert scan["dtype"] == "float32"
    assert scan["data_b64"]

    features = post(server, "scan/features")["data"]
    assert features["features"]
    first = features["features"][0]

    identity = post(server, "scan/identify",
                    {"x_A": first["x_A"], "y_A": first["y_A"]})["data"]
    assert identity["nearest_element"] == "Si"

    inspection = post(server, "atom/inspect",
                      {"atom_id": identity["nearest_atom_id"]})["data"]
    assert inspection["identity"]["element"] == "Si"

    post(server, "select", {"mode": "ids", "ids": [identity["nearest_atom_id"]]})
    post(server, "edit", {"op": "substitute", "element": "P"})
    state = post(server, "state")["data"]
    assert state["structure"]["formula"].startswith("P")

    undone = post(server, "project/undo")["data"]
    assert undone["label"]


def test_script_endpoint_runs_python(server):
    result = post(server, "script/run",
                  {"code": "print(sum(range(5)))", "background": False})["data"]
    assert result["ok"] is True
    assert "10" in result["stdout"]


def test_non_finite_values_are_encoded_as_null(server):
    post(server, "project/new", {"name": "finite"})
    post(server, "wafer/create", {"material_id": "silicon", "orientation": [1, 1, 1],
                                  "miscut_deg": 0.0})
    probe = post(server, "wafer/probe", {"x_mm": 0.0, "y_mm": 0.0})["data"]
    assert probe["terrace_width_nm"] is None


def test_reconstruction_workflow_over_http(server):
    post(server, "project/new", {"name": "reconstruction over http"})
    detail = post(server, "material", {"id": "silicon"})["data"]
    flags = {r["id"]: r["supported"] for r in detail["reconstructions"]}
    assert flags == {"7x7-DAS": False, "2x1-dimer": True}

    built = post(server, "structure/build",
                 {"material": "silicon", "miller": [1, 0, 0], "size": [4, 2, 6],
                  "vacuum_A": 10.0, "fix_bottom_layers": 3,
                  "reconstruction": "2x1-dimer"})["data"]
    assert built["ok"] is True
    assert built["reconstruction"]["n_dimers"] == 4
    assert built["reconstruction"]["measured"]["bond_length_A"] > 0
    assert built["comparison"]["rows"]

    report = post(server, "structure/reconstruction")["data"]
    assert report["reconstructed"] is True
    assert report["record"]["id"] == "2x1-dimer"

    state = post(server, "state")["data"]
    assert state["structure"]["reconstruction"]["id"] == "2x1-dimer"
    assert state["structure"]["surface"]["reconstruction"] == "2x1-dimer"


def test_unsupported_reconstruction_is_declined_over_http(server):
    post(server, "project/new", {"name": "refusal over http"})
    out = post(server, "structure/build",
               {"material": "silicon", "miller": [1, 1, 1], "size": [2, 2, 4],
                "vacuum_A": 10.0, "reconstruction": "7x7-DAS"})["data"]
    assert out["ok"] is False
    assert out["unsupported"]["supported"] is False
    assert out["unsupported"]["result"]["value"] is None
    assert out["unsupported"]["suggested"]
    assert out["state"]["structure"] is None

    refused = post(server, "structure/build",
                   {"material": "silicon", "miller": [1, 0, 0], "size": [3, 2, 6],
                    "vacuum_A": 10.0, "reconstruction": "2x1-dimer"})["data"]
    assert refused["ok"] is False
    assert "even number" in refused["error"]


def test_reconstruct_and_undo_over_http(server):
    post(server, "project/new", {"name": "undo over http"})
    built = post(server, "structure/build",
                 {"material": "silicon", "miller": [1, 0, 0], "size": [4, 2, 6],
                  "vacuum_A": 10.0, "fix_bottom_layers": 3})["data"]
    assert built["ok"] is True
    state = post(server, "state")["data"]
    assert state["project"]["structures"] == [built["key"]]

    applied = post(server, "structure/reconstruct",
                   {"reconstruction": "2x1-dimer"})["data"]
    assert applied["ok"] is True
    state = post(server, "state")["data"]
    assert state["project"]["structures"] == [built["key"]]
    assert state["project"]["undo_label"] == "Apply 2x1-dimer reconstruction"
    assert state["structure"]["reconstruction"]["id"] == "2x1-dimer"

    post(server, "project/undo")
    state = post(server, "state")["data"]
    assert state["structure"]["reconstruction"] is None
    assert state["structure"]["surface"].get("reconstruction") is None

    post(server, "project/redo")
    state = post(server, "state")["data"]
    assert state["structure"]["reconstruction"]["id"] == "2x1-dimer"


def test_a_non_converged_reconstruction_is_not_offered_as_a_comparison(server):
    post(server, "project/new", {"name": "stalled over http"})
    built = post(server, "structure/build",
                 {"material": "silicon", "miller": [1, 0, 0], "size": [4, 2, 6],
                  "vacuum_A": 10.0, "fix_bottom_layers": 3,
                  "reconstruction": "2x1-dimer", "reconstruction_max_steps": 1})["data"]
    assert built["ok"] is True
    assert built["reconstruction"]["relaxed"] is False
    assert built["reconstruction"]["geometry_status"] == "not-converged"
    assert built["comparison"]["comparable"] is False
    assert all(row["within_stated_uncertainty"] is None
               for row in built["comparison"]["rows"])
