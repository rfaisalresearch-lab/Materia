"""Local HTTP server for the Materia desktop interface.

The interface is a local web application served by this process.  It binds to
127.0.0.1 only, is not reachable from the network, and has no authentication
because it is a single-user local tool -- exactly like a Jupyter kernel
started with a token disabled on localhost.  Nothing is sent anywhere: there
is no telemetry and no cloud dependency.

Routes are a thin translation of :class:`materia.desktop_ui.service.Service`.
"""

from __future__ import annotations

import json
import mimetypes
import os
import posixpath
import socket
import threading
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, Optional
from urllib.parse import parse_qs, unquote, urlparse

from ..version import __version__
from .service import Service

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


class ApiError(Exception):
    def __init__(self, message: str, status: int = 400, detail: Optional[dict] = None):
        super().__init__(message)
        self.status = status
        self.detail = detail or {}


def build_routes(service: Service) -> Dict[str, Callable[[dict], Any]]:
    """Map ``/api/<name>`` to a handler taking the decoded request body."""
    s = service
    return {
        "state": lambda b: s.state(),
        "materials": lambda b: {"materials": s.materials(), "errors": s.material_errors()},
        "material": lambda b: s.material_detail(b["id"]),
        "periodic_table": lambda b: {"elements": s.periodic_table()},
        "element": lambda b: s.element_reference(b["symbol"]),

        "wafer/create": lambda b: s.create_wafer(**b),
        "wafer/probe": lambda b: s.wafer_probe(float(b["x_mm"]), float(b["y_mm"])),
        "wafer/map": lambda b: s.wafer_map(b.get("field", "roughness"),
                                           int(b.get("n", 96))),
        "region/extract": lambda b: s.extract_region(**b),
        "structure/build": lambda b: s.build_surface(**b),
        "structure/reconstruct": lambda b: s.reconstruct(
            b.pop("reconstruction"), **b),
        "structure/reconstruction": lambda b: s.reconstruction_report(),
        "structure/render": lambda b: s.render_payload(
            int(b.get("max_atoms", 250000)), bool(b.get("bonds", True))),
        "structure/import": lambda b: s.import_structure(b["path"]),

        "select": lambda b: s.select(b.pop("mode"), **b),
        "atom/inspect": lambda b: s.inspect_atom(int(b["atom_id"])),
        "atom/ldos": lambda b: s.atom_ldos(int(b["atom_id"])),
        "edit": lambda b: s.edit(b.pop("op"), **b),

        "scan": lambda b: s.scan(b.get("technique", "stm"), b.get("settings"),
                                 bool(b.get("background", True))),
        "scan/payload": lambda b: s.scan_payload(b.get("key"), b.get("channel")),
        "scan/filtered": lambda b: s.scan_filtered(b.get("key"), b["channel"], b["method"]),
        "scan/features": lambda b: s.scan_features(b.get("key")),
        "scan/identify": lambda b: s.scan_identify(float(b["x_A"]), float(b["y_A"]),
                                                   b.get("key")),
        "scan/profile": lambda b: s.scan_line_profile(
            float(b["x0"]), float(b["y0"]), float(b["x1"]), float(b["y1"]),
            b.get("key"), b.get("channel"), int(b.get("n", 256))),
        "spectroscopy": lambda b: s.spectroscopy(
            float(b["x_A"]), float(b["y_A"]), float(b.get("height_A", 5.0)),
            b.get("bias_range", (-2.0, 2.0))),
        "force_curve": lambda b: s.force_curve(float(b["x_A"]), float(b["y_A"])),
        "instrument/options": lambda b: s.instrument_options(),

        "solvers": lambda b: s.solvers(),
        "gpaw/status": lambda b: s.gpaw_status(bool(b.get("refresh", False))),
        "gpaw/settings": lambda b: s.gpaw_settings(**b),
        "gpaw/energy": lambda b: s.gpaw_energy(**b),
        "dft/status": lambda b: s.dft_status(),
        "dft/spec": lambda b: s.dft_spec(b.get("variables")),
        "dft/run": lambda b: s.dft_run(b.get("variables"), bool(b.get("background", True)),
                                       bool(b.get("reuse_restart", False)),
                                       bool(b.get("keep_restart", False))),
        "dft/runs": lambda b: {"runs": s.dft_runs()},
        "dft/result": lambda b: s.dft_result(b.get("run_id")),
        "dft/array": lambda b: s.dft_array(b.get("run_id"), b.get("name", "density"),
                                           int(b.get("axis", 2))),
        "dft/study": lambda b: s.dft_study(b["parameter"], b["values"],
                                           b.get("observable", "energy_per_atom_eV"),
                                           float(b.get("tolerance", 1e-3)),
                                           b.get("variables"),
                                           bool(b.get("background", True))),
        "dft/study/result": lambda b: s.dft_study_result(b.get("study_id")),
        "dft/relax/spec": lambda b: s.dft_relax_spec(b.get("variables")),
        "dft/relax/run": lambda b: s.dft_relax(b.get("variables"),
                                               bool(b.get("background", True)),
                                               bool(b.get("apply", True)),
                                               b.get("timeout_s")),
        "dft/relax/runs": lambda b: {"runs": s.dft_relax_runs()},
        "dft/relax/result": lambda b: s.dft_relax_result(b.get("run_id")),
        "dft/relax/apply": lambda b: s.dft_relax_apply(b.get("run_id")),
        "dft/dos/spec": lambda b: s.dft_dos_spec(b.get("variables"), b.get("source")),
        "dft/dos/run": lambda b: s.dft_dos_run(b.get("variables"), b.get("source"),
                                               bool(b.get("background", True)),
                                               b.get("timeout_s")),
        "dft/dos/runs": lambda b: {"runs": s.dft_dos_runs()},
        "dft/dos/result": lambda b: s.dft_dos_result(b.get("run_id"),
                                                     max_points=int(b.get("max_points", 4001))),
        "dft/dos/export": lambda b: s.dft_dos_export(b["path"], b.get("run_id")),
        "dft/bands/spec": lambda b: s.dft_bands_spec(b.get("variables"), b.get("source")),
        "dft/bands/run": lambda b: s.dft_bands_run(b.get("variables"), b.get("source"),
                                                   bool(b.get("background", True)),
                                                   b.get("timeout_s")),
        "dft/bands/runs": lambda b: {"runs": s.dft_bands_runs()},
        "dft/bands/result": lambda b: s.dft_bands_result(b.get("run_id")),
        "dft/bands/export": lambda b: s.dft_bands_export(b["path"], b.get("run_id")),
        "dft/eos/spec": lambda b: s.dft_eos_spec(b.get("variables"), b.get("source")),
        "dft/eos/run": lambda b: s.dft_eos_run(b.get("variables"), b.get("source"),
                                               bool(b.get("background", True)),
                                               b.get("timeout_s")),
        "dft/eos/runs": lambda b: {"runs": s.dft_eos_runs()},
        "dft/eos/result": lambda b: s.dft_eos_result(b.get("run_id")),
        "dft/eos/export": lambda b: s.dft_eos_export(b["path"], b.get("run_id")),
        "dft/eos/apply": lambda b: s.dft_eos_apply(b.get("run_id")),
        "dft/ldos/spec": lambda b: s.dft_ldos_spec(b.get("variables"), b.get("source")),
        "dft/ldos/run": lambda b: s.dft_ldos_run(b.get("variables"), b.get("source"),
                                                 bool(b.get("background", True)),
                                                 b.get("timeout_s")),
        "dft/ldos/runs": lambda b: {"runs": s.dft_ldos_runs()},
        "dft/ldos/result": lambda b: s.dft_ldos_result(b.get("run_id")),
        "dft/ldos/slice": lambda b: s.dft_ldos_slice(b.get("run_id"), b.get("index")),
        "dft/ldos/image": lambda b: s.dft_ldos_image(b.get("run_id"),
                                                     b.get("mode", "constant-height"),
                                                     b.get("height_A"), b.get("isovalue")),
        "dft/ldos/export": lambda b: s.dft_ldos_export(b["path"], b.get("run_id")),
        "dft/ldos/image/export": lambda b: s.dft_ldos_image_export(
            b["path"], b.get("run_id"), b.get("mode", "constant-height"), b.get("height_A"),
            b.get("isovalue")),
        "claims/list": lambda b: s.claims_list(),
        "eam/status": lambda b: s.eam_status(),
        "eam/run": lambda b: s.eam_run(b.get("task", "energy"), b.get("potential"),
                                       b.get("settings"), bool(b.get("background", True))),
        "eam/result": lambda b: s.eam_result(b.get("run_id")),
        "eam/trajectory": lambda b: s.eam_trajectory(b.get("run_id"), b.get("frame")),
        "lammps/status": lambda b: s.lammps_status(bool(b.get("refresh", False))),
        "lammps/spec": lambda b: s.lammps_spec(b.get("task", "energy"), b.get("potential"),
                                               b.get("settings")),
        "lammps/run": lambda b: s.lammps_run(b.get("task", "energy"), b.get("potential"),
                                             b.get("settings"),
                                             bool(b.get("background", True)),
                                             bool(b.get("apply", True)),
                                             b.get("timeout_s")),
        "lammps/runs": lambda b: {"runs": s.lammps_runs()},
        "lammps/result": lambda b: s.lammps_result(b.get("run_id")),
        "lammps/trajectory": lambda b: s.lammps_trajectory(b.get("run_id"), b.get("frame")),
        "lammps/apply": lambda b: s.lammps_apply(b.get("run_id")),
        "electrostatics/status": lambda b: s.electrostatics_status(b.get("settings")),
        "electrostatics/assign": lambda b: s.electrostatics_assign(
            b["kind"], b.get("by_element"), b.get("by_atom_id"), b.get("source", "")),
        "electrostatics/clear": lambda b: s.electrostatics_clear(),
        "electrostatics/run": lambda b: s.electrostatics_run(
            b.get("settings"), bool(b.get("background", True))),
        "electrostatics/result": lambda b: s.electrostatics_result(b.get("run_id")),
        "solve": lambda b: s.solve(b.pop("task", "electronic"),
                                   b.pop("model", "recommended"),
                                   bool(b.pop("background", True)), **b),

        "script/run": lambda b: s.run_script(b["code"], bool(b.get("background", True))),
        "script/mode": lambda b: s.set_script_mode(b["mode"], bool(b.get("confirm", False))),
        "script/environment": lambda b: s.script_environment(),

        "job": lambda b: s.job_status(b["id"]),
        "job/cancel": lambda b: s.cancel_job(b["id"]),
        "jobs": lambda b: {"jobs": s.jobs.list(int(b.get("limit", 40)))},

        "project/undo": lambda b: s.undo(),
        "project/redo": lambda b: s.redo(),
        "project/save": lambda b: s.save_project(b["path"]),
        "project/open": lambda b: s.open_project(b["path"]),
        "project/new": lambda b: s.new_project(b.get("name", "Untitled project")),
        "project/checkpoint": lambda b: s.checkpoint(b["name"], b.get("note", "")),
        "project/restore": lambda b: s.restore_checkpoint(b["name"]),
        "project/provenance": lambda b: {"text": s.provenance_text()},
        "project/recover": lambda b: s.recover_project(),
        "project/recovery/discard": lambda b: s.discard_recovery(),

        "export/structure": lambda b: s.export_structure(b["path"], b.get("format")),
        "export/image": lambda b: s.export_image(
            b["path"], b.get("channel"), b.get("palette", "silver"), b.get("key")),
        "export/csv": lambda b: s.export_csv(b["path"], b.get("what", "atoms")),
        "export/npz": lambda b: s.export_npz(b["path"], b.get("parts")),
    }


class Handler(BaseHTTPRequestHandler):
    server_version = f"Materia/{__version__}"
    protocol_version = "HTTP/1.1"
    service: Service
    routes: Dict[str, Callable]

    def log_message(self, fmt: str, *args) -> None:
        if getattr(self.server, "verbose", False):
            super().log_message(fmt, *args)

    def _send(self, status: int, body: bytes, content_type: str,
              extra: Optional[dict] = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload, default=_default, allow_nan=False).encode()
        self._send(status, body, "application/json; charset=utf-8")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        if path.startswith("/api/"):
            query = {k: v[0] for k, v in parse_qs(parsed.query).items()}
            self._handle_api(path[len("/api/"):], query)
            return
        self._serve_static(path)

    def do_HEAD(self) -> None:
        self.do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        if not path.startswith("/api/"):
            self._json(404, {"error": "Not found"})
            return
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw.decode() or "{}")
        except json.JSONDecodeError as exc:
            self._json(400, {"error": f"Invalid JSON body: {exc}"})
            return
        self._handle_api(path[len("/api/"):], body)

    def _handle_api(self, name: str, body: dict) -> None:
        handler = self.routes.get(name.rstrip("/"))
        if handler is None:
            self._json(404, {"error": f"Unknown endpoint /api/{name}",
                             "available": sorted(self.routes)})
            return
        try:
            payload = _finite(handler(dict(body)))
            self._json(200, {"ok": True, "data": payload})
        except ApiError as exc:
            self._json(exc.status, {"ok": False, "error": str(exc), **exc.detail})
        except KeyError as exc:
            self._json(400, {"ok": False,
                             "error": f"Missing required parameter: {exc}"})
        except (ValueError, TypeError) as exc:
            self._json(400, {"ok": False, "error": str(exc),
                             "type": type(exc).__name__})
        except Exception as exc:
            self._json(500, {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                             "traceback": traceback.format_exc()})

    def _serve_static(self, path: str) -> None:
        if path in ("/", ""):
            path = "/index.html"
        safe = posixpath.normpath(path).lstrip("/")
        if safe.startswith(".."):
            self._json(403, {"error": "Forbidden"})
            return
        full = os.path.join(STATIC_DIR, safe)
        if not os.path.isfile(full):
            self._json(404, {"error": f"No such file: {safe}"})
            return
        ctype, _ = mimetypes.guess_type(full)
        if full.endswith(".js"):
            ctype = "text/javascript; charset=utf-8"
        elif full.endswith(".css"):
            ctype = "text/css; charset=utf-8"
        elif full.endswith(".html"):
            ctype = "text/html; charset=utf-8"
        with open(full, "rb") as fh:
            body = fh.read()
        self._send(200, body, ctype or "application/octet-stream")


def _finite(value: Any) -> Any:
    """Recursively replace non-finite floats with null."""
    import math

    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    return value


def _default(obj):
    """Encode objects the standard encoder cannot handle.

    Non-finite floats are mapped to null: JSON has no representation for them
    and a literal NaN or Infinity would produce a document no strict parser
    accepts.
    """
    import math

    from ..provenance.record import _jsonable

    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    value = _jsonable(obj)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def find_free_port(preferred: int = 8731, attempts: int = 40) -> int:
    for port in range(preferred, preferred + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No free port found in {preferred}..{preferred + attempts}")


def create_server(service: Optional[Service] = None, port: int = 8731,
                  verbose: bool = False) -> ThreadingHTTPServer:
    service = service or Service()
    handler = type("BoundHandler", (Handler,), {
        "service": service, "routes": build_routes(service)})
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    server.daemon_threads = True
    server.verbose = verbose
    server.service = service
    return server


def serve(port: Optional[int] = None, open_browser: bool = True,
          verbose: bool = False, project: Optional[str] = None) -> None:
    from ..project_format.project import Project
    service = Service(Project.load(project) if project else None)
    port = port or find_free_port()
    server = create_server(service, port, verbose)
    url = f"http://127.0.0.1:{port}/"
    print(f"Materia {__version__}")
    print(f"  interface : {url}")
    print(f"  materials : {len(service.library.ids())} definitions")
    print(f"  solvers   : {len(service.lab.solvers())} registered")
    print("  telemetry : none (Materia never contacts the network)")
    print("Press Ctrl-C to stop.")
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        report = service.shutdown()
        if report.get("subprocesses_terminated"):
            print(f"  stopped {report['subprocesses_terminated']} external "
                  "solver process(es)")
        server.server_close()
