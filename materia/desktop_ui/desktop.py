"""Native desktop shell.

Materia is a desktop application, not a web page.  This module opens a real
operating-system window backed by the platform's native web view
(WKWebView on macOS, WebView2 on Windows, WebKitGTK on Linux), installs a
native menu bar, and wires the native file dialogs.  There is no address bar,
no browser tab and no external browser process: the interface is rendered
inside the application's own window by the system's view component, which is
the same architecture Tauri, Electron and Qt WebEngine applications use.

The computational core runs in-process.  A loopback HTTP endpoint on a random
free port carries the interface-to-core messages; it is bound to 127.0.0.1,
is never advertised, and closes with the window.  Nothing leaves the machine.

Run it with::

    materia            # or: python -m materia.desktop_ui.desktop

``tools/make_macos_app.py`` packages exactly this entry point as a
double-clickable ``.app`` bundle.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..version import __version__
from .server import create_server, find_free_port
from .service import Service

HERE = Path(__file__).resolve().parent
MENU_SPEC = HERE / "menus.json"

WINDOW_TITLE = f"Materia {__version__}"
MIN_SIZE = (1180, 720)
DEFAULT_SIZE = (1560, 980)


class NativeBridge:
    """Methods the interface can call on the host application.

    Exposed to the page as ``window.pywebview.api``.  Only these methods are
    reachable: the page cannot execute arbitrary host code.
    """

    def __init__(self) -> None:
        self.window = None

    def open_file(self, title: str = "Open", filters: Optional[List[str]] = None,
                  directory: str = "") -> Optional[str]:
        import webview
        result = self.window.create_file_dialog(
            webview.OPEN_DIALOG, directory=directory or str(Path.home()),
            allow_multiple=False, file_types=tuple(filters or ("All files (*.*)",)))
        return result[0] if result else None

    def save_file(self, title: str = "Save", filename: str = "",
                  directory: str = "") -> Optional[str]:
        import webview
        result = self.window.create_file_dialog(
            webview.SAVE_DIALOG, directory=directory or str(Path.home()),
            save_filename=filename)
        if not result:
            return None
        return result if isinstance(result, str) else result[0]

    def choose_folder(self, title: str = "Choose folder") -> Optional[str]:
        import webview
        result = self.window.create_file_dialog(webview.FOLDER_DIALOG)
        return result[0] if result else None

    def set_title(self, title: str) -> bool:
        if self.window:
            self.window.set_title(f"{title} - Materia")
        return True

    def platform(self) -> dict:
        return {"native": True, "os": sys.platform, "version": __version__,
                "python": sys.version.split()[0]}


def _build_native_menu(bridge: NativeBridge):
    """Build the platform menu bar from the shared menu specification."""
    import webview.menu as wm

    spec: Dict[str, List[dict]] = json.loads(MENU_SPEC.read_text())

    def make_action(action: str):
        def run():
            if bridge.window is None:
                return
            payload = json.dumps(action)
            bridge.window.evaluate_js(
                f"window.__materia && window.__materia.dispatch({payload})")
        return run

    menus = []
    for title, entries in spec.items():
        items = []
        for entry in entries:
            if entry.get("sep"):
                items.append(wm.MenuSeparator())
                continue
            items.append(wm.MenuAction(entry["label"].replace("…", "..."),
                                       make_action(entry["action"])))
        menus.append(wm.Menu(title, items))
    return menus


def launch(project: Optional[str] = None, port: Optional[int] = None,
           debug: bool = False, fullscreen: bool = False) -> int:
    """Open the application window.  Returns a process exit code."""
    try:
        import webview
    except ImportError:
        sys.stderr.write(
            "Materia's desktop window needs the 'pywebview' package, which wraps\n"
            "the operating system's native web view.\n\n"
            "    pip install pywebview\n\n"
            "If you cannot install it, 'materia serve' starts the same interface\n"
            "on a local port that you can open in a browser instead.\n")
        return 2

    from ..project_format.project import Project
    service = Service(Project.load(project) if project else None)
    port = port or find_free_port()
    server = create_server(service, port, verbose=debug)
    thread = threading.Thread(target=server.serve_forever, name="materia-core",
                             daemon=True)
    thread.start()

    bridge = NativeBridge()
    url = f"http://127.0.0.1:{port}/?native=1"
    window = webview.create_window(
        WINDOW_TITLE, url,
        js_api=bridge,
        width=DEFAULT_SIZE[0], height=DEFAULT_SIZE[1],
        min_size=MIN_SIZE,
        background_color="#1d1f21",
        text_select=True,
        confirm_close=False,
        fullscreen=fullscreen,
    )
    bridge.window = window

    def on_closed():
        try:
            service.shutdown()
        except Exception:
            pass
        try:
            server.shutdown()
            server.server_close()
        except Exception:
            pass

    window.events.closed += on_closed

    print(f"Materia {__version__} - native window")
    print(f"  core        : in-process, loopback port {port} (127.0.0.1 only)")
    print(f"  materials   : {len(service.library.ids())} definitions")
    print(f"  solvers     : {len(service.solvers()['registered'])} registered")
    print("  telemetry   : none")

    menu = None
    try:
        menu = _build_native_menu(bridge)
    except Exception as exc:
        print(f"  note        : native menu unavailable ({exc}); "
              "the in-window menu bar is used instead.")

    kwargs: Dict[str, Any] = {"debug": debug}
    if menu:
        kwargs["menu"] = menu
    webview.start(**kwargs)
    on_closed()
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(
        prog="materia-desktop",
        description="Materia native desktop window.")
    parser.add_argument("--project", default=None, help="open a .materia file")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--debug", action="store_true",
                        help="enable the web view inspector")
    parser.add_argument("--fullscreen", action="store_true")
    args = parser.parse_args(argv)
    return launch(project=args.project, port=args.port, debug=args.debug,
                  fullscreen=args.fullscreen)


if __name__ == "__main__":
    raise SystemExit(main())
