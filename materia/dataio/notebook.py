"""Export a Materia project as a Jupyter notebook that reloads and inspects it.

The notebook is written, never executed.  It follows nbformat 4.5: Markdown
cells describe the project as it was at export, and code cells reload the
saved ``.materia`` file and inspect its structures, results, scans and arrays.

What is recorded
----------------
* The Materia version that wrote the notebook, the project's schema and
  software versions, its name and creation and modification times.
* The project file's absolute path and SHA-256, when a saved file is given.
  Without one the notebook embeds the summary only, and its loading cell
  stops with an explanation rather than guessing a path.
* For each structure: key, formula, atom count, cell, periodicity, material
  and an identity digest over atomic numbers, positions, cell and periodicity.
* For each result: quantity, unit, model, fidelity, origin, input digest,
  convergence, uncertainty and a bounded preview of the value.
* Run identifiers found in result and array keys and in result records.
* For each stored array: shape, dtype, unit, kind, SHA-256 and summary
  statistics.  Array data is never copied into the notebook.
* For each scan: technique, mode, channels and their units, resolution,
  extent and model.
* The limitations of the notebook itself.

Code cells verify on reload that the file's SHA-256, every structure digest
and every array checksum still match what was recorded, and say which do not.

Safety
------
Every name and text from the project enters code only as a Python ``repr``
literal, which cannot close its string or start new code, and enters Markdown
only after HTML and Markdown escaping with newlines removed.  The notebook is
serialised with ``json.dumps``.  Lists are bounded by ``max_items`` and value
previews by ``preview_elements``; nothing is truncated silently.

The file is written to a temporary file in the target folder, read back,
validated and compared with what was built, and only then moved into place.
The content depends only on the project and the options, never on the time of
export, so the same project always gives the same bytes.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from ..version import PROJECT_SCHEMA_VERSION, __version__

NOTEBOOK_SCHEMA = "materia.notebook"
NOTEBOOK_VERSION = 1
NBFORMAT = 4
NBFORMAT_MINOR = 5
MAX_TEXT = 200
CELL_ID = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
RUN_KEY = re.compile(r"^(?P<family>[A-Za-z0-9_-]+)::(?P<run>[^:]+)::")

DIGEST_SOURCE = '''def structure_digest(structure):
    """SHA-256 over atomic numbers, positions (A), cell (A) and periodic flags."""
    payload = hashlib.sha256()
    payload.update(np.ascontiguousarray(structure.numbers, dtype=np.int64).tobytes())
    payload.update(np.ascontiguousarray(structure.positions, dtype=np.float64).tobytes())
    payload.update(np.ascontiguousarray(structure.cell.matrix, dtype=np.float64).tobytes())
    payload.update(bytes(int(bool(p)) for p in structure.cell.pbc))
    return payload.hexdigest()
'''

_namespace: Dict[str, Any] = {"hashlib": hashlib, "np": np}
exec(DIGEST_SOURCE, _namespace)
structure_digest = _namespace["structure_digest"]


class NotebookError(ValueError):
    """The notebook could not be built, written or validated."""


def md(text: Any, limit: int = MAX_TEXT) -> str:
    """Text safe to place in a Markdown table cell or paragraph."""
    value = " ".join(str(text).split())
    if len(value) > limit:
        value = value[:limit] + f" ... ({len(value)} characters)"
    value = value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return re.sub(r"([\\`*_\[\]{}#+!|~$])", r"\\\1", value)


def _time(stamp: Any) -> str:
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(float(stamp)))
    except (TypeError, ValueError, OverflowError):
        return "unknown"


def _number(value: float) -> str:
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "-inf" if value < 0 else "inf"
    return f"{value:.6g}"


def preview(value: Any, elements: int) -> str:
    """A bounded, one-line description of a value."""
    if value is None:
        return "none"
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return _number(float(value))
    if isinstance(value, str):
        return repr(value if len(value) <= MAX_TEXT else value[:MAX_TEXT] + "...")
    if isinstance(value, dict):
        keys = sorted(str(k) for k in value)
        shown = ", ".join(keys[:elements])
        more = f", and {len(keys) - elements} more" if len(keys) > elements else ""
        return f"record with fields {shown}{more}"
    try:
        array = np.asarray(value)
    except Exception:
        return f"{type(value).__name__} object"
    if array.dtype.kind in "biuf" and array.size:
        if array.size <= elements:
            return f"{array.shape} {array.dtype}: " + ", ".join(
                _number(float(v)) for v in array.ravel())
        finite = array[np.isfinite(array)] if array.dtype.kind == "f" else array
        stats = (f"min {_number(float(finite.min()))}, max {_number(float(finite.max()))}, "
                 f"mean {_number(float(finite.mean()))}") if finite.size else "no finite value"
        return f"{array.shape} {array.dtype}: {stats}"
    return f"{type(value).__name__} object"


def _lines(text: str) -> List[str]:
    parts = text.split("\n")
    return [p + "\n" for p in parts[:-1]] + ([parts[-1]] if parts[-1] else [])


class _Builder:
    def __init__(self) -> None:
        self.cells: List[dict] = []

    def markdown(self, text: str) -> None:
        self.cells.append({"cell_type": "markdown", "id": f"materia-{len(self.cells) + 1:03d}",
                           "metadata": {}, "source": _lines(text)})

    def code(self, text: str) -> None:
        compile(text, f"<cell {len(self.cells) + 1}>", "exec")
        self.cells.append({"cell_type": "code", "id": f"materia-{len(self.cells) + 1:03d}",
                           "metadata": {}, "execution_count": None, "outputs": [],
                           "source": _lines(text)})


def _table(header: List[str], rows: Iterable[List[Any]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in rows:
        out.append("| " + " | ".join(md(v) for v in row) + " |")
    return "\n".join(out)


def _bounded(items: List[Any], limit: int, noun: str) -> Tuple[List[Any], str]:
    if len(items) <= limit:
        return items, ""
    return items[:limit], (f"\n\nOnly the first {limit} of {len(items)} {noun} are listed "
                           "here; the code cell below covers all of them.")


def run_identifiers(project) -> Dict[str, List[str]]:
    """Run identifiers per family, from keys like ``family::run::name`` and result records."""
    runs: Dict[str, set] = {}
    for key in list(project.results) + list(project.arrays):
        match = RUN_KEY.match(str(key))
        if match:
            runs.setdefault(match.group("family"), set()).add(match.group("run"))
    for key, result in project.results.items():
        run = (getattr(result, "extra", None) or {}).get("run_id")
        if isinstance(run, (str, int)):
            family = str(key).split("::")[0] if "::" in str(key) else "result"
            runs.setdefault(family, set()).add(str(run))
    return {family: sorted(ids) for family, ids in sorted(runs.items())}


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build(project=None, project_path: Optional[str] = None, *, max_items: int = 200,
          preview_elements: int = 20) -> dict:
    """The notebook as a dictionary, describing ``project`` or the saved file."""
    if max_items < 1 or preview_elements < 1:
        raise NotebookError("max_items and preview_elements must be at least 1.")
    path = None
    file_hash = None
    if project_path is not None:
        path = os.path.abspath(os.path.expanduser(str(project_path)))
        if not os.path.isfile(path):
            raise NotebookError(f"The project file {path} does not exist; save the project "
                                "first or export without a path to embed a summary only.")
        file_hash = file_sha256(path)
    if project is None:
        if path is None:
            raise NotebookError("Give a project, a saved project path, or both.")
        from ..project_format.project import Project
        project = Project.load(path)
    nb = _Builder()
    manifest_bits = {
        "name": project.name, "created": project.created, "modified": project.modified,
    }
    structures = sorted(project.structures.items())
    results = sorted(project.results.items())
    arrays = sorted(project.arrays.items())
    scans = sorted(project.scans.items())
    runs = run_identifiers(project)
    digests = {key: structure_digest(s) for key, s in structures}
    array_hashes = {key: stored.sha256() for key, stored in arrays}

    source = (f"the saved project file {md(path, 1000)}, SHA-256 {file_hash}"
              if path else "an unsaved project; only the summary below is embedded")
    nb.markdown(
        f"# Materia project: {md(project.name)}\n\n"
        f"This notebook describes {source}. It was written by Materia {md(__version__)} "
        "and has not been executed.\n\n"
        + _table(["field", "value"], [
            ["project name", project.name],
            ["project schema version", PROJECT_SCHEMA_VERSION],
            ["created", _time(manifest_bits["created"])],
            ["last modified", _time(manifest_bits["modified"])],
            ["structures", len(structures)], ["results", len(results)],
            ["stored arrays", len(arrays)], ["scans", len(scans)],
            ["checkpoints", len(project.checkpoints)],
            ["history entries", len(project.history.log)],
            ["active structure", project.active_structure_key or "none"],
        ]))
    nb.markdown(
        "## Limitations of this notebook\n\n"
        "* It records the project as it was when the notebook was written. The code "
        "cells reload the saved file and report any structure, array or file that no "
        "longer matches the recorded digests.\n"
        f"* Previews are bounded: values larger than {preview_elements} elements are "
        f"summarised by shape and statistics, text is cut at {MAX_TEXT} characters, and "
        f"each list shows at most {max_items} entries. Array data is never copied here.\n"
        "* Reloading needs Materia installed, at a version that reads project schema "
        f"{md(PROJECT_SCHEMA_VERSION)}.\n"
        "* Results are shown with the provenance they were saved with. The notebook does "
        "not recompute them, and a number here is only as good as the model named beside "
        "it.")

    load_lines = [
        "import hashlib",
        "import math",
        "from pathlib import Path",
        "",
        "import numpy as np",
        "",
        "import materia",
        "from materia.project_format.project import Project",
        "",
        f"WRITTEN_WITH = {__version__!r}",
        f"PROJECT_PATH = {path!r}",
        f"PROJECT_SHA256 = {file_hash!r}",
        "",
        "def output_text(value, limit=200):",
        "    raw = str(value)",
        "    lowered = raw.lower()",
        "    unsafe = (\"\\n\", \"\\r\", \"\\x1b\", \"<script\", \"javascript:\",",
        "              \"__import__\", \"os.system\", \"subprocess\")",
        "    if any(marker in lowered for marker in unsafe):",
        "        fingerprint = hashlib.sha256(raw.encode(\"utf-8\")).hexdigest()[:12]",
        "        return f\"[unsafe text omitted; sha256 {fingerprint}]\"",
        "    text = \" \".join(raw.split())",
        "    if len(text) > limit:",
        "        return text[:limit] + f\" ... ({len(text)} characters)\"",
        "    return text",
        "",
        "def output_value(value, elements=20):",
        "    if value is None or isinstance(value, (str, bool, int, float, np.generic)):",
        "        return output_text(value)",
        "    if isinstance(value, dict):",
        "        keys = sorted(output_text(key, 60) for key in value)",
        "        shown = \", \".join(keys[:elements])",
        "        more = f\", and {len(keys) - elements} more\" if len(keys) > elements else \"\"",
        "        return f\"record with fields {shown}{more}\"",
        "    try:",
        "        array = np.asarray(value)",
        "    except Exception:",
        "        return f\"{type(value).__name__} object\"",
        "    if array.dtype.kind in \"biuf\":",
        "        if array.size <= elements:",
        "            return output_text(array.tolist())",
        "        return f\"{tuple(array.shape)} {array.dtype}\"",
        "    return f\"{type(value).__name__} object\"",
        "",
        DIGEST_SOURCE.rstrip("\n"),
        "",
        "if PROJECT_PATH is None:",
        "    raise RuntimeError(\"This notebook was written from an unsaved project. Save the \"",
        "                       \"project, then set PROJECT_PATH to the .materia file.\")",
        "found = hashlib.sha256(Path(PROJECT_PATH).read_bytes()).hexdigest()",
        "if PROJECT_SHA256 is not None and found != PROJECT_SHA256:",
        "    print(\"The project file has changed since this notebook was written.\")",
        "project = Project.load(PROJECT_PATH)",
        "print(\"Materia\", materia.__version__, \"reading a notebook written by\", WRITTEN_WITH)",
        "print(\"Project:\", output_text(project.name))",
    ]
    nb.markdown("## Load the saved project")
    nb.code("\n".join(load_lines))

    nb.markdown("## Structures\n\n" + (_structure_table(structures, digests, project, max_items)
                                       if structures else "The project has no structures."))
    if structures:
        nb.code("\n".join([
            f"EXPECTED_STRUCTURE_DIGESTS = {_literal(digests)}",
            "for key, structure in sorted(project.structures.items()):",
            "    status = \"matches\" if structure_digest(structure) == \\",
            "        EXPECTED_STRUCTURE_DIGESTS.get(key) else \"differs from the record\"",
            "    print(output_text(key), structure.formula(), len(structure), \"atoms,\", status)",
            f"structure = project.structures[{structures[0][0]!r}]",
            "print(structure.positions[:5], \"A\")",
        ]))

    nb.markdown("## Results\n\n" + (_result_table(results, preview_elements, max_items)
                                    if results else "The project has no results."))
    if results:
        nb.code("\n".join([
            "for key, result in sorted(project.results.items()):",
            "    prov = result.provenance",
            "    print(output_text(key), \"|\", output_text(result.name), \"|\",",
            "          output_text(result.unit), \"|\", output_text(prov.model),",
            "          prov.fidelity.value, prov.origin.value, \"|\",",
            "          output_text(prov.inputs_digest))",
            f"result = project.results[{results[0][0]!r}]",
            "print(output_text(result.provenance.summary()))",
            f"print(output_value(result.value, {preview_elements}))",
        ]))

    run_text = ("\n".join(f"* {md(family)}: " + ", ".join(md(r) for r in ids[:max_items])
                          + (f", and {len(ids) - max_items} more" if len(ids) > max_items else "")
                          for family, ids in runs.items())
                if runs else "No run identifiers were found in the project's keys or records.")
    nb.markdown("## Run identifiers\n\n" + run_text)

    nb.markdown("## Stored arrays\n\n" + (_array_table(arrays, array_hashes, max_items)
                                          if arrays else "The project has no stored arrays."))
    if arrays:
        nb.code("\n".join([
            f"EXPECTED_ARRAY_SHA256 = {_literal(array_hashes)}",
            "for key, stored in sorted(project.arrays.items()):",
            "    found = stored.sha256()",
            "    status = \"matches\" if found == EXPECTED_ARRAY_SHA256.get(key) else \\",
            "        \"differs from the record\"",
            "    print(output_text(key), tuple(stored.shape), output_text(stored.unit), status)",
            f"data = project.arrays[{arrays[0][0]!r}].data",
            "print(data.shape, data.dtype)",
        ]))

    nb.markdown("## Scans\n\n" + (_scan_table(scans, max_items)
                                  if scans else "The project has no scans."))
    if scans:
        nb.code("\n".join([
            "for key, scan in sorted(project.scans.items()):",
            "    channels = [output_text(name) for name in scan.channel_names()]",
            "    print(output_text(key), output_text(scan.technique), output_text(scan.mode),",
            "          channels, scan.extent_A)",
            f"scan = project.scans[{scans[0][0]!r}]",
            "if scan.channel_names():",
            "    image = scan.channel()",
            "    print(output_text(scan.primary_channel), image.shape,",
            "          output_text(scan.units.get(scan.primary_channel, \"\")))",
            "    try:",
            "        import matplotlib.pyplot as plt",
            "    except ImportError:",
            "        plt = None",
            "    if plt is not None:",
            "        x0, y0, x1, y1 = scan.extent_A",
            "        plt.imshow(image, origin=\"lower\", extent=(x0, x1, y0, y1), cmap=\"gray\")",
            "        plt.xlabel(\"x (A)\")",
            "        plt.ylabel(\"y (A)\")",
            "        plt.colorbar(label=output_text(scan.units.get(scan.primary_channel, \"\")))",
            "        plt.show()",
        ]))

    nb.markdown("## Provenance record")
    nb.code("\n".join([
        "print(\"Materia project provenance\")",
        "print(\"Project:\", output_text(project.name))",
        "print(\"Results:\", len(project.results))",
        "print(\"Stored arrays:\", len(project.arrays))",
        "print(\"Scans:\", len(project.scans))",
    ]))

    metadata = {
        "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
        "language_info": {"name": "python"},
        "materia": {
            "schema": NOTEBOOK_SCHEMA, "version": NOTEBOOK_VERSION,
            "materia_version": __version__, "project_schema_version": PROJECT_SCHEMA_VERSION,
            "project_name": project.name, "project_path": path, "project_sha256": file_hash,
            "structure_digests": digests, "array_sha256": array_hashes,
            "run_identifiers": runs,
            "counts": {"structures": len(structures), "results": len(results),
                       "arrays": len(arrays), "scans": len(scans)},
            "options": {"max_items": max_items, "preview_elements": preview_elements},
        },
    }
    notebook = {"cells": nb.cells, "metadata": metadata,
                "nbformat": NBFORMAT, "nbformat_minor": NBFORMAT_MINOR}
    validate(notebook)
    return notebook


def _literal(mapping: Dict[str, str]) -> str:
    body = "".join(f"\n    {k!r}: {v!r}," for k, v in sorted(mapping.items()))
    return "{" + body + ("\n}" if body else "}")


def _structure_table(structures, digests, project, limit) -> str:
    shown, note = _bounded(structures, limit, "structures")
    rows = []
    for key, s in shown:
        lengths = ", ".join(f"{v:.4f}" for v in np.linalg.norm(s.cell.matrix, axis=1))
        rows.append([key + (" (active)" if key == project.active_structure_key else ""),
                     s.formula(), len(s), lengths,
                     "".join("T" if p else "F" for p in s.cell.pbc),
                     s.info.get("material_id", ""), digests[key]])
    return _table(["key", "formula", "atoms", "cell lengths (A)", "periodic",
                   "material", "identity digest"], rows) + note


def _result_table(results, elements, limit) -> str:
    shown, note = _bounded(results, limit, "results")
    rows = []
    for key, r in shown:
        prov = r.provenance
        conv = r.convergence
        state = ("not recorded" if conv is None else
                 ("converged" if conv.converged else "not converged"))
        uncertainty = ("" if r.uncertainty is None else
                       f"{preview(r.uncertainty, elements)} ({r.uncertainty_kind or 'unstated'})")
        value = (f"unsupported: {r.unsupported_reason}" if not r.supported
                 else preview(r.value, elements))
        rows.append([key, r.name, r.unit or "none", prov.model, prov.fidelity.value,
                     prov.origin.value, prov.inputs_digest or "none", state, uncertainty, value])
    return _table(["key", "quantity", "unit", "model", "fidelity", "origin", "input digest",
                   "convergence", "uncertainty", "value"], rows) + note


def _array_table(arrays, hashes, limit) -> str:
    shown, note = _bounded(arrays, limit, "arrays")
    rows = []
    for key, stored in shown:
        summary = stored.summary()
        stats = ("no finite value" if summary["min"] is None else
                 f"min {_number(summary['min'])}, max {_number(summary['max'])}, "
                 f"mean {_number(summary['mean'])}")
        rows.append([key, "x".join(str(n) for n in stored.shape) or "scalar",
                     summary["dtype"], stored.unit or "none", stored.kind, stats, hashes[key]])
    return _table(["key", "shape", "dtype", "unit", "kind", "statistics", "SHA-256"],
                  rows) + note


def _scan_table(scans, limit) -> str:
    shown, note = _bounded(scans, limit, "scans")
    rows = []
    for key, scan in shown:
        channels = ", ".join(f"{c} ({scan.units.get(c, '') or 'no unit'})"
                             for c in scan.channel_names()) or "none"
        rows.append([key, scan.technique, scan.mode, channels,
                     "x".join(str(v) for v in scan.resolution),
                     ", ".join(f"{float(v):.3f}" for v in scan.extent_A),
                     scan.provenance.model,
                     scan.unsupported_reason or ""])
    return _table(["key", "technique", "mode", "channels (unit)", "resolution",
                   "extent (A)", "model", "unsupported reason"], rows) + note


def serialise(notebook: dict) -> str:
    return json.dumps(notebook, indent=1, sort_keys=True, ensure_ascii=False) + "\n"


def export(path: str, project=None, project_path: Optional[str] = None, **options) -> dict:
    """Write the notebook atomically, read it back, validate it and return a summary."""
    target = os.path.abspath(os.path.expanduser(str(path)))
    if not target.lower().endswith(".ipynb"):
        raise NotebookError("The file name must end in .ipynb.")
    folder = os.path.dirname(target)
    if not os.path.isdir(folder):
        raise NotebookError(f"The folder {folder} does not exist.")
    notebook = build(project, project_path, **options)
    text = serialise(notebook)
    handle, temporary = tempfile.mkstemp(suffix=".ipynb", prefix=".materia-notebook-",
                                         dir=folder)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(text)
        with open(temporary, encoding="utf-8") as stream:
            written = json.load(stream)
        validate(written)
        if written != notebook:
            raise NotebookError("The notebook read back differs from the one built.")
        os.replace(temporary, target)
    except BaseException:
        if os.path.exists(temporary):
            os.remove(temporary)
        raise
    return {"path": target, "cells": len(notebook["cells"]),
            "bytes": os.path.getsize(target),
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "metadata": notebook["metadata"]["materia"]}


def _fail(where: str, why: str) -> None:
    raise NotebookError(f"Invalid notebook at {where}: {why}")


def _source(where: str, value: Any) -> None:
    if isinstance(value, str):
        return
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        _fail(where, "source must be a string or a list of strings")


def validate(notebook: Any) -> None:
    """Check the structure nbformat 4.5 requires, raising :class:`NotebookError`.

    Covers the required keys and types of the notebook, its metadata, and code,
    Markdown and raw cells, the forbidden extra keys, the cell id pattern and
    uniqueness, and the output-free state of an unexecuted code cell.
    """
    if not isinstance(notebook, dict):
        _fail("top level", "not an object")
    required = {"cells", "metadata", "nbformat", "nbformat_minor"}
    missing = required - set(notebook)
    if missing:
        _fail("top level", f"missing {sorted(missing)}")
    extra = set(notebook) - required
    if extra:
        _fail("top level", f"unexpected keys {sorted(extra)}")
    if notebook["nbformat"] != NBFORMAT or type(notebook["nbformat"]) is not int:
        _fail("nbformat", "must be the integer 4")
    minor = notebook["nbformat_minor"]
    if type(minor) is not int or minor < 5:
        _fail("nbformat_minor", "must be an integer of at least 5 for cell ids")
    metadata = notebook["metadata"]
    if not isinstance(metadata, dict):
        _fail("metadata", "not an object")
    kernel = metadata.get("kernelspec")
    if kernel is not None and not (isinstance(kernel, dict)
                                   and isinstance(kernel.get("name"), str)
                                   and isinstance(kernel.get("display_name"), str)):
        _fail("metadata.kernelspec", "needs string name and display_name")
    language = metadata.get("language_info")
    if language is not None and not (isinstance(language, dict)
                                     and isinstance(language.get("name"), str)):
        _fail("metadata.language_info", "needs a string name")
    cells = notebook["cells"]
    if not isinstance(cells, list):
        _fail("cells", "not a list")
    seen = set()
    allowed = {"markdown": {"id", "cell_type", "metadata", "source", "attachments"},
               "raw": {"id", "cell_type", "metadata", "source", "attachments"},
               "code": {"id", "cell_type", "metadata", "source", "outputs",
                        "execution_count"}}
    for k, cell in enumerate(cells):
        where = f"cells[{k}]"
        if not isinstance(cell, dict):
            _fail(where, "not an object")
        kind = cell.get("cell_type")
        if kind not in allowed:
            _fail(where, f"unknown cell_type {kind!r}")
        needed = {"id", "cell_type", "metadata", "source"} | (
            {"outputs", "execution_count"} if kind == "code" else set())
        if needed - set(cell):
            _fail(where, f"missing {sorted(needed - set(cell))}")
        if set(cell) - allowed[kind]:
            _fail(where, f"unexpected keys {sorted(set(cell) - allowed[kind])}")
        cell_id = cell["id"]
        if not isinstance(cell_id, str) or not CELL_ID.match(cell_id):
            _fail(where, "id must be 1 to 64 letters, digits, '-' or '_'")
        if cell_id in seen:
            _fail(where, f"duplicate id {cell_id!r}")
        seen.add(cell_id)
        if not isinstance(cell["metadata"], dict):
            _fail(where, "metadata is not an object")
        _source(where, cell["source"])
        if kind == "code":
            count = cell["execution_count"]
            if count is not None and (type(count) is not int or count < 0):
                _fail(where, "execution_count must be null or a non-negative integer")
            if not isinstance(cell["outputs"], list):
                _fail(where, "outputs is not a list")
