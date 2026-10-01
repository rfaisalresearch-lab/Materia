"""Static interface contract checks for the native and browser shells."""

from __future__ import annotations

import json
import re
from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
STATIC = ROOT / "materia" / "desktop_ui" / "static"


class InterfaceParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = []
        self.controls = []

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if "id" in values:
            self.ids.append(values["id"])
        if tag in {"button", "input", "select", "textarea", "canvas"}:
            self.controls.append((tag, values))


def test_interface_ids_are_unique_and_complete():
    parser = InterfaceParser()
    parser.feed((STATIC / "index.html").read_text())
    assert len(parser.ids) == len(set(parser.ids))
    required = {
        "menubar", "toolbar", "project-tree", "material-list", "canvas-wafer",
        "canvas-region", "canvas-atoms", "canvas-probe", "canvas-plot",
        "script-editor", "statusbar", "modal-backdrop", "panel-examples",
        "example-search", "example-category", "example-list", "example-detail",
    }
    assert required <= set(parser.ids)


def test_native_and_browser_menus_match_and_have_handlers():
    browser_menu = json.loads((STATIC / "menus.json").read_text())
    native_menu = json.loads((ROOT / "materia" / "desktop_ui" / "menus.json").read_text())
    assert native_menu == browser_menu
    actions = {
        item["action"]
        for entries in browser_menu.values()
        for item in entries
        if "action" in item
    }
    source = (STATIC / "js" / "main.js").read_text()
    handlers = set(re.findall(r"case '([^']+)'", source))
    assert actions <= handlers
    assert "help.selfcheck" in actions


def test_interactive_controls_have_an_accessible_name():
    parser = InterfaceParser()
    parser.feed((STATIC / "index.html").read_text())
    unnamed = []
    for tag, attrs in parser.controls:
        if tag == "button":
            continue
        if attrs.get("aria-label") or attrs.get("title") or attrs.get("id"):
            continue
        unnamed.append((tag, attrs))
    assert unnamed == []


def test_console_example_snippets_are_valid_python():
    """Every snippet offered in the Python console is run verbatim by one click,
    so a syntax error in one is a broken feature, not a typo in a comment."""
    import ast

    source = (STATIC / "js" / "examples.js").read_text()
    blocks = re.findall(r"String\.raw`(.*?)`\),", source, re.S)
    assert len(blocks) >= 28
    for block in blocks:
        ast.parse(block)
        assert not any(line.lstrip().startswith("#") for line in block.splitlines())


def test_example_laboratory_covers_every_declared_feature_area():
    source = (STATIC / "js" / "examples.js").read_text()
    declared = set(re.findall(r"^  \['([a-z0-9-]+)', '[^']+'\],$", source, re.M))
    definitions = source[source.index("export const EXAMPLES"):source.index(
        "export function exampleCoverage")]
    headers = re.findall(r"define\((.*?)String\.raw`", definitions, re.S)
    covered = set()
    for header in headers:
        feature_blocks = re.findall(r"\[([^\[\]]*)\]", header)
        assert feature_blocks
        covered.update(re.findall(r"'([a-z0-9-]+)'", feature_blocks[-1]))
    assert len(declared) == 42
    assert declared == covered
    assert "two-atom-head-on" in definitions
    assert "two-atom-glancing" in definitions
    assert "quantum-state-explorer" in definitions
    assert "quantum-density-dft" in definitions
    assert definitions.count("'3D models'") >= 6


def test_example_tab_loads_and_runs_catalog_entries():
    html = (STATIC / "index.html").read_text()
    main = (STATIC / "js" / "main.js").read_text()
    assert 'data-panel="examples"' in html
    assert "['tree', 'materials', 'examples', 'selection']" in main
    for piece in ("installExamples()", "function renderExamples()", "loadExample(example)",
                  "await runScript()", "exampleCoverage()"):
        assert piece in main
    assert "This is not product completion" in main
    assert "`${EXAMPLES.length} examples`" in main


def test_collision_examples_open_a_timeline_in_the_atomic_view():
    html = (STATIC / "index.html").read_text()
    main = (STATIC / "js" / "main.js").read_text()
    css = (STATIC / "css" / "app.css").read_text()
    for item in ("trajectory-stage", "trajectory-stage-range", "trajectory-stage-play",
                 "trajectory-stage-time", "trajectory-stage-readout"):
        assert f'id="{item}"' in html
    for piece in ("installTrajectoryStage()", "syncTrajectoryStage(payload)",
                  "seekTrajectoryStage", "example.features.includes('collision-playback')"):
        assert piece in main
    assert ".trajectory-stage" in css


def test_primary_scientific_workflows_are_visible_above_the_viewport():
    html = (STATIC / "index.html").read_text()
    main = (STATIC / "js" / "main.js").read_text()
    css = (STATIC / "css" / "app.css").read_text()
    for action in ("build", "simulate", "collide", "quantum", "microscopy", "measure",
                   "results", "export"):
        assert f'data-workflow="{action}"' in html
    for piece in ("installWorkflowBar()", "openExample('Collisions'",
                  "openExample('Quantum'", "await dispatch('export.npz')"):
        assert piece in main
    assert ".workflow-bar" in css
    assert "border-radius: 0" in css


def test_shipped_example_scripts_are_valid_python():
    import ast

    scripts = sorted((ROOT / "examples" / "scripts").glob("*.py"))
    assert len(scripts) >= 8
    for script in scripts:
        ast.parse(script.read_text(), filename=str(script))


def test_material_build_panel_offers_reconstructions():
    source = (STATIC / "js" / "panels.js").read_text()
    assert "mat-recon" in source
    assert "reconstruction: rec.value" in source
    assert "r.supported" in source


def test_interface_uses_one_blue_accent_without_green_status_styling():
    css = (STATIC / "css" / "app.css").read_text()
    scripts = "\n".join(path.read_text() for path in (STATIC / "js").glob("*.js"))
    assert "--accent:        #2b5d8c" in css
    assert "--green" not in css
    assert "on-green" not in css + scripts
    assert "tag.calculated" in css
    assert "table.hud td.v.confidence" in css


def test_interface_copy_contains_no_long_dashes():
    files = list(STATIC.rglob("*")) + [ROOT / "materia" / "desktop_ui" / "desktop.py"]
    text = "\n".join(path.read_text() for path in files if path.is_file())
    assert "—" not in text
    assert "–" not in text
    assert "&mdash;" not in text
    assert "&ndash;" not in text


def test_every_route_the_interface_calls_exists():
    from materia.desktop_ui.server import build_routes
    from materia.desktop_ui.service import Service

    source = (STATIC / "js" / "api.js").read_text()
    called = set(re.findall(r"call(?:Job)?\('([a-z_/]+)'", source))
    routes = set(build_routes(Service()))
    assert {"electrostatics/status", "electrostatics/assign", "electrostatics/clear",
            "electrostatics/run", "electrostatics/result"} <= called
    assert called <= routes, sorted(called - routes)


def test_electrostatics_panel_names_its_controls_and_hides_no_unfinished_mode():
    source = (STATIC / "js" / "panels.js").read_text()
    body = source[source.index("export function renderElectrostatics"):
                  source.index("function renderElectrostaticsResult")]
    for control in ("es-kind", "es-source", "es-accuracy", "es-surrounding",
                    "es-background", "es-check"):
        assert f"id: '{control}'" in body
    assert "value: 'per-atom'" not in body
    assert "Compute electrostatics" in body


def test_eam_panel_names_its_controls_and_offers_only_real_tasks():
    source = (STATIC / "js" / "panels.js").read_text()
    body = source[source.index("export function renderEam"):
                  source.index("function renderEamResult")]
    for control in ("eam-potential", "eam-task", "eam-fmax", "eam-maxsteps", "eam-mdsteps",
                    "eam-dt", "eam-temp", "eam-thermostat", "eam-seed",
                    "eam-sample-every"):
        assert f"id: '{control}'" in body
    tasks = set(re.findall(r"el\('option', \{ value: '([a-z]+)' \}", body))
    assert {"energy", "relax", "md"} <= tasks
    assert "Run EAM" in body
    for field in ("sha256", "license", "citations", "cutoff_A", "notes"):
        assert f"chosen.{field}" in body
    result = source[source.index("function renderEamResult"):]
    assert "r.current" in result and "r.convergence" in result
    for control in ("eam-playback-toggle", "eam-frame", "eam-playback-speed",
                    "eam-fragment-readout"):
        assert f"id: '{control}'" in result


def test_trajectory_view_keeps_the_saved_render_payload_immutable():
    source = (STATIC / "js" / "atoms-view.js").read_text()
    assert "this.data = payload ? { ...payload" in source
    assert "bounds: payload.bounds ? payload.bounds.map((row) => [...row])" in source
    assert "setTrajectoryFrame(payload)" in source
    assert "this.data.bounds = payload.bounds" in source


def test_dft_relaxation_panel_names_its_controls_and_shows_convergence():
    source = (STATIC / "js" / "dft-panel.js").read_text()
    controls = source[source.index("function renderRelaxControls"):
                      source.index("function renderRelaxRuns")]
    for control in ("dft-relax-mode", "dft-relax-optimizer", "dft-relax-fmax",
                    "dft-relax-steps", "dft-relax-maxstep", "dft-relax-symmetry",
                    "dft-relax-stress-tol", "dft-relax-pressure", "dft-relax-hydrostatic",
                    "dft-relax-run", "dft-relax-check"):
        assert f"id: '{control}'" in controls or f"numberInput('{control}'" in controls
    assert "id: `dft-relax-mask-${c}`" in controls
    assert controls.count("'aria-label'") + controls.count("aria: '") >= 10
    assert "value: 'variable-cell', disabled: variableAllowed ? null : true" in controls
    result = source[source.index("function renderRelaxResult"):
                    source.index("export function relaxProvenanceView")]
    for element in ("dft-relax-table", "dft-relax-energy-plot", "dft-relax-force-plot",
                    "dft-relax-apply", "dft-relax-unconverged", "dft-relax-stale"):
        assert f"id: '{element}'" in result
    assert "r.applicable" in result and "r.convergence" in result
    api = (STATIC / "js" / "api.js").read_text()
    for route in ("dft/relax/spec", "dft/relax/run", "dft/relax/result", "dft/relax/apply"):
        assert f"'{route}'" in api
    main = (STATIC / "js" / "main.js").read_text()
    for handler in ("dftRelax:", "dftRelaxChanged:", "dftRelaxApply:", "dftRelaxShow:",
                    "dftRelaxProvenance:"):
        assert handler in main
    lowered = (source + main).lower()
    assert "assistant" not in lowered and "chatbot" not in lowered


def test_dft_dos_panel_names_its_controls_plot_selectors_and_export():
    source = (STATIC / "js" / "dft-panel.js").read_text()
    controls = source[source.index("function renderDosControls"):
                      source.index("function renderDosRuns")]
    for control in ("dft-dos-source", "dft-dos-reference", "dft-dos-emin", "dft-dos-emax",
                    "dft-dos-step", "dft-dos-broadening", "dft-dos-width", "dft-dos-spin",
                    "dft-dos-bands", "dft-dos-run", "dft-dos-check"):
        assert f"id: '{control}'" in controls or f"numberInput('{control}'" in controls
    assert "numberInput(`dft-dos-k${axis}`" in controls
    assert "id: `dft-dos-proj-${i}`" in controls
    result = source[source.index("function renderDosResult"):
                    source.index("export function dosProvenanceView")]
    for element in ("dft-dos-channel", "dft-dos-plot", "dft-dos-table", "dft-dos-export",
                    "dft-dos-provenance", "dft-dos-stale"):
        assert f"id: '{element}'" in result
    assert "id: `dft-dos-show-${i}`" in result
    assert "states / eV" in result and "E - E_F / eV" in result
    assert "standard deviation" in result and "r.checks" in result
    api = (STATIC / "js" / "api.js").read_text()
    for route in ("dft/dos/spec", "dft/dos/run", "dft/dos/result", "dft/dos/export"):
        assert f"'{route}'" in api
    main = (STATIC / "js" / "main.js").read_text()
    for handler in ("dftDos:", "dftDosChanged:", "dftDosShow:", "dftDosView:",
                    "dftDosExport:", "dftDosProvenance:"):
        assert handler in main
    assert "choosePathToSave({ title: 'Export density of states'" in main
    lowered = (source + main).lower()
    assert "assistant" not in lowered and "chatbot" not in lowered


def test_dft_band_structure_panel_names_its_controls_plot_zoom_table_and_export():
    source = (STATIC / "js" / "dft-panel.js").read_text()
    controls = source[source.index("function renderBandsControls"):
                      source.index("function renderBandsRuns")]
    for control in ("dft-bands-source", "dft-bands-reference", "dft-bands-symmetry",
                    "dft-bands-path", "dft-bands-standard", "dft-bands-run",
                    "dft-bands-check", "dft-bands-points", "dft-bands-character"):
        assert f"id: '{control}'" in controls
    for control in ("dft-bands-density", "dft-bands-bands", "dft-bands-extra"):
        assert f"numberInput('{control}'" in controls
    assert "not normalised orbital populations" in controls
    plot = source[source.index("export function bandPlot"):
                  source.index("function renderBandsResult")]
    for piece in ("distance_invA", "divider break", "'fermi'", "kLabel(tick.label)",
                  "E - E_F / eV", "-clip", "onPick", "'line down'", "'grid'"):
        assert piece in plot, piece
    result = source[source.index("function renderBandsResult"):
                    source.index("export function bandsProvenanceView")]
    for element in ("dft-bands-spin-view", "dft-bands-from", "dft-bands-to",
                    "dft-bands-zoom-gap", "dft-bands-zoom-reset", "dft-bands-kpoint-table",
                    "dft-bands-table", "dft-bands-export", "dft-bands-provenance",
                    "dft-bands-stale", "dft-bands-reason", "dft-bands-enlarge"):
        assert f"id: '{element}'" in result, element
    for element in ("dft-bands-zoom-min", "dft-bands-zoom-max"):
        assert f"numberInput('{element}'" in result
    provenance = source[source.index("export function bandsProvenanceView"):]
    provenance = provenance[:provenance.index("\nfunction ")]
    for piece in ("dft-bands-provenance-view", "Citations", "array checksums",
                  "experimental comparison", "source geometry fingerprint", "path generator"):
        assert piece in provenance, piece
    api = (STATIC / "js" / "api.js").read_text()
    for route in ("dft/bands/spec", "dft/bands/run", "dft/bands/result", "dft/bands/export"):
        assert f"'{route}'" in api
    main = (STATIC / "js" / "main.js").read_text()
    for handler in ("dftBands:", "dftBandsChanged:", "dftBandsShow:", "dftBandsView:",
                    "dftBandsExport:", "dftBandsProvenance:", "dftBandsEnlarge:"):
        assert handler in main
    assert "choosePathToSave({ title: 'Export band structure'" in main
    css = (STATIC / "css" / "app.css").read_text()
    band_css = css[css.index("svg.band-plot"):]
    band_css = band_css[:band_css.index("\n\n")] if "\n\n" in band_css else band_css
    assert "border-radius" not in band_css and "green" not in band_css
    assert "var(--accent)" in band_css
    new = controls + plot + result + provenance
    assert "\u2014" not in new and "\u2013" not in new
    assert "border-radius" not in new and "green" not in new.lower()
    lowered = new.lower()
    assert "assistant" not in lowered and "chatbot" not in lowered


def test_dft_eos_panel_names_its_controls_plots_apply_and_export():
    source = (STATIC / "js" / "dft-panel.js").read_text()
    controls = source[source.index("function renderEosControls"):
                      source.index("function renderEosRuns")]
    for control in ("dft-eos-source", "dft-eos-run", "dft-eos-check", "dft-eos-fixed"):
        assert f"id: '{control}'" in controls, control
    for control in ("dft-eos-min", "dft-eos-max", "dft-eos-points"):
        assert f"numberInput('{control}'" in controls, control
    result = source[source.index("function renderEosResult"):
                    source.index("export function eosProvenanceView")]
    for element in ("dft-eos-energy-plot", "dft-eos-pressure-plot", "dft-eos-table",
                    "dft-eos-apply", "dft-eos-export", "dft-eos-provenance", "dft-eos-stale",
                    "dft-eos-reason"):
        assert f"'{element}'" in result, element
    assert "r.applicable" in result and "Pulay" in result
    assert "zero-width energy" in result and "free_pressures_GPa" in result
    assert "E - TS" in result and "dft-eos-energy-note" in result
    provenance = source[source.index("export function eosProvenanceView"):]
    provenance = provenance[:provenance.index("\nfunction ")]
    for piece in ("dft-eos-provenance-view", "Citations", "point specifications",
                  "experimental comparison"):
        assert piece in provenance, piece
    api = (STATIC / "js" / "api.js").read_text()
    for route in ("dft/eos/spec", "dft/eos/run", "dft/eos/result", "dft/eos/export",
                  "dft/eos/apply"):
        assert f"'{route}'" in api
    main = (STATIC / "js" / "main.js").read_text()
    for handler in ("dftEos:", "dftEosChanged:", "dftEosShow:", "dftEosApply:",
                    "dftEosExport:", "dftEosProvenance:"):
        assert handler in main
    new = controls + result + provenance
    assert "—" not in new and "–" not in new and "border-radius" not in new


def test_dft_ldos_panel_names_its_controls_maps_images_and_exports():
    source = (STATIC / "js" / "dft-panel.js").read_text()
    controls = source[source.index("function renderLdosControls"):
                      source.index("function renderLdosRuns")]
    for control in ("dft-ldos-source", "dft-ldos-spin", "dft-ldos-run", "dft-ldos-check",
                    "dft-ldos-bias-filled", "dft-ldos-bias-empty"):
        assert f"id: '{control}'" in controls, control
    for control in ("dft-ldos-emin", "dft-ldos-emax", "dft-ldos-bands"):
        assert f"numberInput('{control}'" in controls, control
    result = source[source.index("function renderLdosResult"):
                    source.index("export function ldosProvenanceView")]
    for element in ("dft-ldos-profile", "dft-ldos-slice", "dft-ldos-stm-mode",
                    "dft-ldos-stm-run", "dft-ldos-stm-export", "dft-ldos-stm-image",
                    "dft-ldos-table", "dft-ldos-export", "dft-ldos-provenance",
                    "dft-ldos-stale", "dft-ldos-reason"):
        assert f"'{element}'" in result, element
    for element in ("dft-ldos-plane", "dft-ldos-stm-height", "dft-ldos-stm-iso"):
        assert f"numberInput('{element}'" in result, element
    assert "No conversion to amperes" in result and "PAW" in result
    heatmap = source[source.index("export function heatmap"):
                     source.index("function renderLdosResult")]
    assert "image.lut" in heatmap and "pixelated" in heatmap
    api = (STATIC / "js" / "api.js").read_text()
    for route in ("dft/ldos/spec", "dft/ldos/run", "dft/ldos/result", "dft/ldos/slice",
                  "dft/ldos/image", "dft/ldos/export", "dft/ldos/image/export"):
        assert f"'{route}'" in api, route
    main = (STATIC / "js" / "main.js").read_text()
    for handler in ("dftLdos:", "dftLdosChanged:", "dftLdosShow:", "dftLdosSlice:",
                    "dftLdosImage:", "dftLdosExport:", "dftLdosImageExport:",
                    "dftLdosProvenance:"):
        assert handler in main, handler
    new = controls + result + heatmap
    assert "—" not in new and "–" not in new and "border-radius" not in new


def test_solver_panel_sends_registry_keys_and_shows_potential_checks():
    panels = (STATIC / "js" / "panels.js").read_text()
    select = panels[panels.index("id: 'solver-model'"):panels.index("const row = el('div.field-grid')")]
    assert "value: d.key" in select and "value: d.name" not in select
    main = (STATIC / "js" / "main.js").read_text()
    checks = main[main.index("function potentialCheckLines"):
                  main.index("async function applySolverResult")]
    for field in ("components_eV", "ewald_message", "ewald_converged", "closest_pair_A",
                  "barrier_margin_A", "collapse barrier"):
        assert field in checks, field
    applied = main[main.index("async function applySolverResult"):]
    assert "potentialCheckLines(result.checks)" in applied
    new = select + checks
    assert "—" not in new and "–" not in new


def test_numpy_export_menu_dialog_and_route():
    menus = json.loads((STATIC.parent / "menus.json").read_text())
    assert any(item.get("action") == "export.npz" for item in menus["File"])
    main = (STATIC / "js" / "main.js").read_text()
    handler = main[main.index("case 'export.npz'"):main.index("case 'edit.undo'")]
    for part in ("'structure'", "'arrays'", "'results'", "'scans'", "allow_pickle=False",
                 "api.exportNpz", "choosePathToSave"):
        assert part in handler, part
    assert "exportNpz: (path, parts) => call('export/npz'" in (STATIC / "js" / "api.js").read_text()
    assert "—" not in handler and "–" not in handler
