"""Material definitions, schema validation and the registry."""

from __future__ import annotations

import json

import pytest

from materia.materials import MaterialLibrary, MaterialValidationError, default_library, validate
from materia.materials.loader import BUILTIN_DIR

EXPECTED = {
    "silicon", "germanium", "silicon_carbide_4h", "silicon_carbide_3c",
    "gallium_arsenide", "gallium_nitride", "indium_phosphide", "sapphire",
    "diamond", "graphene", "hexagonal_boron_nitride", "molybdenum_disulfide",
    "tungsten_diselenide", "silicon_dioxide", "copper", "gold", "silver",
    "platinum", "nickel", "titanium", "tungsten",
}


def test_every_shipped_definition_loads_without_error(library):
    assert library.errors() == {}
    assert EXPECTED.issubset(set(library.ids()))


def test_lookup_by_id_alias_formula_and_name(library):
    for key in ("silicon", "Si", "c-Si", "Silicon"):
        assert library.get(key).id == "silicon"
    assert library.get("h-BN").id == "hexagonal_boron_nitride"
    assert library.get("corundum").id == "sapphire"


def test_unknown_material_lists_the_known_ones(library):
    with pytest.raises(KeyError) as excinfo:
        library.get("unobtainium")
    assert "silicon" in str(excinfo.value)


@pytest.mark.parametrize("material_id", sorted(EXPECTED))
def test_definitions_are_internally_consistent(library, material_id):
    definition = library.get(material_id)
    assert definition.basis
    a, b, c, alpha, beta, gamma = definition.lattice.parameters()
    assert a > 0 and b > 0 and c > 0
    assert 0 < alpha < 180 and 0 < beta < 180 and 0 < gamma < 180
    for site in definition.basis:
        assert all(-1e-9 <= v < 1.0 + 1e-9 for v in site.fractional), site.label
    assert definition.license
    assert definition.provenance_note


DIMENSIONLESS = {"dielectric_constant"}


@pytest.mark.parametrize("material_id", sorted(EXPECTED))
def test_properties_carry_units_and_sources(library, material_id):
    definition = library.get(material_id)
    for key, prop in definition.properties.items():
        assert prop.source, f"{material_id}.{key} has no source"
        if isinstance(prop.value, (int, float)) and key not in DIMENSIONLESS:
            assert prop.unit, f"{material_id}.{key} is numeric but has no unit"


def test_band_gaps_are_physically_ordered(library):
    gaps = {m: library.get(m).properties["band_gap"].value
            for m in ("gold", "silicon", "gallium_arsenide", "silicon_carbide_4h",
                      "diamond", "sapphire")}
    assert gaps["gold"] == 0.0
    assert gaps["silicon"] < gaps["gallium_arsenide"] < gaps["silicon_carbide_4h"]
    assert gaps["silicon_carbide_4h"] < gaps["diamond"] < gaps["sapphire"]


def test_unimplemented_reconstructions_are_declared(library):
    silicon = library.get("silicon")
    seven = next(r for r in silicon.reconstructions if r.id == "7x7-DAS")
    assert seven.implemented is False
    assert "Not implemented" in seven.note
    assert "Takayanagi" in seven.reference

    gold = library.get("gold")
    herringbone = next(r for r in gold.reconstructions if "herringbone" in r.id)
    assert herringbone.implemented is False


def test_schema_rejects_a_missing_required_field():
    with pytest.raises(MaterialValidationError) as excinfo:
        validate({"schema_version": "1.0", "id": "x", "name": "X"})
    assert "formula" in str(excinfo.value)


def test_schema_rejects_an_incompatible_version():
    raw = json.loads((BUILTIN_DIR / "silicon.json").read_text())
    raw["schema_version"] = "9.0"
    with pytest.raises(MaterialValidationError) as excinfo:
        validate(raw)
    assert "incompatible" in str(excinfo.value)


def test_schema_rejects_unknown_elements_and_wrong_units():
    raw = json.loads((BUILTIN_DIR / "silicon.json").read_text())
    raw["structure"]["basis"][0]["element"] = "Xx"
    with pytest.raises(MaterialValidationError) as excinfo:
        validate(raw)
    assert "unknown element" in str(excinfo.value)

    raw = json.loads((BUILTIN_DIR / "silicon.json").read_text())
    raw["structure"]["lattice"]["unit"] = "nm"
    with pytest.raises(MaterialValidationError) as excinfo:
        validate(raw)
    assert "angstrom" in str(excinfo.value)


def test_user_definitions_take_precedence(tmp_path):
    raw = json.loads((BUILTIN_DIR / "silicon.json").read_text())
    raw["name"] = "Silicon (user override)"
    (tmp_path / "silicon.json").write_text(json.dumps(raw))
    library = MaterialLibrary([tmp_path])
    assert library.get("silicon").name == "Silicon (user override)"


def test_malformed_files_are_reported_not_raised(tmp_path):
    (tmp_path / "broken.json").write_text("{ this is not json")
    (tmp_path / "invalid.json").write_text(json.dumps({"schema_version": "1.0"}))
    library = MaterialLibrary([tmp_path])
    library.scan()
    errors = library.errors()
    assert len(errors) == 2
    assert any("unreadable JSON" in message for message in errors.values())
    assert "silicon" in library.ids()


def test_search_matches_formula_and_category(library):
    assert any(d.id == "graphene" for d in library.search("two-dimensional"))
    assert any(d.id == "gold" for d in library.search("Au"))


def test_no_material_claims_a_reconstruction_the_build_cannot_generate(library):
    """`implemented: true` in a file is a claim about this build, not about the
    literature.  A definition that makes it without a registered generator would
    put an "available" badge in front of a user for something that does not
    exist."""
    from materia.structure_builder.reconstruction import generator_info

    offenders = []
    for definition in library.all():
        for declaration in definition.reconstructions:
            if not declaration.implemented:
                continue
            info = generator_info(declaration.id)
            if info is None or definition.prototype not in info.prototypes:
                offenders.append(f"{definition.id}:{declaration.id}")
    assert offenders == []


def test_shipped_reconstructions_cite_reference_geometry(library):
    """A reconstruction Materia ships and can generate must be comparable against
    measurement, so it carries published structural parameters with a citation.

    Only the shipped library is held to this. A plugin may legitimately generate
    a reconstruction for a system nobody has measured, in which case Materia
    reports the computed geometry with no comparison rather than borrowing an
    inapplicable one.
    """
    from materia.structure_builder.reconstruction import available_reconstructions

    for definition in library.all():
        if "materia/materials/library" not in definition.source_path.replace("\\", "/"):
            continue
        for entry in available_reconstructions(definition):
            if not entry["supported"]:
                continue
            geometry = entry["reference_geometry"]
            assert geometry, f"{definition.id}:{entry['id']} has no reference geometry"
            for quantity, value in geometry.items():
                assert value["source"], f"{definition.id}:{entry['id']}:{quantity}"
                assert value["value"] is not None
