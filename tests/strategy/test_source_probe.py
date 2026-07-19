"""Source-level probe that gates the plain-edit float rung (CALIBRATION.md §Bug 4)."""

from agents.strategy.source_probe import region_has_bare_double


def _mk(tmp_path, text, name="B2m.h"):
    (tmp_path / name).write_text(text)
    return name


def test_bare_double_line_is_applicable(tmp_path):
    name = _mk(tmp_path, "line1\n    double x = a * b;\nline3\n")
    assert region_has_bare_double(str(tmp_path), name, 2, 2) is True


def test_template_typed_line_is_inapplicable(tmp_path):
    # `T` template kernel: no bare `double` token on the region line.
    name = _mk(tmp_path, "line1\n    T x = a * b;\nline3\n")
    assert region_has_bare_double(str(tmp_path), name, 2, 2) is False


def test_double_only_as_identifier_substring_is_inapplicable(tmp_path):
    # `redouble` / `double_t` contain the letters but no bare `double` keyword.
    name = _mk(tmp_path, "    redouble_t x;  // double_buffer\n")
    assert region_has_bare_double(str(tmp_path), name, 1, 1) is False


def test_basename_resolution_in_subdir(tmp_path):
    box = tmp_path / "box"
    box.mkdir()
    (box / "B2m.h").write_text("l1\n  double y;\n")
    # bare basename resolves to box/B2m.h
    assert region_has_bare_double(str(tmp_path), "B2m.h", 2, 2) is True


def test_missing_repo_or_file_defaults_applicable(tmp_path):
    # no repo → don't gate (Patcher's patch_inapplicable is the net)
    assert region_has_bare_double(None, "B2m.h", 1, 1) is True
    # file not found → don't gate
    assert region_has_bare_double(str(tmp_path), "nope.h", 1, 1) is True


def test_cache_reads_file_once(tmp_path):
    name = _mk(tmp_path, "l1\n  double y;\n  T z;\n")
    cache: dict = {}
    assert region_has_bare_double(str(tmp_path), name, 2, 2, cache=cache) is True
    assert region_has_bare_double(str(tmp_path), name, 3, 3, cache=cache) is False
    assert len(cache) == 1          # both probes hit the same cached file
