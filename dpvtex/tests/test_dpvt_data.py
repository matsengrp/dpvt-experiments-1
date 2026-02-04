import pytest
import json
from pathlib import Path
from dpvtex.dpvt_data import load_nicknames_dict, _extract_prefix


class TestExtractPrefix:
    def test_extract_prefix_simple(self):
        assert _extract_prefix("random_*") == "random_"

    def test_extract_prefix_complex(self):
        assert _extract_prefix("random_PF*_tree_search") == "random_PF"

    def test_extract_prefix_no_glob(self):
        assert _extract_prefix("no_glob_here") == "no_glob_here"

    def test_extract_prefix_empty(self):
        assert _extract_prefix("*") == ""

    def test_extract_prefix_with_colon_separator(self):
        assert (
            _extract_prefix("random_treesearch_dataset_:alignment")
            == "random_treesearch_dataset_"
        )

    def test_extract_prefix_colon_takes_precedence(self):
        assert _extract_prefix("prefix_:id*pattern") == "prefix_"


class TestLoadNicknamesDictWithGlob:
    def test_backward_compatibility(self, tmp_path):
        """Non-pattern entries should work as before."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "test.p").touch()

        nicknames = {"data_dir": str(data_dir), "test_data": "test.p"}
        json_path = tmp_path / "nicknames.json"
        json_path.write_text(json.dumps(nicknames))

        result = load_nicknames_dict(str(json_path))
        assert "test_data" in result
        assert result["test_data"].endswith("test.p")

    def test_glob_expansion(self, tmp_path):
        """Glob patterns should expand to multiple entries."""
        data_dir = tmp_path / "data"
        subdir = data_dir / "subdir"
        subdir.mkdir(parents=True)
        (subdir / "file1.p").touch()
        (subdir / "file2.p").touch()

        nicknames = {"data_dir": str(data_dir), "prefix_": "subdir/*.p"}
        json_path = tmp_path / "nicknames.json"
        json_path.write_text(json.dumps(nicknames))

        result = load_nicknames_dict(str(json_path))
        assert "prefix_file1" in result
        assert "prefix_file2" in result
        assert len(result) == 2

    def test_recursive_glob(self, tmp_path):
        """Recursive glob (**) should work."""
        data_dir = tmp_path / "data"
        deep_dir = data_dir / "a" / "b"
        deep_dir.mkdir(parents=True)
        (deep_dir / "deep.p").touch()

        nicknames = {"data_dir": str(data_dir), "deep_": "**/*.p"}
        json_path = tmp_path / "nicknames.json"
        json_path.write_text(json.dumps(nicknames))

        result = load_nicknames_dict(str(json_path))
        assert "deep_deep" in result

    def test_mixed_patterns_and_explicit(self, tmp_path):
        """Mix of glob patterns and explicit entries should work."""
        data_dir = tmp_path / "data"
        subdir = data_dir / "subdir"
        subdir.mkdir(parents=True)
        (data_dir / "explicit.p").touch()
        (subdir / "rep1.p").touch()
        (subdir / "rep2.p").touch()

        nicknames = {
            "data_dir": str(data_dir),
            "my_explicit_data": "explicit.p",
            "replicate_": "subdir/rep*.p",
        }
        json_path = tmp_path / "nicknames.json"
        json_path.write_text(json.dumps(nicknames))

        result = load_nicknames_dict(str(json_path))
        assert "my_explicit_data" in result
        assert "replicate_rep1" in result
        assert "replicate_rep2" in result
        assert len(result) == 3
