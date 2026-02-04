"""Tests for analyze_edge_distributions.py utility functions."""

import re
import pytest
from dpvtex.larch.scripts.analyze_edge_distributions import (
    get_pickle_pattern,
    extract_method_label_from_filename,
)


class TestGetPicklePattern:
    """Tests for get_pickle_pattern function."""

    def test_constant_method_matches_old_style(self):
        """Test that 'constant' pattern matches old-style _spr.p suffix."""
        pattern = get_pickle_pattern("constant")
        assert re.search(pattern, "dataset_spr.p")
        assert re.search(pattern, "simulated_15_seq_20_sites_100_algnmnts_spr.p")

    def test_constant_method_matches_new_style_with_params(self):
        """Test that 'constant' pattern matches new-style with radius and target."""
        pattern = get_pickle_pattern("constant")
        assert re.search(pattern, "dataset_spr_r2_t0.1.p")
        assert re.search(pattern, "simulated_15_seq_20_sites_10_algnmnts_spr_r2_t0.1.p")
        assert re.search(pattern, "dataset_spr_r3_t0.167.p")

    def test_constant_method_rejects_other_methods(self):
        """Test that 'constant' pattern doesn't match other method suffixes."""
        pattern = get_pickle_pattern("constant")
        assert not re.search(pattern, "dataset_subtree.p")
        assert not re.search(pattern, "dataset_uniform.p")
        assert not re.search(pattern, "dataset_treesearch_mimic.p")

    def test_random_subtree_matches_old_style(self):
        """Test that 'random_subtree' pattern matches old-style _subtree.p suffix."""
        pattern = get_pickle_pattern("random_subtree")
        assert re.search(pattern, "dataset_subtree.p")
        assert re.search(pattern, "simulated_15_seq_20_sites_50_algnmnts_subtree.p")

    def test_random_subtree_matches_new_style_with_target(self):
        """Test that 'random_subtree' pattern matches new-style with target param."""
        pattern = get_pickle_pattern("random_subtree")
        assert re.search(pattern, "dataset_subtree_t0.1.p")
        assert re.search(
            pattern, "simulated_15_seq_20_sites_10_algnmnts_subtree_t0.167.p"
        )

    def test_random_subtree_rejects_other_methods(self):
        """Test that 'random_subtree' pattern doesn't match other method suffixes."""
        pattern = get_pickle_pattern("random_subtree")
        assert not re.search(pattern, "dataset_spr.p")
        assert not re.search(pattern, "dataset_spr_r2_t0.1.p")

    def test_mixed_method_matches_specific_suffix(self):
        """Test that 'mixed' pattern matches the specific mixed suffix."""
        pattern = get_pickle_pattern("mixed")
        assert re.search(pattern, "dataset_spr_subtree_few_sprs.p")
        assert not re.search(pattern, "dataset_spr.p")
        assert not re.search(pattern, "dataset_subtree.p")

    def test_uniform_method_patterns(self):
        """Test that 'uniform' pattern matches both old and new styles."""
        pattern = get_pickle_pattern("uniform")
        assert re.search(pattern, "dataset_uniform.p")
        assert re.search(pattern, "dataset_uniform_r2_t0.1.p")

    def test_treesearch_mimic_method_patterns(self):
        """Test that 'treesearch_mimic' pattern matches both old and new styles."""
        pattern = get_pickle_pattern("treesearch_mimic")
        assert re.search(pattern, "dataset_treesearch_mimic.p")
        assert re.search(pattern, "dataset_treesearch_mimic_r3_t0.2.p")

    def test_unknown_method_returns_generic_pattern(self):
        """Test that unknown method returns generic .p pattern."""
        pattern = get_pickle_pattern("unknown_method")
        assert re.search(pattern, "anything.p")


class TestExtractMethodLabelFromFilename:
    """Tests for extract_method_label_from_filename function."""

    def test_spr_with_radius_and_target(self):
        """Test extraction of SPR parameters (radius and target)."""
        label = extract_method_label_from_filename(
            "simulated_15_seq_20_sites_10_algnmnts_spr_r2_t0.1.p", "constant"
        )
        assert label == "SPR (r=2, t=0.1)"

    def test_spr_with_decimal_target(self):
        """Test extraction with multi-digit decimal target."""
        label = extract_method_label_from_filename(
            "dataset_spr_r3_t0.167.p", "constant"
        )
        assert label == "SPR (r=3, t=0.167)"

    def test_spr_old_style_fallback(self):
        """Test fallback to base name for old-style SPR suffix."""
        label = extract_method_label_from_filename("dataset_spr.p", "constant")
        assert label == "SPR"

    def test_subtree_with_target(self):
        """Test extraction of subtree target parameter."""
        label = extract_method_label_from_filename(
            "simulated_15_seq_20_sites_10_algnmnts_subtree_t0.167.p", "random_subtree"
        )
        assert label == "random subtree (t=0.167)"

    def test_subtree_old_style_fallback(self):
        """Test fallback to base name for old-style subtree suffix."""
        label = extract_method_label_from_filename(
            "dataset_subtree.p", "random_subtree"
        )
        assert label == "random subtree"

    def test_uniform_with_params(self):
        """Test extraction of uniform method parameters."""
        label = extract_method_label_from_filename(
            "dataset_uniform_r2_t0.1.p", "uniform"
        )
        assert label == "uniform (r=2, t=0.1)"

    def test_treesearch_mimic_with_params(self):
        """Test extraction of treesearch_mimic method parameters."""
        label = extract_method_label_from_filename(
            "dataset_treesearch_mimic_r3_t0.2.p", "treesearch_mimic"
        )
        assert label == "treesearch_mimic (r=3, t=0.2)"

    def test_full_path_extracts_basename(self):
        """Test that full paths are handled correctly (basename extracted)."""
        label = extract_method_label_from_filename(
            "/some/long/path/to/data/dataset_spr_r2_t0.1.p", "constant"
        )
        assert label == "SPR (r=2, t=0.1)"

    def test_unknown_method_uses_method_name(self):
        """Test that unknown methods use the method name as base."""
        label = extract_method_label_from_filename("dataset_custom.p", "custom_method")
        assert label == "custom_method"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
