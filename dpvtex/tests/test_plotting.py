"""Tests for the dpvtex.plotting module."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import tempfile

from dpvtex.plotting import (
    extract_num_leaves,
    extract_num_sites,
    extract_num_trees,
    extract_nonmp_fraction,
    get_dataset_display_name,
    truncate_to_significant_digits,
    format_number,
    plt_subplots,
    build_performance_heatmap,
    plot_hyperparameters_summary,
    plot_precision_recall_curves,
    LabelConfig,
    METRIC_LABELS,
    MODEL_NAMES,
    DATASET_NAMES,
    _build_labels_from_tuples,
    _build_x_labels,
    _find_pr_curve_pdf,
    _find_pr_curve_csv,
)


# =============================================================================
# Dataset Name Parsing Tests
# =============================================================================


class TestExtractNumLeaves:
    """Tests for extract_num_leaves function."""

    def test_simulated_dataset(self):
        assert extract_num_leaves("simulated_15_seq_20_sites_100_algnmnts") == "15"

    def test_simulated_with_suffix(self):
        assert (
            extract_num_leaves(
                "simulated_50_seq_200_sites_50_algnmnts_filtered_0.8_spr"
            )
            == "50"
        )

    def test_simulated_with_nonmp_fraction(self):
        assert (
            extract_num_leaves(
                "simulated_15_seq_20_sites_100_algnmnts_filtered_0.8_spr_r2_t0.1"
            )
            == "15"
        )

    def test_unrecognized_format(self):
        assert extract_num_leaves("unknown_dataset") is None


class TestExtractNumSites:
    """Tests for extract_num_sites function."""

    def test_simulated_dataset(self):
        assert extract_num_sites("simulated_15_seq_20_sites_100_algnmnts") == "20"

    def test_simulated_with_suffix(self):
        assert (
            extract_num_sites("simulated_50_seq_200_sites_50_algnmnts_filtered_0.8_spr")
            == "200"
        )

    def test_unrecognized_format(self):
        assert extract_num_sites("unknown_dataset") is None


class TestExtractNumTrees:
    """Tests for extract_num_trees function."""

    def test_simulated_dataset(self):
        assert extract_num_trees("simulated_15_seq_20_sites_100_algnmnts") == "100"

    def test_simulated_with_suffix(self):
        assert (
            extract_num_trees(
                "simulated_50_seq_200_sites_50_algnmnts_filtered_0.8_subtree"
            )
            == "50"
        )

    def test_unrecognized_format(self):
        assert extract_num_trees("unknown_dataset") is None


class TestExtractNonmpFraction:
    """Tests for extract_nonmp_fraction function."""

    def test_fraction_with_pickle_extension(self):
        assert extract_nonmp_fraction("dataset_t0.1.p") == 0.1

    def test_fraction_without_extension(self):
        assert extract_nonmp_fraction("dataset_t0.25") == 0.25

    def test_integer_fraction(self):
        assert extract_nonmp_fraction("dataset_t1.p") == 1.0

    def test_small_fraction(self):
        assert extract_nonmp_fraction("some_data_t0.05.p") == 0.05

    def test_no_fraction(self):
        assert extract_nonmp_fraction("dataset.p") is None

    def test_no_fraction_no_extension(self):
        assert extract_nonmp_fraction("plain_dataset") is None


class TestGetDatasetDisplayName:
    """Tests for get_dataset_display_name function."""

    def test_simulated_dataset(self):
        assert get_dataset_display_name("simulated_50_sites") == "sim"

    def test_alisim_dataset(self):
        assert get_dataset_display_name("alisim_50_1000") == "sim"

    def test_rotavirus_dataset(self):
        assert get_dataset_display_name("rotavirus_A_H_H2") == "rota A H H2"

    def test_orthomam_dataset(self):
        assert get_dataset_display_name("orthomam_train") == "OrthoMaM"

    def test_pandit_dataset(self):
        assert get_dataset_display_name("pandit_test") == "PANDIT"

    def test_with_stats(self):
        result = get_dataset_display_name(
            "simulated_data", num_leaves=50, num_sites=1000, num_trees=100
        )
        assert "sim" in result
        assert "n=50" in result
        assert "N=1000" in result
        assert "T=100" in result

    def test_with_partial_stats(self):
        result = get_dataset_display_name(
            "orthomam_train", num_leaves=100, num_sites=500
        )
        assert "OrthoMaM" in result
        assert "n=100" in result
        assert "N=500" in result


# =============================================================================
# Formatting Utility Tests
# =============================================================================


class TestTruncateToSignificantDigits:
    """Tests for truncate_to_significant_digits function."""

    def test_zero(self):
        assert truncate_to_significant_digits(0, 4) == 0

    def test_positive_number(self):
        result = truncate_to_significant_digits(12345.6789, 4)
        assert result == 12340.0

    def test_small_number(self):
        result = truncate_to_significant_digits(0.0012345, 3)
        assert result == 0.00123

    def test_negative_number(self):
        result = truncate_to_significant_digits(-12345.6789, 4)
        # The function uses floor, so for negative numbers it truncates toward more negative
        assert abs(result) >= 12340.0


class TestFormatNumber:
    """Tests for format_number function."""

    def test_normal_number(self):
        result = format_number(1234.5678, sig_digits=4)
        assert result == 1234.0

    def test_small_number_scientific(self):
        result = format_number(0.0000001234, sig_digits=4)
        assert "e" in str(result)

    def test_large_number_scientific(self):
        result = format_number(12345678901.0, sig_digits=4)
        assert "e" in str(result)

    def test_zero(self):
        result = format_number(0, sig_digits=4)
        assert result == 0


class TestPltSubplots:
    """Tests for plt_subplots helper function."""

    def test_single_subplot_returns_array(self):
        import matplotlib.pyplot as plt

        fig, axs = plt_subplots()
        assert isinstance(axs, np.ndarray)
        assert len(axs) == 1
        plt.close(fig)

    def test_multiple_subplots_returns_array(self):
        import matplotlib.pyplot as plt

        fig, axs = plt_subplots(1, 3)
        assert isinstance(axs, np.ndarray)
        assert len(axs) == 3
        plt.close(fig)


# =============================================================================
# Constants Tests
# =============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_metric_labels_keys(self):
        assert "test_auroc" in METRIC_LABELS
        assert "test_accuracy" in METRIC_LABELS
        assert "test_loss" in METRIC_LABELS

    def test_model_names_keys(self):
        assert "TraverseMaxPooling" in MODEL_NAMES
        assert "TraverseAvgPooling" in MODEL_NAMES
        assert "TraverseNN" in MODEL_NAMES
        assert "BaselineReversion" in MODEL_NAMES

    def test_dataset_names_keys(self):
        assert "orthomam" in DATASET_NAMES
        assert "pandit" in DATASET_NAMES
        assert "rotavirus" in DATASET_NAMES


# =============================================================================
# Integration Tests
# =============================================================================


class TestBuildPerformanceHeatmap:
    """Integration tests for build_performance_heatmap function."""

    @pytest.fixture
    def sample_dataframe(self):
        """Create a minimal sample DataFrame for testing."""
        return pd.DataFrame(
            {
                "model": [
                    "TraverseMaxPooling",
                    "TraverseMaxPooling",
                    "TraverseAvgPooling",
                    "TraverseAvgPooling",
                ],
                "train_data": ["train_set1", "train_set1", "train_set1", "train_set1"],
                "test_data": ["test_set1", "test_set2", "test_set1", "test_set2"],
                "train_num_leaves": [50, 50, 50, 50],
                "train_num_sites": [1000, 1000, 1000, 1000],
                "train_num_trees": [100, 100, 100, 100],
                "test_num_leaves": [50, 60, 50, 60],
                "test_num_sites": [1000, 1200, 1000, 1200],
                "test_num_trees": [50, 50, 50, 50],
                "test_auroc": [0.85, 0.82, 0.83, 0.80],
            }
        )

    def test_heatmap_creates_file(self, sample_dataframe):
        """Test that heatmap creates an output file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_heatmap.pdf"
            build_performance_heatmap(
                df=sample_dataframe,
                value_column="test_auroc",
                output_path=str(output_path),
                title="Test AUROC",
            )
            assert output_path.exists()

    def test_heatmap_with_mixed_source(self, sample_dataframe):
        """Test heatmap with mixed source flags via LabelConfig."""
        sample_dataframe = sample_dataframe.copy()
        sample_dataframe["train_data"] = [
            "spr_train",
            "spr_train",
            "subtree_train",
            "subtree_train",
        ]
        sample_dataframe["test_data"] = [
            "spr_test",
            "subtree_test",
            "spr_test",
            "subtree_test",
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_heatmap_mixed.pdf"
            # Use LabelConfig to explicitly show perturbation labels
            label_config = LabelConfig(show_perturbation=True)
            build_performance_heatmap(
                df=sample_dataframe,
                value_column="test_auroc",
                output_path=str(output_path),
                label_config=label_config,
            )
            assert output_path.exists()

    def test_heatmap_with_nonmp_fraction(self, sample_dataframe):
        """Test heatmap distinguishes datasets with and without non-MP fraction."""
        sample_dataframe = sample_dataframe.copy()
        sample_dataframe["train_data"] = [
            "sim_filtered_0.8_spr",
            "sim_filtered_0.8_spr",
            "sim_filtered_0.8_spr_r2_t0.1",
            "sim_filtered_0.8_spr_r2_t0.1",
        ]
        sample_dataframe["test_data"] = [
            "sim_filtered_0.8_spr",
            "sim_filtered_0.8_spr_r2_t0.1",
            "sim_filtered_0.8_spr",
            "sim_filtered_0.8_spr_r2_t0.1",
        ]
        sample_dataframe["train_nonmp_fraction"] = [None, None, 0.1, 0.1]
        sample_dataframe["test_nonmp_fraction"] = [None, 0.1, None, 0.1]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_heatmap_nonmp.pdf"
            build_performance_heatmap(
                df=sample_dataframe,
                value_column="test_auroc",
                output_path=str(output_path),
            )
            assert output_path.exists()


class TestPlotHyperparametersSummary:
    """Integration tests for plot_hyperparameters_summary function."""

    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample DataFrame with hyperparameters."""
        return pd.DataFrame(
            {
                "model_and_train_data": [
                    "MaxPool-train1",
                    "AvgPool-train1",
                    "MaxPool-train2",
                ],
                "learning_rate": [0.001, 0.001, 0.0001],
                "batch_size": [32, 64, 32],
                "accum_grad_batches": [1, 2, 1],
                "max_epochs": [100, 100, 200],
                "feature_length": [128, 128, 256],
                "dim_mlp_layers": [64, 64, 128],
            }
        )

    def test_hyperparameters_creates_file(self, sample_dataframe):
        """Test that hyperparameters plot creates an output file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "hyperparams.pdf"
            plot_hyperparameters_summary(sample_dataframe, str(output_path))
            assert output_path.exists()

    def test_skips_when_too_many_models(self, sample_dataframe):
        """Test that plotting is skipped when there are too many models."""
        # Create DataFrame with >10 unique model_and_train_data entries
        many_models_df = pd.DataFrame(
            {
                "model_and_train_data": [f"Model{i}-train{i}" for i in range(12)],
                "learning_rate": [0.001] * 12,
                "batch_size": [32] * 12,
                "accum_grad_batches": [1] * 12,
                "max_epochs": [100] * 12,
                "feature_length": [128] * 12,
                "dim_mlp_layers": [64] * 12,
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "hyperparams.pdf"
            plot_hyperparameters_summary(many_models_df, str(output_path))
            # File should not be created when too many models
            assert not output_path.exists()


# =============================================================================
# Label Builder Tests
# =============================================================================


class TestBuildLabelsFromTuples:
    def test_scalar_items(self):
        assert _build_labels_from_tuples([10, 20], ["n="]) == ["n=10", "n=20"]

    def test_tuple_items(self):
        assert _build_labels_from_tuples([(10, 20)], ["n=", "N="]) == ["n=10\nN=20"]

    def test_empty_prefix_shows_value_as_is(self):
        assert _build_labels_from_tuples(["foo", "bar"], [""]) == ["foo", "bar"]

    def test_start_offset(self):
        assert _build_labels_from_tuples([(0, 10, 20)], ["n=", "N="], start_offset=1) == [
            "n=10\nN=20"
        ]

    def test_out_of_bounds_index_gives_empty_string(self):
        result = _build_labels_from_tuples([(10,)], ["n=", "N="])
        assert result == ["n=10\nN="]

    def test_empty_items(self):
        assert _build_labels_from_tuples([], ["n="]) == []


class TestBuildXLabels:
    def _flags(self, mixed_testing=False, test_sites=False, test_nonmp=False):
        return {
            "mixed_testing": mixed_testing,
            "test_sites": test_sites,
            "test_nonmp": test_nonmp,
            "mixed_training": False,
            "mixed_train_sources": False,
            "train_leaves": False,
            "train_sites": False,
            "train_trees": False,
            "train_nonmp": False,
        }

    def test_display_name_columns_returned_as_is(self):
        idx = pd.Index(["Dataset A", "Dataset B"], name="test_data_name")
        heatmap_data = pd.DataFrame([[1, 2]], columns=idx)
        result = _build_x_labels(heatmap_data, self._flags())
        assert result == ["Dataset A", "Dataset B"]

    def test_single_leaf_column(self):
        idx = pd.Index([15, 50], name="test_num_leaves")
        heatmap_data = pd.DataFrame([[1, 2]], columns=idx)
        result = _build_x_labels(heatmap_data, self._flags())
        assert result == ["n=15", "n=50"]

    def test_mixed_testing_prepends_empty_prefix(self):
        idx = pd.MultiIndex.from_tuples([("spr", 15), ("spr", 50)],
                                         names=["perturbation", "test_num_leaves"])
        heatmap_data = pd.DataFrame([[1, 2]], columns=idx)
        result = _build_x_labels(heatmap_data, self._flags(mixed_testing=True))
        assert result == ["spr\nn=15", "spr\nn=50"]

    def test_sites_flag_adds_N_prefix(self):
        idx = pd.MultiIndex.from_tuples([(15, 100)],
                                         names=["test_num_leaves", "test_num_sites"])
        heatmap_data = pd.DataFrame([[1]], columns=idx)
        result = _build_x_labels(heatmap_data, self._flags(test_sites=True))
        assert result == ["n=15\nN=100"]


# =============================================================================
# PR Curve File Finder Tests
# =============================================================================


class TestFindPrCurve:
    def test_none_input_returns_none(self):
        assert _find_pr_curve_pdf(None) is None
        assert _find_pr_curve_csv(None) is None

    def test_sentinel_string_returns_none(self):
        assert _find_pr_curve_pdf("none") is None
        assert _find_pr_curve_csv("none") is None

    def test_directory_with_no_files_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "version_0").mkdir()
            assert _find_pr_curve_pdf(tmpdir) is None
            assert _find_pr_curve_csv(tmpdir) is None

    def test_finds_file_in_version_subdir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            version_dir = Path(tmpdir) / "version_0"
            version_dir.mkdir()
            pdf = version_dir / "pr_curve.pdf"
            csv = version_dir / "pr_curve.csv"
            pdf.write_text("")
            csv.write_text("")
            assert _find_pr_curve_pdf(tmpdir) == pdf
            assert _find_pr_curve_csv(tmpdir) == csv

    def test_returns_last_version_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for v in ["version_0", "version_1", "version_2"]:
                d = Path(tmpdir) / v
                d.mkdir()
                (d / "pr_curve.csv").write_text("")
            result = _find_pr_curve_csv(tmpdir)
            assert result == Path(tmpdir) / "version_2" / "pr_curve.csv"


# =============================================================================
# plot_precision_recall_curves Tests
# =============================================================================


class TestPlotPrecisionRecallCurves:
    def _make_pr_df(self, ap=0.8):
        return pd.DataFrame(
            {
                "recall": [0.0, 0.5, 1.0],
                "precision": [1.0, 0.8, 0.5],
                "avg_precision": [ap, ap, ap],
            }
        )

    def test_creates_output_file(self):
        grid = {("n=15", "MaxPool"): self._make_pr_df()}
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "pr.pdf"
            plot_precision_recall_curves(grid, ["n=15"], ["MaxPool"], str(out))
            assert out.exists()

    def test_missing_grid_entry_leaves_blank_panel(self):
        grid = {("n=15", "MaxPool"): self._make_pr_df()}
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "pr.pdf"
            plot_precision_recall_curves(
                grid, ["n=15", "n=50"], ["MaxPool", "AvgPool"], str(out)
            )
            assert out.exists()

    def test_empty_row_labels_skips_plot(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "pr.pdf"
            plot_precision_recall_curves({}, [], ["MaxPool"], str(out))
            assert not out.exists()

    def test_empty_col_labels_skips_plot(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "pr.pdf"
            plot_precision_recall_curves({}, ["n=15"], [], str(out))
            assert not out.exists()
