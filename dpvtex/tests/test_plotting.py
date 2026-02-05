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
    get_evolution_model,
    get_data_source,
    get_dataset_display_name,
    truncate_to_significant_digits,
    format_number,
    plt_subplots,
    build_performance_heatmap,
    plot_hyperparameters_summary,
    LabelConfig,
    METRIC_LABELS,
    MODEL_NAMES,
    DATASET_NAMES,
)


# =============================================================================
# Dataset Name Parsing Tests
# =============================================================================


class TestExtractNumLeaves:
    """Tests for extract_num_leaves function."""

    def test_alisim_dataset(self):
        assert extract_num_leaves("alisim_50_sites_1000_trees") == "50"

    def test_leaf_in_name(self):
        assert extract_num_leaves("100leaf_dataset") == "100"

    def test_perfect_phylogeny(self):
        assert extract_num_leaves("perfect_25_trees_100") == "25"

    def test_unrecognized_format(self):
        assert extract_num_leaves("unknown_dataset") is None


class TestExtractNumSites:
    """Tests for extract_num_sites function."""

    def test_alisim_dataset(self):
        assert extract_num_sites("alisim_50_sites_1000_trees") == "1000"

    def test_perfect_phylogeny(self):
        # Format: perfect_{leaves}_{trees}_{more}_{more2}_{more3}_{more4}_{sites}
        assert extract_num_sites("perfect_25_100_more_500_stuff_sites_xyz") == "xyz"

    def test_unrecognized_format(self):
        assert extract_num_sites("unknown_dataset") is None


class TestExtractNumTrees:
    """Tests for extract_num_trees function."""

    def test_pp_dataset(self):
        # Format: dataset_100_pp_trees => splits on "trees", then takes last "_" part
        assert extract_num_trees("dataset_pp_100trees_spr") == "100"

    def test_perfect_phylogeny(self):
        assert extract_num_trees("perfect_25_trees_100") == "100"

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


class TestGetEvolutionModel:
    """Tests for get_evolution_model function."""

    def test_gtr_model(self):
        assert get_evolution_model("alisim_50_GTR_1000") == "GTR"

    def test_hky_model(self):
        assert get_evolution_model("alisim_50_hky_1000") == "HKY"

    def test_jc_model(self):
        assert get_evolution_model("alisim_50_1000") == "JC"

    def test_non_alisim(self):
        assert get_evolution_model("flu_dataset") == ""


class TestGetDataSource:
    """Tests for get_data_source function."""

    def test_alisim_dataset(self):
        assert get_data_source("alisim_50_sites_1000") == "alisim"

    def test_flu_dataset(self):
        assert get_data_source("fluC_NS_dataset") == "fluC"

    def test_rotavirus_dataset(self):
        assert get_data_source("rotavirus_A_H") == "rotavirus"


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
