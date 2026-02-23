"""Tests for quantify_labeling_problem and plot_labeling_problem scripts."""

import os
import tempfile

import pandas as pd
import pytest

from dpvtex.log_tree_searches.plot_labeling_problem import (
    _filter_last_tree_per_replicate as plot_filter,
    plot_labeling_problem,
    plot_labeling_problem_all_trees,
)
from dpvtex.log_tree_searches.quantify_labeling_problem import (
    _filter_last_tree_per_replicate as quant_filter,
    discover_replicates,
)


# =============================================================================
# _filter_last_tree_per_replicate
# =============================================================================


class TestFilterLastTreePerReplicate:
    """Tests for _filter_last_tree_per_replicate (both copies)."""

    def _make_df(self, rows):
        return pd.DataFrame(
            rows,
            columns=["dataset", "start_type", "replicate", "tree_index", "value"],
        )

    def test_keeps_max_tree_index_per_group(self):
        df = self._make_df(
            [
                ("ds1", "random", "rep1", 0, "a"),
                ("ds1", "random", "rep1", 1, "b"),
                ("ds1", "random", "rep1", 2, "c"),
                ("ds1", "random", "rep2", 0, "d"),
                ("ds1", "random", "rep2", 5, "e"),
            ]
        )
        for fn in (quant_filter, plot_filter):
            result = fn(df)
            assert len(result) == 2
            assert set(result["value"]) == {"c", "e"}

    def test_single_tree_per_replicate_returns_all(self):
        df = self._make_df(
            [
                ("ds1", "random", "rep1", 0, "a"),
                ("ds1", "random", "rep2", 0, "b"),
                ("ds1", "parsimony", "rep1", 0, "c"),
            ]
        )
        for fn in (quant_filter, plot_filter):
            result = fn(df)
            assert len(result) == 3

    def test_filters_correctly_per_group(self):
        df = self._make_df(
            [
                ("ds1", "random", "rep1", 0, "a"),
                ("ds1", "random", "rep1", 3, "b"),
                ("ds2", "parsimony", "rep1", 0, "c"),
                ("ds2", "parsimony", "rep1", 1, "d"),
                ("ds1", "parsimony", "rep1", 0, "e"),
                ("ds1", "parsimony", "rep1", 7, "f"),
            ]
        )
        for fn in (quant_filter, plot_filter):
            result = fn(df)
            assert len(result) == 3
            assert set(result["value"]) == {"b", "d", "f"}


# =============================================================================
# discover_replicates
# =============================================================================


class TestDiscoverReplicates:
    """Tests for discover_replicates."""

    def test_yields_correct_tuples(self):
        with tempfile.TemporaryDirectory() as root:
            # Create expected layout
            ds, st = "mydata", "random"
            rep_dir = os.path.join(root, "treesearch", f"{st}_starting", ds)
            os.makedirs(rep_dir)
            for i in range(3):
                path = os.path.join(rep_dir, f"{ds}_rep{i}_tree_search.p")
                open(path, "w").close()

            results = list(discover_replicates(root, [ds], [st]))
            assert len(results) == 3
            for dataset, start_type, pickle_path in results:
                assert dataset == ds
                assert start_type == st
                assert os.path.isfile(pickle_path)

    def test_warns_on_missing_pickle_dir(self, capsys):
        with tempfile.TemporaryDirectory() as root:
            results = list(discover_replicates(root, ["nope"], ["random"]))
            assert results == []
            captured = capsys.readouterr()
            assert "Warning" in captured.out


# =============================================================================
# plot_labeling_problem
# =============================================================================


class TestPlotLabelingProblem:
    """Tests for plot_labeling_problem."""

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame(
            {
                "dataset": ["ds1"] * 4 + ["ds2"] * 4,
                "start_type": ["random", "random", "parsimony", "parsimony"] * 2,
                "replicate": [f"rep{i}" for i in range(4)] * 2,
                "score_gap": [1, 2, -1, 0, 3, 4, 2, 1],
                "frac_non_dag_edges": [0.1, 0.2, 0.05, 0.15, 0.3, 0.25, 0.1, 0.2],
            }
        )

    def test_creates_pdf(self, sample_df):
        with tempfile.TemporaryDirectory() as tmpdir:
            plot_labeling_problem(sample_df, tmpdir)
            assert os.path.isfile(os.path.join(tmpdir, "labeling_problem_summary.pdf"))

    def test_backward_compat_frac_suspect_labels(self):
        df = pd.DataFrame(
            {
                "dataset": ["ds1", "ds1"],
                "start_type": ["random", "random"],
                "replicate": ["rep0", "rep1"],
                "score_gap": [1, 2],
                "frac_suspect_labels": [0.1, 0.2],
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            plot_labeling_problem(df, tmpdir)
            assert os.path.isfile(os.path.join(tmpdir, "labeling_problem_summary.pdf"))


# =============================================================================
# plot_labeling_problem_all_trees
# =============================================================================


class TestPlotLabelingProblemAllTrees:
    """Tests for plot_labeling_problem_all_trees."""

    @pytest.fixture
    def sample_all_trees_df(self):
        rows = []
        for ds in ["ds1", "ds2"]:
            for st in ["random", "parsimony"]:
                for rep_i in range(2):
                    for ti in range(5):
                        rows.append(
                            {
                                "dataset": ds,
                                "start_type": st,
                                "replicate": f"{ds}_rep{rep_i}_tree_search.p",
                                "tree_index": ti,
                                "normalized_tree_index": ti / 4,
                                "frac_non_dag_edges": 0.3 - 0.05 * ti,
                            }
                        )
        return pd.DataFrame(rows)

    def test_creates_pdf(self, sample_all_trees_df):
        with tempfile.TemporaryDirectory() as tmpdir:
            plot_labeling_problem_all_trees(sample_all_trees_df, tmpdir)
            assert os.path.isfile(
                os.path.join(tmpdir, "labeling_problem_all_trees.pdf")
            )
