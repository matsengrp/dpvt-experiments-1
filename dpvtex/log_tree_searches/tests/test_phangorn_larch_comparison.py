"""Tests for quantify_phangorn_larch_comparison and plot_phangorn_larch_comparison scripts."""

import os
import tempfile

import pandas as pd

from dpvtex.log_tree_searches.plot_phangorn_larch_comparison import (
    plot_phangorn_larch_comparison,
    plot_phangorn_larch_comparison_all_trees,
)
from dpvtex.log_tree_searches.quantify_phangorn_larch_comparison import (
    discover_replicates,
)


# =============================================================================
# discover_replicates
# =============================================================================


def test_discover_replicates_yields_correct_tuples():
    with tempfile.TemporaryDirectory() as root:
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


def test_discover_replicates_warns_on_missing_pickle_dir(capsys):
    with tempfile.TemporaryDirectory() as root:
        results = list(discover_replicates(root, ["nope"], ["random"]))
        assert results == []
        captured = capsys.readouterr()
        assert "Warning" in captured.out


# =============================================================================
# plot_phangorn_larch_comparison
# =============================================================================


def test_plot_phangorn_larch_comparison_creates_pdf():
    df = pd.DataFrame(
        {
            "dataset": ["ds1"] * 4 + ["ds2"] * 4,
            "start_type": ["random", "random", "nj", "nj"] * 2,
            "replicate": [f"rep{i}" for i in range(4)] * 2,
            "score_gap": [1, 2, -1, 0, 3, 4, 2, 1],
            "frac_non_dag_edges": [0.1, 0.2, 0.05, 0.15, 0.3, 0.25, 0.1, 0.2],
        }
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        plot_phangorn_larch_comparison(df, tmpdir)
        assert os.path.isfile(
            os.path.join(tmpdir, "phangorn_larch_comparison_summary.pdf")
        )


def test_plot_phangorn_larch_comparison_backward_compat_frac_suspect_labels():
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
        plot_phangorn_larch_comparison(df, tmpdir)
        assert os.path.isfile(
            os.path.join(tmpdir, "phangorn_larch_comparison_summary.pdf")
        )


# =============================================================================
# plot_phangorn_larch_comparison_all_trees
# =============================================================================


def test_plot_phangorn_larch_comparison_all_trees_creates_pdf():
    rows = []
    for ds in ["ds1", "ds2"]:
        for st in ["random", "nj"]:
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
    df = pd.DataFrame(rows)
    with tempfile.TemporaryDirectory() as tmpdir:
        plot_phangorn_larch_comparison_all_trees(df, tmpdir)
        assert os.path.isfile(
            os.path.join(tmpdir, "phangorn_larch_comparison_all_trees.pdf")
        )
