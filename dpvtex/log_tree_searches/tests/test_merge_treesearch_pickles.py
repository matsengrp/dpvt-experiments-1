"""Tests for merge_treesearch_pickles.py."""

import os
import pickle
from glob import glob

import pytest

from dpvtex.log_tree_searches.merge_treesearch_pickles import merge_pickle_files


@pytest.fixture
def pickle_dir(tmp_path):
    """Create a directory with sample treesearch pickle files."""
    alignment_dir = tmp_path / "nj_starting" / "alignment1"
    alignment_dir.mkdir(parents=True)

    data1 = {"tree_A": [0, 1, 0], "tree_B": [1, 0, 1]}
    with open(alignment_dir / "alignment1_tree_search.p", "wb") as f:
        pickle.dump(data1, f)

    alignment_dir2 = tmp_path / "nj_starting" / "alignment2"
    alignment_dir2.mkdir(parents=True)

    data2 = {"tree_C": [1, 1, 0], "tree_D": [0, 0, 1]}
    with open(alignment_dir2 / "alignment2_tree_search.p", "wb") as f:
        pickle.dump(data2, f)

    return tmp_path


def test_merge_pickle_files(pickle_dir):
    """Merging two non-overlapping pickles produces the union of their keys."""
    pattern = os.path.join(str(pickle_dir), "**", "*_tree_search.p")
    files = sorted(glob(pattern, recursive=True))
    merged = merge_pickle_files(files)

    assert len(merged) == 4
    assert set(merged.keys()) == {"tree_A", "tree_B", "tree_C", "tree_D"}
    assert merged["tree_A"] == [0, 1, 0]
    assert merged["tree_D"] == [0, 0, 1]


def test_merge_pickle_files_duplicate_keys(pickle_dir):
    """Duplicate tree keys across files raise ValueError."""
    # Add a third pickle with a duplicate key
    alignment_dir3 = pickle_dir / "nj_starting" / "alignment3"
    alignment_dir3.mkdir(parents=True)
    data3 = {"tree_A": [9, 9, 9]}  # duplicate of tree_A in alignment1
    with open(alignment_dir3 / "alignment3_tree_search.p", "wb") as f:
        pickle.dump(data3, f)

    pattern = os.path.join(str(pickle_dir), "**", "*_tree_search.p")
    files = sorted(glob(pattern, recursive=True))

    with pytest.raises(ValueError, match="duplicate tree keys"):
        merge_pickle_files(files)


def test_merge_pickle_files_empty_list():
    """Merging an empty list returns an empty dict."""
    merged = merge_pickle_files([])
    assert merged == {}
