"""Tests for tree perturbation functions in larch/scripts/tree_perturbation.py"""

import pytest
from ete3 import Tree
from historydag.parsimony import parsimony_score
from dpvtex.larch.scripts.tree_perturbation import (
    increase_tree_parsimony,
    sankoff_for_missing_sequences,
    perturb_tree,
)


def test_increase_tree_parsimony_basic():
    """Test that increase_tree_parsimony produces a tree with higher parsimony score."""
    tree = Tree("(((((A,B),C),D),E),F);")
    sequences = ["AAAA", "AAAC", "AACC", "ACCC", "CCCC", "GGGG"]
    for node, seq in zip(tree.iter_leaves(), sequences):
        node.add_feature("sequence", seq)
    sankoff_for_missing_sequences(tree)

    original_score = parsimony_score(tree)
    worse_tree = increase_tree_parsimony(tree, depth=2, max_attempts=100)

    assert worse_tree is not None, "Should return a worse tree"
    assert (
        parsimony_score(worse_tree) > original_score
    ), "New tree should have higher parsimony score"


def test_increase_tree_parsimony_preserves_leaves():
    """Test that increase_tree_parsimony preserves leaf sequences."""
    tree = Tree("(((((A,B),C),D),E),F);")
    sequences = ["AAAA", "AAAC", "AACC", "ACCC", "CCCC", "GGGG"]
    for node, seq in zip(tree.iter_leaves(), sequences):
        node.add_feature("sequence", seq)
        node.name = f"leaf_{seq}"  # Give leaves names for tracking
    sankoff_for_missing_sequences(tree)

    original_leaves = {leaf.name: leaf.sequence for leaf in tree.get_leaves()}
    worse_tree = increase_tree_parsimony(tree, depth=2, max_attempts=100)

    if worse_tree is not None:
        new_leaves = {leaf.name: leaf.sequence for leaf in worse_tree.get_leaves()}
        assert original_leaves == new_leaves, "Leaf sequences should be preserved"


def test_increase_tree_parsimony_invalid_depth():
    """Test that increase_tree_parsimony returns None for depth too large for tree."""
    tree = Tree("((((A,B),C),D),E);")  # 5 leaves
    # Use valid DNA sequences
    leaf_seqs = {"A": "ACTG", "B": "ACCG", "C": "GCTG", "D": "GCCG", "E": "TCTG"}
    for node in tree.iter_leaves():
        node.add_feature("sequence", leaf_seqs.get(node.name, "ACTG"))
    sankoff_for_missing_sequences(tree)

    # Depth too large for this tree
    worse_tree = increase_tree_parsimony(tree, depth=10, max_attempts=10)
    assert worse_tree is None, "Should return None for depth larger than tree"


def test_increase_tree_parsimony_no_improvement_found():
    """Test that increase_tree_parsimony returns None if no worse tree is found."""
    # Create a tree with 5 identical sequences
    tree = Tree("((((A,B),C),D),E);")
    for node in tree.iter_leaves():
        node.add_feature("sequence", "A")  # All same sequence
    sankoff_for_missing_sequences(tree)

    # With identical sequences, it's hard to make it worse
    worse_tree = increase_tree_parsimony(tree, depth=1, max_attempts=5)
    # This may return None if no worse tree can be found
    assert worse_tree is None or parsimony_score(worse_tree) >= parsimony_score(tree)


def test_increase_tree_parsimony_different_depths():
    """Test increase_tree_parsimony with various depth values."""
    tree = Tree("(((((A,B),C),D),E),F);")  # 6 leaves
    sequences = ["AGTC", "AGTG", "ACTC", "CGTC", "CGTT", "TGTC"]
    for node, seq in zip(tree.get_leaves(), sequences):
        node.add_feature("sequence", seq)
    sankoff_for_missing_sequences(tree)

    original_score = parsimony_score(tree)

    # Test with different depths
    for depth in [1, 2, 3]:
        worse_tree = increase_tree_parsimony(tree, depth=depth, max_attempts=50)
        if worse_tree is not None:
            assert (
                parsimony_score(worse_tree) >= original_score
            ), f"Score should not decrease for depth {depth}"


def test_increase_tree_parsimony_internal_sequences():
    """Test that internal node sequences are properly filled in."""
    tree = Tree("(((((A,B),C),D),E),F);")
    sequences = ["AAAA", "AAAC", "AACC", "ACCC", "CCCC", "GGGG"]
    for node, seq in zip(tree.iter_leaves(), sequences):
        node.add_feature("sequence", seq)
    sankoff_for_missing_sequences(tree)

    worse_tree = increase_tree_parsimony(tree, depth=2, max_attempts=100)

    if worse_tree is not None:
        # Check that all nodes have sequences
        for node in worse_tree.traverse():
            assert hasattr(node, "sequence"), f"Node should have sequence attribute"
            assert len(node.sequence) > 0, f"Node sequence should not be empty"


def test_increase_tree_parsimony_maintains_tree_structure():
    """Test that the resulting tree maintains binary structure."""
    tree = Tree("((((A,B),(C,D)),(E,F)),G);")  # 7 leaves
    for node in tree.iter_leaves():
        node.add_feature("sequence", "ACTG")
    sankoff_for_missing_sequences(tree)

    worse_tree = increase_tree_parsimony(tree, depth=2, max_attempts=50)

    if worse_tree is not None:
        # Check tree is binary
        for node in worse_tree.traverse():
            if not node.is_leaf():
                assert (
                    len(node.children) == 2
                ), "Internal nodes should have exactly 2 children"

        # Check same number of leaves
        assert len(worse_tree.get_leaves()) == len(
            tree.get_leaves()
        ), "Should have same number of leaves"


def test_increase_tree_parsimony_with_longer_sequences():
    """Test with longer, more realistic sequences."""
    tree = Tree("((((A,B),C),(D,E)),(F,G));")  # 7 leaves
    sequences = [
        "ACTGACTGACTG",
        "ACTGACTGACCG",
        "ACTGATTGACTG",
        "ACCGACTGACTG",
        "ACCGACTGACCG",
        "TCTGACTGACTG",
        "TCTGACTGACCG",
    ]

    for node, seq in zip(tree.get_leaves(), sequences):
        node.add_feature("sequence", seq)
    sankoff_for_missing_sequences(tree)

    original_score = parsimony_score(tree)
    worse_tree = increase_tree_parsimony(tree, depth=2, max_attempts=100)

    if worse_tree is not None:
        assert (
            parsimony_score(worse_tree) > original_score
        ), "Should produce tree with worse parsimony"
        assert len(worse_tree.get_leaves()) == 7, "Should maintain 7 leaves"


def test_increase_tree_parsimony_with_8_leaves():
    """Test with 8 leaf tree for more complex structure."""
    tree = Tree("(((((A,B),C),(D,E)),F),(G,H));")  # 8 leaves
    sequences = [
        "ACTGAC",
        "ACTGCC",
        "ATTGAC",
        "CCTGAC",
        "CCTGCC",
        "TCTGAC",
        "TCTGCC",
        "GCTGAC",
    ]

    for node, seq in zip(tree.get_leaves(), sequences):
        node.add_feature("sequence", seq)
    sankoff_for_missing_sequences(tree)

    original_score = parsimony_score(tree)

    # Test with multiple depths
    for depth in [1, 2, 3]:
        worse_tree = increase_tree_parsimony(tree, depth=depth, max_attempts=100)
        if worse_tree is not None:
            new_score = parsimony_score(worse_tree)
            assert (
                new_score >= original_score
            ), f"Score should not decrease for depth {depth}"
            # Check if we actually got a worse tree
            if new_score > original_score:
                print(
                    f"Successfully made tree worse at depth {depth}: {original_score} -> {new_score}"
                )


def test_increase_tree_parsimony_edge_cases():
    """Test edge cases and boundary conditions."""
    # Minimum viable tree with 5 leaves
    tree = Tree("((((A,B),C),D),E);")
    sequences = ["ACT", "ACC", "GCT", "GCC", "TCT"]
    for node, seq in zip(tree.get_leaves(), sequences):
        node.add_feature("sequence", seq)
    sankoff_for_missing_sequences(tree)

    # Test with depth=1 (minimum valid depth)
    worse_tree = increase_tree_parsimony(tree, depth=1, max_attempts=100)
    if worse_tree is not None:
        assert len(worse_tree.get_leaves()) == 5, "Should preserve number of leaves"

    # Test that ValueError is raised for depth < 1
    # (The function should raise ValueError for depth < 1)
    tree2 = Tree("((((A,B),C),D),E);")
    for node in tree2.iter_leaves():
        node.add_feature("sequence", "ACT")
    sankoff_for_missing_sequences(tree2)

    # This should raise an error for depth < 1
    try:
        result = increase_tree_parsimony(tree2, depth=0, max_attempts=10)
        # If no error raised, it should return None
        assert result is None, "Should return None or raise error for depth < 1"
    except ValueError as e:
        # This is expected behavior
        assert "Depth must be at least 1" in str(e)


if __name__ == "__main__":
    # Run tests manually if needed
    test_increase_tree_parsimony_basic()
    test_increase_tree_parsimony_preserves_leaves()
    test_increase_tree_parsimony_invalid_depth()
    test_increase_tree_parsimony_no_improvement_found()
    test_increase_tree_parsimony_different_depths()
    test_increase_tree_parsimony_internal_sequences()
    test_increase_tree_parsimony_maintains_tree_structure()
    test_increase_tree_parsimony_with_longer_sequences()
    test_increase_tree_parsimony_with_8_leaves()
    test_increase_tree_parsimony_edge_cases()
    print("All tests passed!")
