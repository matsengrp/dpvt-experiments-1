from dpvt import models
from ete3 import Tree
import torch
from dpvtex.larch.scripts.extract_data_from_hdag import root_and_outgroup_leaf
from dpvt.wrapper import Wraplet, TreeDataset
from dpvt.models import BaselineReversion


def assign_sequences(trees, seq_dict, expect_internal_sequences=True):
    # assign sequences to all nodes in trees
    # seq_dict: dict with node.name : node.sequence for this assignment
    for tree in trees:
        for node in tree.traverse():
            if node.name in seq_dict:
                node.sequence = seq_dict[node.name]
            elif expect_internal_sequences:
                raise ValueError(f"No sequence found for node '{node.name}'")
            else:
                node.sequence = ""
            if node.is_leaf():
                node.node_id = node.name
            else:
                node.node_id = ""
    return trees


def create_test_trees():
    # create tree for testing
    nwk_tree1 = "((((s1,s2)i1,s3)i2,s4)i3,s5)i3;"
    node_to_sequence1 = {
        "s1": "AA",
        "s2": "CA",
        "s3": "CG",
        "s4": "TG",
        "s5": "GA",
        "i1": "AA",
        "i2": "CG",
        "i3": "GG",
    }
    tree1 = Tree(nwk_tree1, format=8)
    assign_sequences([tree1], node_to_sequence1)
    root_and_outgroup_leaf(tree1, tree1 & "s5")
    nwk_tree2 = "((((s1,s2)i1,(s3,s4)i2)i3,(s5,s6)i4)i5,s7)i5;"
    node_to_sequence2 = {
        "s1": "ACG",
        "s2": "CAG",
        "s3": "AGG",
        "s4": "CGG",
        "s5": "GCC",
        "s6": "GGG",
        "s7": "CCG",
        "i1": "CCG",
        "i2": "CGG",
        "i3": "ACG",
        "i4": "GCC",
        "i5": "CCG",
    }
    tree2 = Tree(nwk_tree2, format=8)
    assign_sequences([tree2], node_to_sequence2)
    root_and_outgroup_leaf(tree2, tree2 & "s7")
    trees = [tree1, tree2]
    return trees


def test_baseline():
    trees = create_test_trees()
    true_labels = []
    true_labels.append(
        [1 if i.name == "i1" else 0 for i in trees[0].traverse("preorder")]
    )
    true_labels.append(
        [1 if i.name in ["i1", "i2"] else 0 for i in trees[1].traverse("preorder")]
    )

    # Create your test dataset
    test_dataset = TreeDataset(trees, true_labels)

    # Initialize the wrapper with your model
    wraplet = Wraplet(
        test_data=test_dataset,
        model=BaselineReversion,
        device="cpu",  # or "cuda" if using GPU
    )

    m = BaselineReversion()
    pred_labels = []
    for i in [0, 1]:
        pred_labels.append(m.get_reversion_labels_from_tree(trees[i]))
        true_labels[i] = torch.tensor(true_labels[i], dtype=torch.float32)
    print(pred_labels)
    print(true_labels)
    assert torch.equal(pred_labels[0], true_labels[0]) and torch.equal(
        pred_labels[1], true_labels[1]
    )
