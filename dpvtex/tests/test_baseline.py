from dpvt import models
from ete3 import Tree
from dpvtex.larch.scripts.extract_data_from_hdag import root_and_outgroup_leaf
from dpvt.wrapper import Wraplet, TreeDataset
from dpvt.models import BaselineReversion

node_to_sequence = {
    "s1": "AA",
    "s2": "CA",
    "s3": "CG",
    "s4": "TG",
    "s5": "GA",
    "i1": "AA",
    "i2": "CG",
    "i3": "GG",
}


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


def create_test_tree():
    # create tree for testing
    nwk_tree = "((((s1,s2)i1,s3)i2,s4)i3,s5)i3;"
    tree = Tree(nwk_tree, format=8)
    assign_sequences([tree], node_to_sequence)
    root_and_outgroup_leaf(tree, tree & "s5")
    return tree


def test_baseline():
    tree = create_test_tree()
    true_labels = [1 if i.name == "i1" else 0 for i in tree.traverse("preorder")]
    # Create your test dataset
    test_dataset = TreeDataset([tree], [true_labels])

    # Initialize the wrapper with your model
    wraplet = Wraplet(
        test_data=test_dataset,
        model=BaselineReversion,
        device="cpu"  # or "cuda" if using GPU
    )

    # Run the test
    results = wraplet.test()
    print(results)
    print(results[0]["test_auroc"])
    auroc = results[0]["test_auroc"]
    assert auroc == 1.0, f"Expected AUROC of 1.0, but got {auroc}"
    
    
    
    
    


