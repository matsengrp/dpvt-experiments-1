import historydag as hdag
from ete3 import Tree

from dpvtex.larch.scripts.extract_data_from_hdag import (
    extract_hdag_splits,
    root_and_outgroup_leaf,
    get_MP_trees_from_hdag,
    del_outgroup_eq_root,
    edge_labels_for_split_set,
)


# dictionary for assigning node sequences
node_to_sequence = {
    "s1": "AA",
    "s2": "CA",
    "s3": "CG",
    "s4": "TG",
    "s5": "GG",
    "i1": "CA",
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


def create_test_trees():
    # create two trees for testing
    tree1_nwk = "((((s1,s2)i1,s3)i2,s4)i3,s5)i3;"
    tree2_nwk = "((s1,s2)i1,((s4,s5)i3,s3)i2)i2;"
    tree1 = Tree(tree1_nwk, format=8)
    tree2 = Tree(tree2_nwk, format=8)
    trees = [tree1, tree2]
    assign_sequences(trees, node_to_sequence)
    return trees


def trees_rooted_at_outgroup():
    # Trees identical to those in create_test_trees(), but with outgroup s5
    tree1_expected_nwk = "((((s1,s2)i1,s3)i2,s4)i3,s5)s5;"
    tree2_expected_nwk = "((((s1,s2)i1,s3)i2,s4)i3,s5)s5;"
    tree1_expected = Tree(tree1_expected_nwk, format=8)
    tree2_expected = Tree(tree2_expected_nwk, format=8)
    trees = [tree1_expected, tree2_expected]
    assign_sequences(trees, node_to_sequence)
    return trees


def compare_trees(node1, node2):
    # Recursively check if the trees rooted in node1 and node2 have
    # the same topology and same sequences assigned to all nodes
    # Check if both nodes have the same sequence
    if node1.sequence != node2.sequence:
        print("node1 sequence: ", node1.sequence, node1.name)
        print("node2 sequence: ", node2.sequence, node2.name)
        return False

    # Check if both nodes have the same number of children
    children1 = node1.children
    children2 = node2.children
    if len(children1) != len(children2):
        return False

    # Match and compare children pairs
    used_indices = set()
    for child1 in children1:
        match_found = False
        for i, child2 in enumerate(children2):
            if i not in used_indices and compare_trees(child1, child2):
                used_indices.add(i)
                match_found = True
                break
        if not match_found:
            return False
    return True


def test_root_and_outgroup_leaf():
    trees = create_test_trees()
    expected_trees = trees_rooted_at_outgroup()
    for tree in trees:
        root_and_outgroup_leaf(tree, tree & "s5")
    assert compare_trees(trees[0], expected_trees[0]) and compare_trees(
        trees[1], expected_trees[1]
    )


def create_test_hdag():
    trees = create_test_trees()
    dag = hdag.history_dag_from_trees(trees, ["sequence", "name", "node_id"])
    return dag


def test_get_MP_trees_from_hdag():
    # check if the tree topology extracted from hdag matches the two MP tree
    # topologies we put in -- this is testing get_MP_trees_from_hdag with the
    # flag unlabel=True
    dag = create_test_hdag()
    [tree1, tree2] = create_test_trees()
    trees_from_dag = get_MP_trees_from_hdag(dag, 3, unlabel=True)
    print("hdag trees:")
    for tree in trees_from_dag:
        print(tree)
    print("input trees:")
    print(tree1)
    print(tree2)
    if len(trees_from_dag) != 2:
        raise ValueError("DAG contains more than 2 trees")
    assert (
        tree1.robinson_foulds(trees_from_dag[0])[0] == 0
        and tree2.robinson_foulds(trees_from_dag[1])[0] == 0
    ) or (
        tree1.robinson_foulds(trees_from_dag[1])[0] == 0
        and tree2.robinson_foulds(trees_from_dag[0])[0] == 0
    )


def test_extract_hdag_splits():
    # make sure all splits from all input trees are extracted from hdag
    # this function indirectly also tests the function split()
    trees = create_test_trees()
    expected_splits = []
    for tree in trees:
        all_taxa = frozenset(tree.get_leaf_names())
        for node in tree.traverse():
            clade1 = frozenset(node.get_leaf_names())
            clade2 = frozenset(all_taxa - clade1)
            expected_splits.append(frozenset({clade1, clade2}))
    expected_splits = frozenset(expected_splits)
    dag = create_test_hdag()
    extracted_splits = extract_hdag_splits(dag)
    assert extracted_splits == expected_splits


def test_del_outgroup_eq_root():
    trees = trees_rooted_at_outgroup()
    # expected trees are rooted in leaf s5 that was outgroup in original trees
    tree1_expected_nwk = "((((s1,s2)i1,s3)i2,s4)i3)s5;"
    tree2_expected_nwk = "((((s1,s2)i1,s3)i2,s4)i3)s5;"
    tree1_expected = Tree(tree1_expected_nwk, format=8)
    tree2_expected = Tree(tree2_expected_nwk, format=8)
    assign_sequences([tree1_expected, tree2_expected], node_to_sequence)
    for tree in trees:
        del_outgroup_eq_root(tree)
    assert compare_trees(trees[0], tree1_expected) and compare_trees(
        trees[1], tree2_expected
    )


def test_edge_labels_for_split_set():
    # create tree3 that has split s1s3|s2s4s5 that is not present in dag
    tree3_nwk = "((s4,s5)a1,((s1,s3)a2,s2)a3)a4;"
    tree3 = Tree(tree3_nwk, format=8)
    assign_sequences([tree3], node_to_sequence, expect_internal_sequences=False)
    dag = create_test_hdag()
    dag_splits = extract_hdag_splits(dag)
    taxon_set = frozenset(tree3.get_leaf_names())
    expected_edge_list = [0, 0, 0, 0, 0, 1, 0, 0, 0]
    edge_list = edge_labels_for_split_set(tree3, taxon_set, dag_splits)
    assert expected_edge_list == edge_list
