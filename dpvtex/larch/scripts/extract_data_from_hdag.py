import historydag as hdag
import pickle
import random

from perfect_phylogenies.perturb_phylogeny import (
    make_worse_tree,
    tree_depth,
    sankoff_for_missing_sequences,
)
import sys


def get_MP_trees_from_hdag(dag, num_trees, unlabel=False):
    """
    Samples num_trees uniformly from input historydag dag without replacement
    Args:
        dag: compact genome historydag without ambiguous sequences
        num_trees: int - number of trees to sample
        unlabel: if true, we unlabel the DAG and then only sample each topology
            once. This needs to be used with care if the input dag has ambiguous
            sequences.
    Returns:
        list of ete trees
    """
    if unlabel:
        dag = dag.unlabel()
    dag.uniform_distribution_annotate()
    num_samples = min(num_trees, dag.count_topologies())
    if num_samples != num_trees:
        print(
            "Not enough trees in DAG to sample",
            num_trees,
            "trees. Sample",
            num_samples,
            "trees instead.",
        )
    sample_ids = random.sample(range(dag.count_histories()), num_samples)

    tree_samples = [
        dag[i].to_ete(name_func=lambda n: n.label.node_id, features=["sequence"])
        for i in sample_ids
    ]
    return tree_samples


def split(taxon_set, node):
    """Returns the split given by node, a node of a historydag.
    Args:
        node: historydag.dag_node
    Returns:
        frozenset of bipartitions {S_1, S_2} of leaf set, i.e. splits"""
    cu = node.clade_union()
    set1 = frozenset(node.node_id for node in cu)
    set2 = frozenset(node.node_id for node in taxon_set - cu)
    return frozenset({set1, set2})


def edge_labels_for_split_set(tree, taxon_set, split_set):
    """
    Check for each edge in tree if its split is in the given list of splits.
    Returns list of 0/1 labels indicating whether an edge is in dag or not,
    where edges are sorted according to preorder traversal in tree
    Args:
        tree: ete3 tree
        split_set: frozenset containing frozensets S_i representing splits.
            Each S_i represents one split and contains two frozensets, one for
            each clade of the split
        taxon_set: frozenset containing names of all taxa of tree and split_set --
            both are assumed to have the the same taxon set
    Returns:
        edge_list: list of 0/1 indicating for each edge in tree (perorder) whether
            it is in dag or not
    """
    # preorder traversal needed for correct assignment of edge labels
    clades = [frozenset(node.get_leaf_names()) for node in tree.traverse("preorder")]
    splits = [frozenset({clade, taxon_set - clade}) for clade in clades]
    # Note that hDAG is not binary, so there could be MP edges that are in binary tree
    # and not in hDAG
    edge_labels = [0 if split in split_set else 1 for split in splits]
    return edge_labels


def root_and_outgroup_leaf(tree, leaf):
    """
    Re-root tree by setting given leaf as outgroup, and set root's name and sequence to be
    this leaf's sequence.
    Args:
        tree: ete3 tree
        leaf: leaf in given tree
    """
    tree.set_outgroup(leaf)
    tree.name = leaf.name
    tree.sequence = leaf.sequence


def extract_hdag_splits(dag):
    """
    Generate frozenset containing frozensets S_1, .., S_k, each representing a split in
    dag. S_i itself is a frozenset containing two frozenset that build bipartition of
    leaf set of dag
    Args:
        dag: historydag.sequence_dag
    Returns:
        frozenset: contains one frozenset for each edge in dag that contains bipartition
            for this split in dag
    """
    taxon_set = frozenset([n.label for n in dag.get_leaves()])
    dag_splits = frozenset(
        split(taxon_set, dag_node)
        for dag_node in dag.postorder()
        if not dag_node.is_ua_node()
    )
    return dag_splits


def del_outgroup_eq_root(tree):
    """
    Assumes that tree has single outgroup taxon with the same sequence as root.
    If not, returns an error, otherwise deletes this outgroup taxon, so that
    root has one child and represents this leaf.
    Args:
        tree: ete3 tree
    Returns:
        None
    """
    # find outgroup leaf -- levelorder traversal should find outgroup leaf very
    # quickly
    for node in tree.traverse("levelorder"):
        if node.is_leaf() and node.sequence == tree.sequence:
            root_leaf = node
            break
    root_leaf.delete()


def get_non_dag_edges(dag, num_trees=0):
    """
    Perturbs trees in tree_list to create num_trees perturbed trees containing
    edges that are not present in the given dag
    Args:
        tree_list: list of ete trees
        dag: sequence_dag
        num_trees: number of trees to return. If 0, returns as many trees as
            there are in tree_list
    Returns:
        dictionary with keys ete3 trees and values list of edge labels indicating
        MP (0) vs non-MP (1) edges, sorted by preorder traversal
    """
    mp_trees = get_MP_trees_from_hdag(dag, num_trees, unlabel=True)
    output_trees = []
    for tree in mp_trees:
        # delete sequences on internal nodes - can probably be done more efficiently
        for node in tree.traverse():
            if not node.is_leaf():
                node.del_feature("sequence")
        # random leaf 'root_leaf' chosen as outgroup -- only delete leaf when we have
        # edge labels MP/non-MP to be able to compare tree edges to hDAG edges
        root_leaf = tree.get_leaves()[0]
        root_and_outgroup_leaf(tree, root_leaf)
        # make tree binary + disambiguate (Sankoff)
        tree.resolve_polytomy()
        sankoff_for_missing_sequences(tree)
        # introduce non-MP edges, if possible
        td = tree_depth(tree)
        modified_tree = make_worse_tree(tree, td // 3)  # NOTE: This is slow
        output_trees.append(modified_tree if modified_tree is not None else tree)

    dag_splits = extract_hdag_splits(dag)

    # extract splits from tree and compare with hDAG splits
    tree_to_label_dict = {}  # output dict
    taxon_names = frozenset(output_trees[0].get_leaf_names())
    for tree in output_trees:
        edge_list = edge_labels_for_split_set(tree, taxon_names, dag_splits)
        tree_to_label_dict[tree] = edge_list
        # Delete outgroup leaf -- root has same sequence, i.e. root is this leaf now
        del_outgroup_eq_root(tree)
    return tree_to_label_dict


def main():
    if len(sys.argv) < 2:
        print(
            "Error: Please provide file containing pickled hDAG or protobuf and filename for dpvt data."
        )
        sys.exit(1)
    else:
        dag_file = sys.argv[1]
        dpvt_data_file = sys.argv[2]

    if dag_file[-2:] == ".p":
        with open(dag_file, "rb") as f:
            dag = pickle.load(f)
    elif dag_file[-3:] == ".pb":
        dag = hdag.mutation_annotated_dag.load_MAD_protobuf_file(
            dag_file, compact_genomes=True
        )
    else:
        print("Error: First input file should be pickled hDAG or protobuf.")
        sys.exit(1)
    # trim to only MP topologies + convert to sequence_dag
    dag.trim_optimal_weight()
    dag = hdag.sequence_dag.SequenceHistoryDag.from_history_dag(dag)
    num_topologies = dag.count_topologies()

    tree_label_dict = get_non_dag_edges(dag, num_topologies)
    with open(dpvt_data_file, "wb") as f:
        pickle.dump(tree_label_dict, f)


if __name__ == "__main__":
    main()
