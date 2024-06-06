import historydag as hdag
import pickle
import random

from dpvtex.perfect_phylogenies.perturb_phylogeny import (
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


def root_and_outgroup_leaf(tree, leaf):
    """
    Re-root tree by setting given leaf as root with its only child being the previous
    root.
    Args:
        tree: ete3 tree
        leaf: leaf in given tree
    """
    tree.set_outgroup(leaf)
    leaf.detach()
    tree.name = leaf.name
    tree.sequence = leaf.sequence


def extract_hdag_clade_child_clades(dag):
    """
    Generate dict containing frozensets C: C_1, .., C_k where C is clade in DAG and
        C_1, ..., C_k its child clades
    Args:
        dag: historydag.sequence_dag
    Returns:
        dict: contains for each clade in dag a list ofits child clades
    """

    def get_clade(node):
        # extract clade for node in dag
        cu = node.clade_union()
        clade = frozenset(node.node_id for node in cu)
        child_clades = frozenset(
            frozenset(n.node_id for n in cu) for cu in node.child_clades()
        )
        return {clade: child_clades}

    dag_clades = {}
    for node in dag.postorder():
        if not node.is_ua_node():
            dag_clades.update(get_clade(node))
    return dag_clades


def assign_edge_labels(modified_tree, tree, dag_clades):
    """
    Assigns label 0/1 to modified tree edges, depending on whether the edges are
    supported by dag_splits.
    Edges that are present in tree are assigned 0, as the tree is assumed to be
    extracted from the hdag, whlich makes the label assignment more efficient.
    Args:
        modified_tree: ete3 tree for which we want to get edge label list
        tree: ete3 tree that is mostly identical to modified tree (tree before
            make_worse)
        dag_clades: dictionary with clades: child_clades
    """
    # label edges that differ between tree and  modified tree as 1, else 0
    tree_clades = [
        frozenset(node.get_leaf_names()) for node in tree.traverse("preorder")
    ]
    edge_labels = [
        0 if frozenset(node.get_leaf_names()) in tree_clades else 1
        for node in modified_tree.traverse("preorder")
    ]
    # update 1s if corresponding edge exists in dag_splits or is resolution of
    # a multifurcation in dag_splits.
    i = 0
    for node in modified_tree.traverse("preorder"):
        if edge_labels[i] == 1:
            clade = frozenset(node.get_leaf_names())
            # if clade actually exists in DAG, label as 0:
            if clade in dag_clades:
                edge_labels[i] = 0
            # if edge is resolution of multifurcation in dag, we also label as 0
            # (MP edge with 0 mutations)
            else:
                for dag_clade in dag_clades:
                    if clade.issubset(dag_clade):
                        at_edge_resolution = True  # we assume we are at multifurcation that supports edge
                        for child_clade in dag_clades[dag_clade]:
                            if clade.intersection(child_clade) not in [
                                frozenset(),
                                child_clade,
                                clade,
                            ]:
                                at_edge_resolution = False
                                break
                        if at_edge_resolution:
                            edge_labels[i] = 0
                            break
        i += 1
    return edge_labels


def get_non_dag_edges(dag, num_children_file, num_trees=0):
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
    tree_to_label_dict = {}  # output dict
    dag_clades = extract_hdag_clade_child_clades(dag)

    with open(num_children_file, "w") as nc_file:
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
            num_children_list = []
            for node in tree.traverse():
                if not node.is_leaf():
                    num_children_list.append(len(node.get_children()))
            line = ','.join(str(i) for i in num_children_list)
            nc_file.write(line + "\n")
            tree.resolve_polytomy()
            sankoff_for_missing_sequences(tree)
            # introduce non-MP edges, if possible
            td = tree_depth(tree)
            done_modifying=False
            modified_tree = tree.copy()
            i=0
            while not done_modifying:
                # make tree worse until at least a third of all edges are non MP
                print("Tree modification iteration ", i)
                i+=1
                modified_tree = make_worse_tree(modified_tree, td // 2)
                if modified_tree is None:
                    modified_tree = tree
                # assign edge labels
                edge_labels = assign_edge_labels(modified_tree, tree, dag_clades)
                tree_to_label_dict[modified_tree] = edge_labels
                # if sum(edge_labels)/len(edge_labels) >= 1/6:
                #     # note that len(edge_labels) is roughly 2*internal edges
                done_modifying = True
    if len(tree_to_label_dict) < num_trees:
        print("Produced ", len(tree_to_label_dict), " trees instead of ", num_trees)
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
        num_children_file = sys.argv[3]

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

    tree_label_dict = get_non_dag_edges(dag, num_children_file, num_topologies)
    with open(dpvt_data_file, "wb") as f:
        pickle.dump(tree_label_dict, f)


if __name__ == "__main__":
    main()
