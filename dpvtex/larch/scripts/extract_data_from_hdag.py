import historydag as hdag
import pickle
import random
import os
import signal
import multiprocessing
import time
import subprocess
import psutil

from dpvtex.perfect_phylogenies.perturb_phylogeny import (
    make_worse_tree,
    make_worse_spr,
    tree_depth,
    sankoff_for_missing_sequences,
)
import sys


def get_MP_trees_from_hdag(dag, num_trees, unlabel=True):
    """
    Samples num_trees uniformly from input historydag dag without replacement
    Args:
        dag: compact genome historydag without ambiguous sequences num_trees:
        int - number of trees to sample unlabel: if true, we unlabel the DAG and
        then only sample each topology
            once. This needs to be used with care if the input dag has ambiguous
            sequences.
    Returns:
        list of ete trees
    """
    if unlabel:
        dag = dag.unlabel()
    dag.uniform_distribution_annotate()
    # Only call memory_safe_count_topologies once
    dag_num_topologies = memory_safe_count_topologies(dag)
    num_samples = min(num_trees, dag_num_topologies)
    if num_samples != num_trees:
        print(
            "Not enough trees in DAG to sample",
            num_trees,
            "trees. Sample",
            num_samples,
            "trees instead.",
        )
    if dag_num_topologies == float("inf"):
        # we can reasonably expect to have more than 1000 topolgies in the DAG
        sample_ids = random.sample(range(1000), num_samples)
    else:
        sample_ids = random.sample(range(dag_num_topologies), num_samples)

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
    Re-root tree by setting given leaf as root with its only child being the
    previous root. Args:
        tree: ete3 tree leaf: leaf in given tree
    """
    tree.set_outgroup(leaf)
    leaf.detach()
    tree.name = leaf.name
    tree.sequence = leaf.sequence


def extract_hdag_clade_child_clades(dag):
    """
    Generate dict containing frozensets C: C_1, .., C_k where C is clade in DAG
    and
        C_1, ..., C_k its child clades
    Args:
        dag: historydag.sequence_dag
    Returns:
        dict: contains for each clade in dag a list of its child clades
    """

    def get_clade_child_clades(node):
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
            dag_clades.update(get_clade_child_clades(node))
    return dag_clades


def exists_subset_union(S, C):
    """
    Find if there is a collection S_1, ..., S_k of sets in frozenset S = {S_1,
    ..., S_N} (k <= N) whose union is exactly C. We assume that |S| = |S_1| +
    ... + |S_N|, i.e. no element is present in more than one S_i Args:
        S: frozenset containing frozensets C: frozenset
    Returns:
        True if there are sets S_1, ..., S_k with union C Else otherwise
    """
    union = set()  # we aim to create a union of subsets of S that equals C
    for subset in S:
        if subset.issubset(C):
            union.update(subset)
        elif len(subset.intersection(C)) > 0:
            # if subset intersects C but is not a subset of C, there is no union
            # of sets in S that results in C, as the elements in subset\C cannot
            # be added without adding elements in C\subset and we assume that no
            # element appears in more than one set in S
            return False
    if len(union) == len(C):
        return True
    return False


def assign_edge_labels(modified_tree, tree, dag_clades):
    """
    Assigns label 0/1 to modified_tree edges, depending on whether the edges are
    supported by dag_clades. `modified_tree' is assumed to be received from
    changing some edges in tree, so that a lot of edges are still shared between
    then two. `tree' is assumed to be extracted from the hdag, and is used to
    make the label assignment more efficient. Args:
        modified_tree: ete3 tree for which we want to get edge label list tree:
        ete3 tree that is mostly identical to modified tree (tree before
            make_worse)
        dag_clades: dictionary with clades: child_clades. Can be computed with
        extract_hdag_clade_child_clades
    Returns:
        list of 0/1 assigned to each node for each edge above it (preorder)
            whether it is present in the dag with dag_clades or not
    """
    # label edges that differ between tree and  modified tree as 1, else 0
    tree_clades = [
        frozenset(node.get_leaf_names()) for node in tree.traverse("preorder")
    ]
    leaf_set = frozenset(modified_tree.get_leaf_names())
    edge_labels = [
        0 if frozenset(node.get_leaf_names()) in tree_clades else 1
        for node in modified_tree.traverse("preorder")
    ]
    # update 1s if corresponding edge exists in dag_splits or is resolution of a
    # multifurcation in dag_splits.
    i = 0
    for node in modified_tree.traverse("preorder"):
        if i in [0, 1]:
            # Root leaf and node below root are assigned 0 by default This
            # doesn't change anything, as they will be masked in training
            edge_labels[i] = 0
            i += 1
            continue
        if edge_labels[i] == 1:
            this_clade = frozenset(node.get_leaf_names())
            # check this clade and its complement
            for clade in [this_clade, leaf_set - this_clade]:
                # if clade actually exists in DAG, label as 0:
                if clade in dag_clades:
                    edge_labels[i] = 0
                # if edge is resolution of multifurcation in dag, we also label as 0
                # (MP edge with 0 mutations)
                else:
                    for dag_clade in dag_clades:
                        if clade.issubset(dag_clade):
                            # If there is a union of clades that are children of
                            # dag_clade, then there is a multifurcation at dag_clade
                            # that could be resolved so that clade is in a DAG tree
                            if exists_subset_union(dag_clades[dag_clade], clade):
                                edge_labels[i] = 0
                                break
        i += 1
    return edge_labels


def get_non_dag_edges(dag, num_children_file, num_trees=0, use_make_worse_spr=True):
    """
    Perturbs trees in tree_list to create num_trees perturbed trees containing
    edges that are not present in the given dag Args:
        tree_list: list of ete trees dag: sequence_dag num_trees: number of
        trees to return. If 0, returns as many trees as
            there are in tree_list
    Returns:
        dictionary with keys ete3 trees and values list of edge labels
        indicating MP (0) vs non-MP (1) edges, sorted by preorder traversal
    """
    print("Start extracting MP trees from hDAG")
    mp_trees = get_MP_trees_from_hdag(dag, num_trees, unlabel=True)
    print("Extracted ", len(mp_trees), " trees from hDAG")
    tree_to_label_dict = {}  # output dict
    print("Start extracting clades from hDAG")
    dag_clades = extract_hdag_clade_child_clades(dag)
    print("Extracted clades from hDAG")

    print("Start adding non-MP edges...")
    print("Number of MP trees:", len(mp_trees))
    with open(num_children_file, "w") as nc_file:
        for tree in mp_trees:
            print("Processing tree number", len(tree_to_label_dict))
            # delete sequences on internal nodes - can probably be done more
            # efficiently
            for node in tree.traverse():
                if not node.is_leaf():
                    node.del_feature("sequence")
            # random leaf 'root_leaf' chosen as outgroup -- only delete leaf
            # when we have edge labels MP/non-MP to be able to compare tree
            # edges to hDAG edges
            root_leaf = tree.get_leaves()[0]
            root_and_outgroup_leaf(tree, root_leaf)
            # make tree binary + disambiguate (Sankoff)
            num_children_list = []
            for node in tree.traverse():
                if not node.is_leaf():
                    num_children_list.append(len(node.get_children()))
            line = ",".join(str(i) for i in num_children_list)
            nc_file.write(line + "\n")
            tree.resolve_polytomy()
            sankoff_for_missing_sequences(tree)
            # introduce non-MP edges, if possible
            td = tree_depth(tree)
            done_modifying = False
            modified_tree = tree.copy()
            i = 0
            while not done_modifying:
                # make tree worse until at least a third of all edges are non MP
                print("Tree modification iteration ", i)
                i += 1
                if use_make_worse_spr:
                    # Maximum of 100 SPR moves per iteration -- if we don't have enough non-MP
                    # edges after that, we add more in next iteration (until done_modifying)
                    num_spr_moves = min(len(modified_tree) // 2, 100)
                    efficient_sprs = False
                    if num_spr_moves == 100:
                        # for large trees, we use a more efficient version of make_worse_spr
                        # that doesn't check the parsimony score for each move
                        efficient_sprs = True
                    new_tree = make_worse_spr(modified_tree, num_spr_moves, efficient_sprs)
                    # Maximum of 100 SPR moves per iteration -- if we don't have enough non-MP
                    # edges after that, we add more in next iteration (until done_modifying)
                    num_spr_moves = min(len(modified_tree) // 2, 100)
                    efficient_sprs = False
                    if num_spr_moves == 100:
                        # for large trees, we use a more efficient version of make_worse_spr
                        # that doesn't check the parsimony score for each move
                        efficient_sprs = True
                    new_tree = make_worse_spr(modified_tree, num_spr_moves, efficient_sprs)
                else:
                    new_tree = make_worse_tree(modified_tree, td // 2)
                if new_tree is not None:
                    modified_tree = new_tree
                # assign edge labels
                edge_labels = assign_edge_labels(modified_tree, tree, dag_clades)
                print("Fraction of non-MP edges (of all edges incl pendant): ", sum(edge_labels)/len(edge_labels))
                if sum(edge_labels) / len(edge_labels) >= 1 / 6 or i > 100:
                    # note that len(edge_labels) is roughly 2*internal edges, so we are aiming at a third of non-MP edges here
                    done_modifying = True
            tree_to_label_dict[modified_tree] = edge_labels
    if len(tree_to_label_dict) < num_trees:
        print("Produced ", len(tree_to_label_dict), " trees instead of ", num_trees)
    return tree_to_label_dict


def memory_safe_count_topologies(dag, max_time=10):
    """Count topologies with a timeout, using SIGKILL if needed."""
    
    # Create a queue for communication between processes
    result_queue = multiprocessing.Queue()
    
    # Define a function to put the result in the queue
    def count_and_return():
        try:
            count = dag.count_topologies()
            result_queue.put(count)
        except Exception as e:
            result_queue.put(f"Error: {str(e)}")
    # Start a separate process
    p = multiprocessing.Process(target=count_and_return)
    p.start()
    # Allow the process to run for max_time seconds
    p.join(timeout=max_time)
    # Check if process is still running after timeout
    if p.is_alive():
        print("Stop counting topologies, returning inf: Function timed out")
        # Use SIGKILL to forcefully terminate the process
        try:
            os.kill(p.pid, signal.SIGKILL)
        except Exception as e:
            print(f"Error killing process: {e}")
        return float('inf')
    # Process completed within time limit, get the result
    if not result_queue.empty():
        result = result_queue.get()
        # Check if we got an error
        if isinstance(result, str) and result.startswith("Error"):
            print(f"Stop counting topologies, returning inf: {result}")
            return float('inf')
        return result
    else:
        print("Stop counting topologies, returning inf: No result returned")
        return float('inf')


def extract_data_from_hdag(dag_file, dpvt_data_file, num_children_file, make_worse_tree):
    """
    Extracts dpvt data from a history DAG and saves it to a file.
    Args:
        dag_file (str): Path to the history DAG file.
        dpvt_data_file (str): Path to save the DPVT data.
        num_children_file (str): Path to save the number of children data.
        make_worse_tree (bool): Whether to use make_worse_tree or not.
    """
    print("Start reading DAG")
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
    print("Done reading DAG")
    print("Start trimming DAG")
    dag.trim_optimal_weight()
    print("Done trimming DAG")
    print("Start converting DAG to sequence_dag")
    dag = hdag.sequence_dag.SequenceHistoryDag.from_history_dag(dag)
    print("Done converting DAG to sequence_dag")
    dag.unlabel()
    print("Start counting DAG topologies")
    num_topologies = min(memory_safe_count_topologies(dag), 200)
    print("Number of topologies in DAG:", num_topologies)
    print("Done counting DAG topologies")

    tree_label_dict = get_non_dag_edges(
        dag, num_children_file, num_topologies, make_worse_tree
    )
    with open(dpvt_data_file, "wb") as f:
        pickle.dump(tree_label_dict, f)

