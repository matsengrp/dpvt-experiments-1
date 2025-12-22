import historydag as hdag
import pickle
import random
import os
import signal
import multiprocessing
import sys
from ete3 import Tree

from dpvtex.larch.scripts.tree_perturbation import (
    increase_tree_parsimony,
    make_worse_spr,
    tree_depth,
    sankoff_for_missing_sequences,
    populate,
    root_and_outgroup_leaf,
    create_random_tree_on_same_leaf_set,
    perturb_tree_with_spr,
    perturb_tree_with_subtree_replacement,
    prepare_tree_for_perturbation,
    generate_random_trees_for_treesearch,
)


def get_MP_trees_from_hdag(dag, max_trees, unlabel=True):
    """
    Samples up to max_trees uniformly from input historydag dag without replacement.

    Args:
        dag: compact genome historydag without ambiguous sequences
        max_trees: Maximum number of trees to sample. Actual count returned may be
            less if the DAG contains fewer topologies than requested.
        unlabel: if true, we unlabel the DAG and then only sample each topology
            once. This needs to be used with care if the input dag has ambiguous
            sequences.

    Returns:
        list of ete trees (may contain fewer than max_trees if DAG has fewer topologies)
    """
    if unlabel:
        dag = dag.unlabel()
    dag.uniform_distribution_annotate()
    # Only call memory_safe_count_topologies once
    dag_num_topologies = memory_safe_count_topologies(dag)
    num_samples = min(max_trees, dag_num_topologies)
    if num_samples != max_trees:
        print(
            "Not enough trees in DAG to sample",
            max_trees,
            "trees. Sampling",
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
    make the label assignment more efficient.
    Args:
        modified_tree: ete3 tree for which we want to get edge label list
        tree: ete3 tree that is mostly identical to modified tree (tree before
            make_worse) - this can make the label assignment more efficient, but
            can be replaced with a random tree
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


def get_non_dag_edges(
    dag,
    num_children_file,
    max_trees=0,
    edge_distribution="constant",
    max_spr_moves=100,
    subtree_max_attempts=100,
    spr_move_divisor=4,
    subtree_target_non_mp_proportion=1 / 6,
):
    """
    Perturbs trees to create perturbed trees containing edges not present in the given dag.

    Args:
        dag: sequence_dag representing the history DAG
        num_children_file: File path to write number of children per node in each tree
        max_trees: Maximum number of trees to return. Actual count may be less if
            the DAG contains fewer topologies. If 0, returns as many trees as
            there are MP trees in the DAG (capped by available topologies).
        edge_distribution: Strategy for introducing non-MP edges:
            - "constant": perform num_leaves/spr_move_divisor (max max_spr_moves) SPR moves
            - "uniform": draw number of SPR moves from [0, min(num_leaves, max_spr_moves)]
            - "treesearch_mimic": mimic tree search distribution:
              * 1/2 random trees (most edges non-MP)
              * 1/4 trees with min(num_leaves, max_spr_moves) SPR moves
              * 1/4 trees with uniform SPR moves
            - "random_subtree": replace random subtree of depth d/2
        max_spr_moves: Maximum number of SPR moves to perform (default: 100)
        subtree_max_attempts: Maximum attempts for random_subtree perturbation (default: 100)
        spr_move_divisor: Divisor for constant edge distribution (default: 4)
        subtree_target_non_mp_proportion: Target proportion of non-MP edges for random_subtree (default: 1/6)

    Returns:
        dict: Dictionary with ete3 trees as keys and edge label lists as values,
            where labels indicate MP (0) vs non-MP (1) edges in preorder traversal
    """
    # Extract MP trees and clades from hDAG
    print("Start extracting MP trees from hDAG")
    mp_trees = get_MP_trees_from_hdag(dag, max_trees, unlabel=True)
    print(f"Extracted {len(mp_trees)} trees from hDAG")

    print("Start extracting clades from hDAG")
    dag_clades = extract_hdag_clade_child_clades(dag)
    print("Extracted clades from hDAG")

    tree_to_label_dict = {}

    # For treesearch_mimic, first generate random trees
    if edge_distribution == "treesearch_mimic":
        tree_to_label_dict = generate_random_trees_for_treesearch(mp_trees, dag_clades)

    # Process each MP tree and introduce non-MP edges
    print("Start adding non-MP edges...")
    with open(num_children_file, "w") as nc_file:
        for index, tree in enumerate(mp_trees):
            print(f"Processing tree number {index} of {len(mp_trees)}")

            # Prepare tree for perturbation
            tree = prepare_tree_for_perturbation(tree, nc_file)

            # Perturb tree based on edge distribution strategy
            if edge_distribution == "random_subtree":
                modified_tree, edge_labels = perturb_tree_with_subtree_replacement(
                    tree,
                    tree,
                    dag_clades,
                    subtree_max_attempts,
                    subtree_target_non_mp_proportion,
                )
            else:
                # Use SPR-based perturbation
                modified_tree = perturb_tree_with_spr(
                    tree,
                    edge_distribution,
                    index,
                    len(mp_trees),
                    max_spr_moves,
                    spr_move_divisor,
                )
                edge_labels = assign_edge_labels(modified_tree, tree, dag_clades)

            tree_to_label_dict[modified_tree] = edge_labels

    if len(tree_to_label_dict) < max_trees:
        print(f"Produced {len(tree_to_label_dict)} trees (requested max: {max_trees})")

    return tree_to_label_dict


def memory_safe_count_topologies(dag, max_time=10):
    """Count topologies in a DAG with timeout protection.

    Runs topology counting in a separate process with a timeout to prevent
    memory exhaustion on large DAGs. Uses SIGKILL to forcefully terminate
    if the count doesn't complete in time.

    Args:
        dag: A historydag object to count topologies in.
        max_time: Maximum seconds to wait before killing the count (default: 10).

    Returns:
        int or float: Number of topologies, or float('inf') if count timed out
            or encountered an error.
    """
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
        return float("inf")
    # Process completed within time limit, get the result
    if not result_queue.empty():
        result = result_queue.get()
        # Check if we got an error
        if isinstance(result, str) and result.startswith("Error"):
            print(f"Stop counting topologies, returning inf: {result}")
            return float("inf")
        return result
    else:
        print("Stop counting topologies, returning inf: No result returned")
        return float("inf")


def extract_data_from_hdag(
    dag_file,
    dpvt_data_file,
    num_children_file,
    edge_distribution="constant",
    logger=None,
    max_trees=200,
    max_spr_moves=100,
    spr_move_divisor=10,
    subtree_max_attempts=100,
    subtree_target_non_mp_proportion=1 / 6,
):
    """
    Extracts dpvt data from a history DAG and saves it to a file.

    Args:
        dag_file (str): Path to the history DAG file.
        dpvt_data_file (str): Path to save the DPVT data.
        num_children_file (str): Path to save the number of children data.
        edge_distribution (str): Method for introducing non-MP edges. Options:
            "constant", "uniform", "treesearch_mimic", "random_subtree"
        logger (PipelineLogger): Logger for tracking operations
        max_trees (int): Maximum number of trees to extract from DAG (default: 200)
        max_spr_moves (int): Maximum SPR moves per tree (default: 100)
        spr_move_divisor (int): Divisor for constant SPR distribution (default: 10)
        subtree_max_attempts (int): Max attempts for subtree replacement (default: 100)
        subtree_target_non_mp_proportion (float): Target non-MP edge proportion for
            subtree replacement (default: 1/6)
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

    logger.log("EXTRACTION", "Counting DAG topologies (max 10s timeout)")
    num_topologies = min(memory_safe_count_topologies(dag), max_trees)
    logger.log(
        "EXTRACTION",
        f"Number of topologies in DAG: {num_topologies} (capped at {max_trees})",
    )

    tree_label_dict = get_non_dag_edges(
        dag,
        num_children_file,
        max_trees=num_topologies,
        edge_distribution=edge_distribution,
        max_spr_moves=max_spr_moves,
        spr_move_divisor=spr_move_divisor,
        subtree_max_attempts=subtree_max_attempts,
        subtree_target_non_mp_proportion=subtree_target_non_mp_proportion,
    )
    with open(dpvt_data_file, "wb") as f:
        pickle.dump(tree_label_dict, f)
