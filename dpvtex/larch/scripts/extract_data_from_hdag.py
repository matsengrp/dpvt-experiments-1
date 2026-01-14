import historydag as hdag
import multiprocessing
import os
import pickle
import random
import signal
import sys

from ete3 import Tree

from dpvtex.larch.scripts.pipeline_logger import get_logger
from dpvtex.larch.scripts.utils import get_alignment_name_from_path
from dpvtex.larch.scripts.tree_perturbation import (
    increase_tree_parsimony,
    make_worse_spr,
    tree_depth,
    sankoff_for_missing_sequences,
    populate,
    root_and_outgroup_leaf,
    create_random_tree_on_same_leaf_set,
    perturb_tree_with_spr_target,
    perturb_tree_with_subtree_replacement,
    prepare_tree_for_perturbation,
    generate_random_trees_for_treesearch,
)

# Constants
_TOPOLOGY_COUNT_TIMEOUT_SECONDS = 10
_ASSUMED_MIN_DAG_TOPOLOGIES = 1000  # Assumed minimum for DAGs with inf topologies


def _create_empty_output_files(dpvt_data_file, num_children_file):
    """Create empty output files to allow pipeline to continue on failures."""
    with open(dpvt_data_file, "wb") as f:
        pickle.dump({}, f)
    with open(num_children_file, "w") as f:
        f.write("")


def _load_dag_from_file(dag_file):
    """Load a DAG from file (pickle or protobuf format).

    Args:
        dag_file: Path to DAG file (.p for pickle, .pb for protobuf)

    Returns:
        The loaded DAG object

    Raises:
        ValueError: If file format is not recognized
        Various exceptions from pickle/protobuf loading
    """
    if dag_file.endswith(".p"):
        with open(dag_file, "rb") as f:
            return pickle.load(f)
    elif dag_file.endswith(".pb") or dag_file.endswith(".pb.gz"):
        return hdag.mutation_annotated_dag.load_MAD_protobuf_file(
            dag_file, compact_genomes=True
        )
    else:
        raise ValueError(
            f"Unrecognized DAG file format: {dag_file}. Expected .p or .pb/.pb.gz"
        )


def _prepare_dag_for_extraction(dag):
    """Prepare a DAG for tree extraction.

    Trims to optimal weight topologies and converts to sequence_dag.

    Args:
        dag: A historydag object

    Returns:
        A prepared SequenceHistoryDag
    """
    dag.trim_optimal_weight()
    dag = hdag.sequence_dag.SequenceHistoryDag.from_history_dag(dag)
    dag.unlabel()
    return dag


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
        # DAG has too many topologies to count; assume at least this many exist
        sample_ids = random.sample(range(_ASSUMED_MIN_DAG_TOPOLOGIES), num_samples)
    else:
        sample_ids = random.sample(range(dag_num_topologies), num_samples)

    tree_samples = [
        dag[i].to_ete(name_func=lambda n: n.label.node_id, features=["sequence"])
        for i in sample_ids
    ]
    return tree_samples


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


def _is_clade_supported_by_dag(clade, dag_clades):
    """Check if a clade is supported by the DAG (directly or as multifurcation resolution).

    A clade is supported if it either:
    1. Exists directly in dag_clades, or
    2. Is a resolution of a multifurcation (union of child clades of some DAG clade)

    Args:
        clade: frozenset of leaf names representing the clade
        dag_clades: dict mapping clades to their child clades

    Returns:
        True if clade is supported by the DAG, False otherwise
    """
    # Direct match
    if clade in dag_clades:
        return True

    # Check if clade is a resolution of a multifurcation
    for dag_clade in dag_clades:
        if clade.issubset(dag_clade):
            if exists_subset_union(dag_clades[dag_clade], clade):
                return True

    return False


def assign_edge_labels(modified_tree, tree, dag_clades):
    """
    Assigns label 0/1 to modified_tree edges based on DAG support.

    Labels are 0 for MP edges (supported by DAG) and 1 for non-MP edges.
    The original tree is used to efficiently initialize labels, since most
    edges are shared between the original and modified trees.

    Args:
        modified_tree: ete3 tree to label
        tree: original ete3 tree (before perturbation) for efficient initialization
        dag_clades: dict mapping clades to child clades (from extract_hdag_clade_child_clades)

    Returns:
        list of 0/1 labels in preorder traversal (0=MP edge, 1=non-MP edge)
    """
    # Initialize labels: 0 if edge exists in original tree, 1 otherwise
    tree_clades = {
        frozenset(node.get_leaf_names()) for node in tree.traverse("preorder")
    }
    leaf_set = frozenset(modified_tree.get_leaf_names())

    edge_labels = [
        0 if frozenset(node.get_leaf_names()) in tree_clades else 1
        for node in modified_tree.traverse("preorder")
    ]

    # Update labels: check if edges marked as non-MP are actually supported by DAG
    for i, node in enumerate(modified_tree.traverse("preorder")):
        # Root and first child are always labeled 0 (masked in training)
        if i < 2:
            edge_labels[i] = 0
            continue

        if edge_labels[i] == 1:
            this_clade = frozenset(node.get_leaf_names())
            complement_clade = leaf_set - this_clade

            # Check if either the clade or its complement is supported
            if _is_clade_supported_by_dag(
                this_clade, dag_clades
            ) or _is_clade_supported_by_dag(complement_clade, dag_clades):
                edge_labels[i] = 0

    return edge_labels


def generate_perturbed_trees_with_labels(
    dag,
    num_children_file,
    max_trees=0,
    edge_distribution="constant",
    # SPR parameters
    spr_radius=None,
    spr_target_non_mp_proportion=0.1,
    max_spr_attempts=100,
    # Subtree parameters
    subtree_max_attempts=100,
    subtree_target_non_mp_proportion=1 / 6,
):
    """
    Generate perturbed trees with MP/non-MP edge labels from a history DAG.

    Args:
        dag: sequence_dag representing the history DAG
        num_children_file: File path to write number of children per node in each tree
        max_trees: Maximum number of trees to return. Actual count may be less if
            the DAG contains fewer topologies. If 0, returns as many trees as
            there are MP trees in the DAG (capped by available topologies).
        edge_distribution: Strategy for introducing non-MP edges:
            - "constant": target-based SPR with radius control
            - "uniform": target-based SPR with radius control
            - "treesearch_mimic": mimic tree search distribution:
              * 1/2 random trees (most edges non-MP)
              * 1/4 trees with full target proportion
              * 1/4 trees with uniform target in [0, target proportion]
            - "random_subtree": replace random subtree of depth d/2
        spr_radius: Maximum topological distance between prune and regraft locations.
            None means no limit. (default: None)
        spr_target_non_mp_proportion: Target proportion of non-MP edges for SPR
            perturbation (default: 0.1)
        max_spr_attempts: Maximum SPR attempts before stopping (default: 100)
        subtree_max_attempts: Maximum attempts for random_subtree perturbation (default: 100)
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
            elif edge_distribution in ("constant", "uniform"):
                # Use target-based SPR perturbation with radius control
                modified_tree, edge_labels = perturb_tree_with_spr_target(
                    tree,
                    tree,
                    dag_clades,
                    spr_radius,
                    spr_target_non_mp_proportion,
                    max_spr_attempts,
                )
            elif edge_distribution == "treesearch_mimic":
                # treesearch_mimic: use target-based SPR with variable target
                # First half: use full target proportion
                # Second half: draw uniformly from [0, target proportion]
                if index < len(mp_trees) // 2:
                    target_proportion = spr_target_non_mp_proportion
                else:
                    target_proportion = random.uniform(0, spr_target_non_mp_proportion)
                modified_tree, edge_labels = perturb_tree_with_spr_target(
                    tree,
                    tree,
                    dag_clades,
                    spr_radius,
                    target_proportion,
                    max_spr_attempts,
                )
            else:
                raise ValueError(f"Unknown edge distribution: {edge_distribution}")

            tree_to_label_dict[modified_tree] = edge_labels

    if len(tree_to_label_dict) < max_trees:
        print(f"Produced {len(tree_to_label_dict)} trees (requested max: {max_trees})")

    return tree_to_label_dict


def memory_safe_count_topologies(dag, max_time=_TOPOLOGY_COUNT_TIMEOUT_SECONDS):
    """Count topologies in a DAG with timeout protection.

    Runs topology counting in a separate process with a timeout to prevent
    memory exhaustion on large DAGs. Uses SIGKILL to forcefully terminate
    if the count doesn't complete in time.

    Args:
        dag: A historydag object to count topologies in.
        max_time: Maximum seconds to wait before killing the count.

    Returns:
        int or float: Number of topologies, or float('inf') if count timed out
            or encountered an error.
    """
    result_queue = multiprocessing.Queue()

    def count_and_return():
        try:
            count = dag.count_topologies()
            result_queue.put(count)
        except Exception as e:  # Broad catch needed - runs in subprocess
            result_queue.put(f"Error: {str(e)}")

    process = multiprocessing.Process(target=count_and_return)
    process.start()
    process.join(timeout=max_time)

    if process.is_alive():
        print("Stop counting topologies, returning inf: Function timed out")
        try:
            os.kill(process.pid, signal.SIGKILL)
        except OSError as e:
            print(f"Error killing process: {e}")
        return float("inf")

    if not result_queue.empty():
        result = result_queue.get()
        if isinstance(result, str) and result.startswith("Error"):
            print(f"Stop counting topologies, returning inf: {result}")
            return float("inf")
        return result

    print("Stop counting topologies, returning inf: No result returned")
    return float("inf")


def extract_data_from_hdag(
    dag_file,
    dpvt_data_file,
    num_children_file,
    edge_distribution="constant",
    logger=None,
    max_trees=200,
    # SPR parameters
    spr_radius=None,
    spr_target_non_mp_proportion=0.1,
    max_spr_attempts=100,
    # Subtree parameters
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
        spr_radius (int or None): Maximum topological distance between prune and
            regraft locations. None means no limit. (default: None)
        spr_target_non_mp_proportion (float): Target proportion of non-MP edges
            for SPR perturbation (default: 0.1)
        max_spr_attempts (int): Maximum SPR attempts before stopping (default: 100)
        subtree_max_attempts (int): Max attempts for subtree replacement (default: 100)
        subtree_target_non_mp_proportion (float): Target non-MP edge proportion for
            subtree replacement (default: 1/6)
    """
    alignment_name = get_alignment_name_from_path(dag_file)
    logger.log_section("EXTRACTION", f"Starting tree extraction for {alignment_name}")
    logger.log("EXTRACTION", f"Edge distribution method: {edge_distribution}")

    # Check for empty DAG file (happens when larch times out or fails)
    if os.path.getsize(dag_file) == 0:
        logger.log(
            "EXTRACTION",
            f"Skipping {alignment_name}: Empty DAG file (likely larch timeout/failure)",
        )
        print(f"WARNING: Skipping {alignment_name}: Empty DAG file", file=sys.stderr)
        _create_empty_output_files(dpvt_data_file, num_children_file)
        return

    # Load DAG from file
    logger.log("EXTRACTION", "Reading DAG from file")
    try:
        dag = _load_dag_from_file(dag_file)
    except (ValueError, pickle.UnpicklingError, OSError) as e:
        logger.log("EXTRACTION", f"Error loading DAG for {alignment_name}: {str(e)}")
        print(
            f"WARNING: Skipping {alignment_name}: Invalid/corrupted DAG file",
            file=sys.stderr,
        )
        _create_empty_output_files(dpvt_data_file, num_children_file)
        return

    logger.log("EXTRACTION", "DAG loaded successfully")

    # Prepare DAG for extraction
    logger.log("EXTRACTION", "Preparing DAG (trimming and converting)")
    dag = _prepare_dag_for_extraction(dag)

    # Count topologies
    logger.log(
        "EXTRACTION",
        f"Counting DAG topologies (max {_TOPOLOGY_COUNT_TIMEOUT_SECONDS}s timeout)",
    )
    num_topologies = min(memory_safe_count_topologies(dag), max_trees)
    logger.log(
        "EXTRACTION",
        f"Number of topologies in DAG: {num_topologies} (capped at {max_trees})",
    )

    # Generate perturbed trees with edge labels
    tree_label_dict = generate_perturbed_trees_with_labels(
        dag,
        num_children_file,
        max_trees=num_topologies,
        edge_distribution=edge_distribution,
        spr_radius=spr_radius,
        spr_target_non_mp_proportion=spr_target_non_mp_proportion,
        max_spr_attempts=max_spr_attempts,
        subtree_max_attempts=subtree_max_attempts,
        subtree_target_non_mp_proportion=subtree_target_non_mp_proportion,
    )

    logger.log("EXTRACTION", f"Generated {len(tree_label_dict)} trees with edge labels")

    # Save output
    with open(dpvt_data_file, "wb") as f:
        pickle.dump(tree_label_dict, f)

    logger.log("EXTRACTION", f"Tree data saved to: {dpvt_data_file}")
