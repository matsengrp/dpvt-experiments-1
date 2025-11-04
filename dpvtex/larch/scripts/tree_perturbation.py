"""Tree perturbation utilities for generating non-MP trees.

This module provides functionality for perturbing phylogenetic trees to create
trees with non-maximum parsimony (non-MP) edges. It includes:
- Random tree generation utilities
- Tree distance and depth calculations
- Subtree replacement perturbation
- SPR (Subtree Pruning and Regrafting) move perturbation
- Various perturbation strategies (uniform, constant, treesearch_mimic)
"""

from ete3 import Tree
import random
import copy
from collections import deque
import itertools
from historydag.parsimony import disambiguate, parsimony_score


# =============================================================================
# Tree utility functions
# =============================================================================


def _get_distance(target, target2, topology_only=False):
    """
    UPDATED VERSION OF ETE3 BUILT-IN get_distance FUNCTION.
    Returns the distance between two nodes. If only one target is
    specified, it returns the distance between the target and the
    current node.
    :argument target: node in ete3 Tree
    :argument target2: node in same ete3 as target
    :argument False topology_only: If set to True, distance will
        refer to the number of nodes between target and target2.
    :returns: branch length distance between target and
        target2. If topology_only flag is True, returns the number
        of nodes between target and target2.
    """
    root = target.get_tree_root()
    ancestor = root.get_common_ancestor(target, target2)
    dist = 0.0
    for n in [target2, target]:
        current = n
        while current != ancestor:
            if topology_only:
                dist += 1
            else:
                dist += current.dist
            current = current.up
    if topology_only and target != target2:
        # counted ancestor once more than needed in while loop
        dist -= 1
    return dist


def populate(
    tree,
    size,
    names_library=None,
    reuse_names=False,
    seed=None,
    model="fast",
    ladderize=True,
    random_branches=False,
    branch_range=(0, 1),
    support_range=(0, 1),
):
    """
    Generates a random topology by populating current node.

    :argument tree: ete3.Tree, the tree instance

    :argument size: int, number of leaves to add under current node

    :argument None names_library: If provided, names library (list, set, dict, etc.)
        will be used to name nodes.

    :argument False reuse_names: If True, node names will not be necessarily unique,
        which makes the process a bit more efficient.

    :argument None seed: If provided, seed for random number generator

    :argument "fast" distribution: Determines the algorithm used to place leaves, which
        controls the resulting distribution over possible topologies. Parameter can be
        "fast" (original implementation), "yule", or "pda" aka "uniform".
        "fast": newly added leaves are stored in a deque (two-sided linked list), in
            each step a leaf is chosen from one end randomly, and the chosen leaf grows
            two new children.
        "yule": newly added leaves are stored in a list; in each step a leaf is chosen
            randomly from anywhere in the list, and the chosen leaf grows two new
            children. The leaf names are shuffled before assigning names to leaves.
        "uniform" or "pda": newly added leaves and interior nodes are stored in a list;
            in each step a node (interior or tip) is chosen randomly from anywhere in
            the list, and the chosen node grows a new sister leaf. The leaf names are
            shuffled before assigning names to leaves.

    :argument True ladderize: If True, newly populated subtree is ladderized before
        returning.

    :argument False random_branches: If True, branch distances and support values will
        be randomized.

    :argument (0,1) branch_range: If random_branches is True, this range of values will
        be used to generate random distances.

    :argument (0,1) support_range: If random_branches is True, this range of values will
        be used to generate random branch support values.
    """
    NewNode = tree.__class__

    if size <= 0:
        return
    if not reuse_names and names_library is not None:
        if size > len(names_library):
            raise ValueError("not enough names provided in names_library")
    random.seed(seed)

    if len(tree.children) > 1:
        # add `connector` node between current node `self` and its children
        connector = NewNode()
        for ch in tree.get_children():
            ch.detach()
            connector.add_child(child=ch)
        tree.add_child(child=connector)
    if len(tree.children) > 0:
        # add new `root` under `self` where additional nodes will populate
        # a subtree
        root = tree.add_child()
    else:
        root = tree

    if model == "fast":
        new_leaves = deque([root])
        for _ in range(size - 1):
            if random.randint(0, 1):
                p = new_leaves.pop()
            else:
                p = new_leaves.popleft()

            c1 = p.add_child()
            c2 = p.add_child()
            new_leaves.extend([c1, c2])
            if random_branches:
                for c in [c1, c2]:
                    c.dist = random.uniform(*branch_range)
                    c.support = random.uniform(*support_range)
            # else: DEFAULT_DIST and DEFAULT_SUPPORT values will be used
    elif model == "yule":
        if size == 1:
            new_leaves = [root]
        elif size >= 2:
            c1 = root.add_child()
            c2 = root.add_child()
            new_leaves = [c1, c2]
        for _ in range(size - 2):
            # choose random leaf
            prev_leaf = random.choice(new_leaves)

            old_parent = prev_leaf.up
            # new internal node below `old_parent`
            new_parent = old_parent.add_child()
            new_leaf = new_parent.add_child()
            prev_leaf.detach()
            new_parent.add_child(child=prev_leaf)
            new_leaves.append(new_leaf)
            c1, c2 = new_leaf, new_parent
            if random_branches:
                for c in [c1, c2]:
                    c.dist = random.uniform(*branch_range)
                    c.support = random.uniform(*support_range)
    elif model == "pda" or model == "uniform":
        new_leaves = [root]
        new_nodes = [root]
        for _ in range(size - 1):
            # choose random node to add new leaf as sister
            grow_node = random.choice(new_nodes)
            if grow_node != root:
                # `grow_node` has a parent node
                old_parent = grow_node.up
                new_parent = old_parent.add_child()
                grow_node.detach()
                new_parent.add_child(child=grow_node)
                # add child to new_node
                new_leaf = new_parent.add_child()
            else:
                # `grow_node` is the root; sister has no parent
                new_parent = NewNode()
                if grow_node.is_leaf():
                    new_leaves.remove(grow_node)
                    new_leaves.append(new_parent)
                for child in grow_node.get_children():
                    child.detach()
                    new_parent.add_child(child=child)
                grow_node.add_child(child=new_parent)
                new_leaf = grow_node.add_child()

            # add new node, leaf to `new_nodes`, `new_leaves`
            new_leaves.append(new_leaf)
            new_nodes.extend([new_parent, new_leaf])
            if random_branches:
                for c in [new_parent, new_leaf]:
                    c.dist = random.uniform(*branch_range)
                    c.support = random.uniform(*support_range)
    else:
        raise ValueError(f"parameter topology={model} not recognized")
    if ladderize:
        root.ladderize()

    # assign names to leaf nodes in `new_leaves`
    if names_library is not None:
        names_library = deque(names_library)
    else:
        charset = "abcdefghijklmnopqrstuvwxyz"
        avail_names = itertools.combinations_with_replacement(charset, 10)
    if model != "fast":
        # shuffle `new_leaves` in random order
        random.shuffle(new_leaves)
    for n in new_leaves:
        if names_library is not None:
            # choose next name
            tname = names_library.popleft()
            if reuse_names:
                names_library.append(tname)
        else:
            tname = "".join(next(avail_names))
        n.name = tname


def tree_depth(node):
<<<<<<< HEAD
    """Calculate the depth of a tree from the given node.

    Args:
        node: ete3 tree node to use as the root.

    Returns:
        int: Number of nodes along the longest path from node to a leaf.
=======
    """
    Returns the depth of the tree with the given node as the root. This depth is
    the number of nodes along the longest path from the given node to a leaf
    node.
>>>>>>> 423d3be (refactoring)
    """
    if node.is_leaf():
        return 1
    tree_depth = max(
        [_get_distance(node, leaf, topology_only=True) + 2 for leaf in node]
    )
    return tree_depth


def _edge_distance(node1, node2):
<<<<<<< HEAD
    """Calculate the number of edges between two nodes in a tree.

    Args:
        node1: First ete3 tree node.
        node2: Second ete3 tree node.

    Returns:
        int: Number of edges on the path between node1 and node2.
    """
=======
    """Returns the number of edges between the nodes."""
>>>>>>> 423d3be (refactoring)
    return _get_distance(node1, node2, topology_only=True) + 1


def sankoff_for_missing_sequences(tree):
    """
    Fills in the sequence attribute of the nodes in tree where the attribute is
    missing. Sequences are determined by the Sankoff algorithm and nodes are
    modified in-place.
    """
    if tree is None:
        return None
    some_leaf = tree.get_leaves()[0]
    if not hasattr(some_leaf, "sequence"):
        raise ValueError("Set leaf sequences first.")
    site_count = len(some_leaf.sequence)
    fake_sequence = "N" * site_count

    # The next few lines are a silly python trick. This is the current fast way
    # to apply a function to items in an iterable, where the function modifies
    # the items in-place and returns None. Usually faster than a for-loop.
    any(
        node.add_feature("sequence", fake_sequence)
        for node in tree.traverse(strategy="preorder")
        if not hasattr(node, "sequence")
    )

    disambiguate(tree, use_internal_node_sequences=True)
    return None


def root_and_outgroup_leaf(tree, leaf):
<<<<<<< HEAD
    """Re-root tree by setting given leaf as root.

    The leaf becomes the root node, with its only child being the previous root.
    The leaf's name and sequence are copied to the new root.

    Args:
        tree: ete3 tree to re-root.
        leaf: Leaf node in the tree to use as the new root.
=======
    """
    Re-root tree by setting given leaf as root with its only child being the
    previous root. Args:
        tree: ete3 tree leaf: leaf in given tree
>>>>>>> 423d3be (refactoring)
    """
    tree.set_outgroup(leaf)
    leaf.detach()
    tree.name = leaf.name
    tree.sequence = leaf.sequence


def create_random_tree_on_same_leaf_set(tree):
    """
    Create a random tree on the leaf set of the given tree.
    Args:
        tree: ete3 tree
    Returns:
        ete3 tree with random topology on the same leaf set with the same leaf
        sequences as the input tree
    """
    leaf_names = [leaf.name for leaf in tree.get_leaves()]
    new_tree = Tree()
    populate(new_tree, len(leaf_names), model="uniform")
    leaf_names = tree.get_leaf_names()
    for leaf in new_tree.get_leaves():
        # copy sequences from original tree
        leaf.name = leaf_names.pop(random.randint(0, len(leaf_names) - 1))
        leaf.sequence = tree.get_leaves_by_name(leaf.name)[0].sequence
    # re-root tree on first leaf
    outgroup_leaf = new_tree.search_nodes(name=tree.get_leaves()[0].name)[0]
    root_and_outgroup_leaf(new_tree, outgroup_leaf)
    sankoff_for_missing_sequences(new_tree)
    return new_tree


# =============================================================================
# Subtree perturbation functions
# =============================================================================


def _is_subtree_depth_tip(root_node, other_node, depth):
    """
    Returns the truth value of other_node being a tip of the subtree at
    root_node of the given depth. The tips are either exactly depth nodes away
    from root_node or are leaf nodes that are fewer than depth nodes away. This
    method assumes other_node is a descendent of root_node.
    """
    distance = _edge_distance(root_node, other_node)
    return (depth == distance) or (depth > distance and other_node.is_leaf())


def _make_random_tree(tip_subtrees):
    """
    Returns an ete3 Tree with the entries of tip_subtrees placed at the leaves.
    The tree is randomly generated by the utils.py-defined populate method with
    'uniform' model. The randomly generated tree is made with len(tip_subtrees)
    leaf nodes, which are then replaced with with the nodes of tip_subtrees. The
    non-root nodes of the tree are marked with the node attribute
    random_tree=True.
    """
    leaf_count = len(tip_subtrees)

    # Handle edge case: if there's only one tip, just return it
    if leaf_count == 1:
        tip_subtrees[0].add_feature("random_tree", False)
        return tip_subtrees[0]

    tree = Tree()
    populate(tree, leaf_count, model="uniform")
    for old_leaf, new_leaf in zip(tree.get_leaves(), tip_subtrees):
        assert old_leaf.up is not None
        old_leaf.up.add_child(new_leaf)
        old_leaf.delete(prevent_nondicotomic=False)
    any(node.add_feature("random_tree", True) for node in tree.get_descendants())
    tree.add_feature("random_tree", False)
    return tree


def perturb_tree(tree, depth, skip_root=True, exception_on_fail=False):
    """
    Returns a new ete3 Tree instance built from the input tree with a subtree of
    the specified depth replaced with a random tree, where "depth" means
    distance to farthest leaf. The subtree is selected uniformly from the
    subtrees of required depth that consist of an internal node as the root and
    all nodes below this root to the specified depth. Leaf nodes of this subtree
    are unaltered.

    Parameters:
        tree (ete3.Tree): The input tree. depth (int): The depth of the subtree
        to replace with a random tree. skip_root (bool): When True, the subtree
        does not begin at the root of the tree. exception_on_fail (bool): When
        True, the method raises a ValueError exception
            when the tree does not contain a subtree of the given depth.
            Otherwise, the method returns None in such cases.
    """
    if tree is None:
        if exception_on_fail:
            raise ValueError("Tree cannot be None.")
        else:
            return None
    if depth < 1:
        raise ValueError("Depth must be at least 1.")

    tree = tree.copy()
    any(node.add_feature("random_tree", False) for node in tree.traverse())
    valid_nodes = [
        node
        for node in tree.traverse(strategy="preorder")
        if not (skip_root and node.is_root()) and 0 < depth <= tree_depth(node)
    ]
    n = len(valid_nodes)
    if n == 0:
        if exception_on_fail:
            raise ValueError("Input tree has no subtree of required depth.")
        else:
            return None

    selected_node = random.choice(valid_nodes)
    parent_node = None if selected_node.is_root() else selected_node.up
    selected_node.detach()
    is_sorta_tip = lambda x: _is_subtree_depth_tip(selected_node, x, depth)
    sorta_tips = [n.detach() for n in selected_node.get_leaves(is_leaf_fn=is_sorta_tip)]
    new_subtree = _make_random_tree(sorta_tips)

    if parent_node is None:
        tree = new_subtree
    else:
        parent_node.add_child(new_subtree)

    return tree


def increase_tree_parsimony(tree, depth, max_attempts=100):
    """
    Attempt to replace a random subtree of the given depth of the tree, with the
    new tree having a higher parsimony score (less parsimonious).

    This function perturbs a tree to introduce non-maximum parsimony edges,
    which is useful for generating training data to distinguish MP from non-MP edges.

    Args:
        tree: ete3.Tree to perturb
        depth: Depth of the subtree to replace
        max_attempts: Maximum number of attempts to find a tree with higher parsimony

    Returns:
        ete3.Tree with higher parsimony score, or None if no such tree found
    """
    old_score = parsimony_score(tree)
    perturbed_tree = None
    for _ in range(max_attempts):
        perturbed_tree = perturb_tree(tree, depth, exception_on_fail=False)
        if perturbed_tree is None:
            return None
        sankoff_for_missing_sequences(perturbed_tree)
        if parsimony_score(perturbed_tree) > old_score:
            return perturbed_tree
    return None


# =============================================================================
# SPR (Subtree Pruning and Regrafting) functions
# =============================================================================


def spr_move(tree, node1, node2):
    """
    SPR move on tree that prunes rootward edge incident to node1 and regrafts it
    on rootward edge incident to node2. Returns a new tree that is SPR neighbor
    of `tree`
    """

    # deep copy tree and find node1 and node2 in new tree
    new_tree = copy.deepcopy(tree)
    node1_set = node1.get_leaf_names()
    node2_set = node2.get_leaf_names()
    if len(node1_set) > 1:
        node1 = new_tree.get_common_ancestor(node1_set)
    else:
        node1 = new_tree.search_nodes(name=node1_set[0])[0]

    if len(node2_set) > 1:
        node2 = new_tree.get_common_ancestor(node2_set)
    else:
        node2 = new_tree.search_nodes(name=node2_set[0])[0]

    if node2 in node1.get_descendants() or node2 in node1.get_sisters():
        raise ValueError("No SPR move possible, node2 is child of node1")

    # prune node1 and reattach on edge above node2
    node1_parent = node1.up
    node1_sibling = node1.get_sisters()[0]  # bifurcating tree -> only one sibling
    pruned_tree = node1.detach()
    node1_parent.up.add_child(node1_sibling)
    node1_parent.detach()

    node2_p = node2.up
    new_node = node2_p.add_child()
    node2.detach()
    new_node.add_child(node2)
    new_node.add_child(pruned_tree)
    new_node.add_feature("random_tree", True)
    return new_tree


<<<<<<< HEAD
def make_worse_spr(input_tree, max_sprs, efficient=True):
    """
    Peform at least max_sprs random SPR moves on input tree to create tree
=======
def make_worse_spr(input_tree, num_sprs, efficient=True):
    """
    Peform a at least num_sprs random SPR move on input tree to create tree
>>>>>>> 423d3be (refactoring)
    with higher parsimony score.
    If the keyword efficient is set to True, the function will not check
    parsimony score of the new tree. This is faster, but the resulting tree
    may not have a higher parsimony score.
    Arguments:
        tree: ete3.Tree
            Input tree to be perturbed
<<<<<<< HEAD
        max_sprs: int
=======
        num_sprs: int
>>>>>>> 423d3be (refactoring)
            Number of SPR moves to perform
        efficient: bool
            If True, perform moves without checking parsimony
    """
    tree = copy.deepcopy(input_tree)
    sankoff_for_missing_sequences(tree)
    any(node.add_feature("random_tree", False) for node in tree.traverse())
<<<<<<< HEAD
    for i in range(max_sprs):
=======
    for i in range(num_sprs):
>>>>>>> 423d3be (refactoring)
        random.seed()
        # we cannot prune children or grandchildren of root
        # also avoid moving leaves, as that doesn't change MP edge to non-MP edge
        node_list = list(
            [
                node
                for node in tree.iter_descendants()
                if not (
                    node.up.is_root()
                    or node.up in tree.children
                    or node in tree.get_leaves()
                )
            ]
        )
        # prune edge above randomly chosen node prune_node
        if len(node_list) == 0:
            # No valid nodes to prune, skip this iteration
            continue
        prune_node = random.choice(node_list)
        # pick random edge to insert -- cannot be in pruned subtree
        allowed_edges = [
            node
            for node in tree.iter_descendants()
            if node not in list(prune_node.traverse())
            and not node.up == prune_node.up
            and not node == prune_node.up
        ]
        if len(allowed_edges) > 0:
            insertion_node = random.choice(allowed_edges)
            perturbed_tree = spr_move(tree, prune_node, insertion_node)
            sankoff_for_missing_sequences(perturbed_tree)
            if efficient or (
                not efficient
                and parsimony_score(perturbed_tree) > parsimony_score(tree)
            ):
                tree = perturbed_tree
    if not efficient and parsimony_score(tree) <= parsimony_score(input_tree):
        print("No worse tree found")
        return None
    return tree


# =============================================================================
# Perturbation strategy configuration
# =============================================================================


def calculate_spr_count_uniform(num_leaves, max_spr_moves):
    """Calculate number of SPR moves for uniform distribution strategy.

    Args:
        num_leaves: Number of leaves in the tree
        max_spr_moves: Maximum number of SPR moves

    Returns:
        int: Number of SPR moves drawn from uniform distribution [0, min(num_leaves, max_spr_moves)]
    """
    max_num_spr_moves = min(num_leaves, max_spr_moves)
    return random.randint(0, max_num_spr_moves)


def calculate_spr_count_constant(num_leaves, max_spr_moves, spr_move_divisor):
    """Calculate number of SPR moves for constant distribution strategy.

    Args:
        num_leaves: Number of leaves in the tree
        max_spr_moves: Maximum number of SPR moves
        spr_move_divisor: Divisor for calculating SPR moves

    Returns:
        int: Number of SPR moves (num_leaves // spr_move_divisor, capped at max_spr_moves)
    """
    return min(num_leaves // spr_move_divisor, max_spr_moves)


<<<<<<< HEAD
def calculate_spr_count_treesearch_mimic(
    num_leaves, tree_index, total_trees, max_spr_moves
):
=======
def calculate_spr_count_treesearch_mimic(num_leaves, tree_index, total_trees, max_spr_moves):
>>>>>>> 423d3be (refactoring)
    """Calculate number of SPR moves for treesearch_mimic distribution strategy.

    For the first half of trees, uses min(num_leaves, max_spr_moves) SPR moves.
    For the second half, draws from uniform distribution.

    Args:
        num_leaves: Number of leaves in the tree
        tree_index: Index of current tree being processed
        total_trees: Total number of MP trees
        max_spr_moves: Maximum number of SPR moves

    Returns:
        int: Number of SPR moves
    """
    if tree_index < total_trees // 2:
        return min(num_leaves, max_spr_moves)
    else:
        return calculate_spr_count_uniform(num_leaves, max_spr_moves)


<<<<<<< HEAD
def determine_spr_count(
    edge_distribution,
    num_leaves,
    tree_index,
    total_trees,
    max_spr_moves,
    spr_move_divisor,
):
=======
def determine_spr_count(edge_distribution, num_leaves, tree_index, total_trees, max_spr_moves, spr_move_divisor):
>>>>>>> 423d3be (refactoring)
    """Determine the number of SPR moves based on edge distribution strategy.

    Args:
        edge_distribution: Strategy name ("uniform", "constant", or "treesearch_mimic")
        num_leaves: Number of leaves in the tree
        tree_index: Index of current tree being processed (for treesearch_mimic)
        total_trees: Total number of MP trees (for treesearch_mimic)
        max_spr_moves: Maximum number of SPR moves
        spr_move_divisor: Divisor for constant distribution

    Returns:
        int: Number of SPR moves to perform
    """
    if edge_distribution == "uniform":
        return calculate_spr_count_uniform(num_leaves, max_spr_moves)
    elif edge_distribution == "constant":
        return calculate_spr_count_constant(num_leaves, max_spr_moves, spr_move_divisor)
    elif edge_distribution == "treesearch_mimic":
        return calculate_spr_count_treesearch_mimic(
            num_leaves, tree_index, total_trees, max_spr_moves
        )
    else:
        raise ValueError(f"Unknown edge distribution: {edge_distribution}")


<<<<<<< HEAD
def perturb_tree_with_spr(
    tree, edge_distribution, tree_index, total_trees, max_spr_moves, spr_move_divisor
):
=======
def perturb_tree_with_spr(tree, edge_distribution, tree_index, total_trees, max_spr_moves, spr_move_divisor):
>>>>>>> 423d3be (refactoring)
    """Perturb a tree using SPR moves to introduce non-MP edges.

    Args:
        tree: ete3 Tree to perturb
        edge_distribution: Strategy for determining number of SPR moves
        tree_index: Index of current tree being processed
        total_trees: Total number of MP trees
        max_spr_moves: Maximum number of SPR moves
        spr_move_divisor: Divisor for constant distribution

    Returns:
        ete3.Tree: Perturbed tree, or original tree if perturbation fails
    """
    num_leaves = len(tree.get_leaves())
    num_spr_moves = determine_spr_count(
<<<<<<< HEAD
        edge_distribution,
        num_leaves,
        tree_index,
        total_trees,
        max_spr_moves,
        spr_move_divisor,
=======
        edge_distribution, num_leaves, tree_index, total_trees, max_spr_moves, spr_move_divisor
>>>>>>> 423d3be (refactoring)
    )

    print(f"Number of SPR moves: {num_spr_moves}")

    # Use efficient SPRs (without parsimony checking) for large numbers of moves
    efficient_sprs = num_spr_moves >= max_spr_moves
    if efficient_sprs:
        print("Using efficient SPRs")

    perturbed_tree = make_worse_spr(tree, num_spr_moves, efficient_sprs)

    if perturbed_tree is None:
        print("Cannot get non-MP edges, keeping original tree")
        return tree

    return perturbed_tree


<<<<<<< HEAD
def perturb_tree_with_subtree_replacement(
    tree,
    original_tree,
    dag_clades,
    subtree_max_attempts,
    subtree_target_non_mp_proportion,
):
=======
def perturb_tree_with_subtree_replacement(tree, original_tree, dag_clades, max_perturbation_attempts, target_non_mp_proportion):
>>>>>>> 423d3be (refactoring)
    """Perturb a tree by replacing random subtrees until target proportion of non-MP edges.

    Args:
        tree: ete3 Tree to perturb
        original_tree: Original unperturbed tree for edge label comparison
        dag_clades: Clades extracted from the hDAG
<<<<<<< HEAD
        subtree_max_attempts: Maximum number of subtree replacement attempts
        subtree_target_non_mp_proportion: Target proportion of non-MP edges
=======
        max_perturbation_attempts: Maximum number of perturbation attempts
        target_non_mp_proportion: Target proportion of non-MP edges
>>>>>>> 423d3be (refactoring)

    Returns:
        tuple: (perturbed_tree, edge_labels)
    """
    # Import here to avoid circular dependency
    from dpvtex.larch.scripts.extract_data_from_hdag import assign_edge_labels

    tree_d = tree_depth(tree)
    depth = int(tree_d // 2)
    modified_tree = tree.copy()

<<<<<<< HEAD
    for attempt in range(subtree_max_attempts):
=======
    for attempt in range(max_perturbation_attempts):
>>>>>>> 423d3be (refactoring)
        perturbed = increase_tree_parsimony(modified_tree, depth)
        if perturbed is not None:
            modified_tree = perturbed

        edge_labels = assign_edge_labels(modified_tree, original_tree, dag_clades)
        non_mp_proportion = sum(edge_labels) / len(edge_labels)

<<<<<<< HEAD
        if non_mp_proportion >= subtree_target_non_mp_proportion:
=======
        if non_mp_proportion >= target_non_mp_proportion:
>>>>>>> 423d3be (refactoring)
            # Target achieved: ~1/6 of edges are non-MP
            # (len(edge_labels) ≈ 2 * internal edges, so we're aiming for ~1/3 non-MP internal edges)
            break

    return modified_tree, edge_labels


def prepare_tree_for_perturbation(tree, num_children_file):
    """Prepare a tree for perturbation by cleaning and rooting it.

    Args:
        tree: ete3 Tree to prepare
        num_children_file: File handle to write number of children per node

    Returns:
        ete3.Tree: Prepared tree
    """
    # Remove sequences from internal nodes
    for node in tree.traverse():
        if not node.is_leaf():
            node.del_feature("sequence")

    # Root the tree with a random leaf as outgroup
    root_leaf = tree.get_leaves()[0]
    root_and_outgroup_leaf(tree, root_leaf)

    # Record tree structure (number of children per node)
    num_children_list = [
        len(node.get_children()) for node in tree.traverse() if not node.is_leaf()
    ]
    num_children_file.write(",".join(str(n) for n in num_children_list) + "\n")

    # Make tree binary and fill in internal sequences using Sankoff
    tree.resolve_polytomy()
    sankoff_for_missing_sequences(tree)

    return tree


def generate_random_trees_for_treesearch(mp_trees, dag_clades):
    """Generate random trees for the treesearch_mimic strategy.

    Args:
        mp_trees: List of maximum parsimony trees
        dag_clades: Clades extracted from the hDAG

    Returns:
        dict: Dictionary mapping random trees to their edge labels
    """
    # Import here to avoid circular dependency
    from dpvtex.larch.scripts.extract_data_from_hdag import assign_edge_labels

    tree_to_label_dict = {}
    num_random_trees = len(mp_trees)

    print("Start adding random trees for treesearch_mimic...")
    for i in range(num_random_trees):
        print(f"Adding random tree number {i}")
        random_tree = create_random_tree_on_same_leaf_set(mp_trees[0])
        edge_labels = assign_edge_labels(random_tree, mp_trees[0], dag_clades)
        tree_to_label_dict[random_tree] = edge_labels
    print("Done adding random trees for treesearch_mimic...")

    return tree_to_label_dict
