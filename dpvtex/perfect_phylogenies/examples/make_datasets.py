import pickle
import torch
from multiprocessing import Pool
from ete3 import Tree
import argparse
from sklearn.model_selection import train_test_split

from dpvtex.perfect_phylogenies.utils import populate
from dpvtex.perfect_phylogenies.random_perfect_phylogeny import RandomPerfectPhylogeny
from dpvtex.perfect_phylogenies.perturb_phylogeny import (
    perturb_tree,
    make_worse_tree,
    make_worse_spr,
    sankoff_for_missing_sequences,
)


def make_phylogeny_data(n_leaves, n_phylos_per_tree, depth, spr=False):
    """
    Create a collection of phylogenies obtained by randomly mixing a subtree of
    a perfect phylogeny.
    """
    # create a random tree
    tree = Tree()
    populate(tree, n_leaves, model="uniform")
    rpp = RandomPerfectPhylogeny(tree)
    data_pairs = []

    for _ in range(n_phylos_per_tree):
        # create a random prefect phylogeny
        phylo = rpp.make_random_perfect_phylogeny()
        if spr:
            mixed_phylo = make_worse_spr(phylo, len(tree) / 2, max_attempts=500)
        else:
            mixed_phylo = make_worse_tree(phylo, depth=depth, max_attempts=500)
        if mixed_phylo is None:
            mixed_phylo = perturb_tree(phylo, depth=depth, exception_on_fail=True)
            # generate sequences on internal nodes
            sankoff_for_missing_sequences(mixed_phylo)

        # add "extra" unifurcating root above previous root
        new_tree = Tree()
        new_tree.add_child(mixed_phylo)
        new_tree.sequence = mixed_phylo.sequence
        new_tree.random_tree = mixed_phylo.random_tree
        mixed_phylo = new_tree
        edge_classifier = [
            1.0 if (node.random_tree and not node.is_leaf()) else 0.0
            for node in mixed_phylo.traverse(strategy="preorder")
        ]
        data_pairs.append((mixed_phylo, edge_classifier))
    return data_pairs


def create_training_data(
    n_trees, n_phylos_per_tree, n_leaves, depth=4, n_threads=1, spr=False
):
    """
    Create a collection of random trees and phylogenies obtained by randomly
    mixing a subtree of a perfect phylogeny.

    """
    if n_threads > 1:
        process_pool = Pool(n_threads)
        pooled_results = process_pool.starmap(
            make_phylogeny_data, [(n_leaves, n_phylos_per_tree, depth, spr)] * n_trees
        )
    else:
        # Skip the performance hit from using multiprocessing with one process.
        pooled_results = [
            make_phylogeny_data(n_leaves, n_phylos_per_tree, depth, spr)
            for _ in range(n_trees)
        ]
    tree_data_dict = {
        mixed_phylo: edge_classifier
        for paired_data in pooled_results
        for mixed_phylo, edge_classifier in paired_data
    }
    return tree_data_dict


def main():
    # Create argument parser - requires number of leaves as input!
    parser = argparse.ArgumentParser(
        description="Create perfect phylogenies with some non-MP edges for the provided number of leaves"
    )
    parser.add_argument("n_leaves", type=int, help="Number of leaves")
    parser.add_argument("n_trees", type=int, help="Number of trees to be generated")
    parser.add_argument(
        "SPR",
        type=str,
        help="SPR moves for creating sub-optimal edges? If True -> SPR, otherwise subtree replacement.",
    )

    # Parse arguments
    args = parser.parse_args()
    spr = False
    if args.SPR == "True":
        spr = True
    data_dict = create_training_data(
        n_trees=args.n_trees,
        n_phylos_per_tree=1,
        n_leaves=args.n_leaves,
        depth=2,
        n_threads=4,
        spr=spr,
    )

    file_path = f"{args.n_leaves}leaf_perfect_{args.n_trees}_distinct_trees.p"
    if spr:
        file_path = file_path[:-2] + "_spr.p"

    with open(file_path, "wb") as fh:
        pickle.dump(data_dict, fh)


if __name__ == "__main__":
    main()
