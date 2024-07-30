import pickle
import torch
from multiprocessing import Pool
from ete3 import Tree

from dpvtex.perfect_phylogenies.utils import populate
from dpvtex.perfect_phylogenies.random_perfect_phylogeny import RandomPerfectPhylogeny
from dpvtex.perfect_phylogenies.perturb_phylogeny import (
    perturb_tree,
    make_worse_tree,
    sankoff_for_missing_sequences,
)


def make_phylogeny_data(n_leaves, n_phylos_per_tree, depth):
    """
    Create a collection of phylogenies obtained by randomly mixing a subtree of a
    perfect phylogeny.
    """
    # create a random tree
    tree = Tree()
    populate(tree, n_leaves, model="uniform")
    rpp = RandomPerfectPhylogeny(tree)
    data_pairs = []

    for _ in range(n_phylos_per_tree):
        # create a random prefect phylogeny
        phylo = rpp.make_random_perfect_phylogeny()
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
    n_trees, n_phylos_per_tree, n_leaves, depth=4, n_threads=1
):
    """
    Create a collection of random trees and phylogenies obtained by randomly mixing a
    subtree of a perfect phylogeny.
    """
    if n_threads > 1:
        process_pool = Pool(n_threads)
        pooled_results = process_pool.starmap(
            make_phylogeny_data, [(n_leaves, n_phylos_per_tree, depth)] * n_trees
        )
    else:
        # Skip the performance hit from using multiprocessing with one process.
        pooled_results = [
            make_phylogeny_data(n_leaves, n_phylos_per_tree, depth)
            for _ in range(n_trees)
        ]
    tree_data_dict = {
        mixed_phylo: edge_classifier
        for paired_data in pooled_results
        for mixed_phylo, edge_classifier in paired_data
    }
    return tree_data_dict



N_LEAVES = 100


def main():
    data_dict = create_training_data(
        n_trees=32,
        n_phylos_per_tree=32,
        n_leaves=N_LEAVES,
        n_threads=4,
    )
    file_path=f"{N_LEAVES}leaf_perfect.p"
    with open(file_path, "wb") as fh:
        pickle.dump(data_dict, file=fh)



if __name__ == "__main__":
    main()
