import sys

sys.path.append("..")
import pickle
import torch
from ete3 import Tree
from utils import populate
from random_perfect_phylogeny import RandomPerfectPhylogeny
from perturb_phylogeny import (
    perturb_tree,
    make_worse_tree,
    sankoff_for_missing_sequences,
)


def create_training_data(file_path, n_trees, n_leaves):
    """
    Create a collection of phylogenies obtained by randomly mixing a subtree of a perfect
    phylogeny.
    """
    DEPTH = 4
    n_phylos_per_tree = 32
    tree_data_dict = {}
    for _ in range(n_trees):
        tree = Tree()
        populate(tree, n_leaves, model="uniform")
        # pp = PerfectPhylogeny(tree)
        rpp = RandomPerfectPhylogeny(tree)

        for _ in range(n_phylos_per_tree):
            # phylo = pp.make_random_phylogeny()
            phylo = rpp.make_random_perfect_phylogeny()
            mixed_phylo = make_worse_tree(phylo, depth=DEPTH, max_attempts=500)
            if mixed_phylo is None:
                mixed_phylo = perturb_tree(phylo, depth=DEPTH, exception_on_fail=True)
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
            tree_data_dict[mixed_phylo] = edge_classifier
    # shuffle keys and make train / validation split
    num_items = n_trees * n_phylos_per_tree
    num_train = int(num_items * 0.8)

    keys = list(tree_data_dict.keys())
    random_idx = torch.randperm(num_items)
    train_keys = [keys[i] for i in random_idx[:num_train]]
    val_keys = [keys[i] for i in random_idx[num_train:]]

    train_data = {key: tree_data_dict[key] for key in train_keys}
    val_data = {key: tree_data_dict[key] for key in val_keys}

    data_dict = {"train": train_data, "val": val_data}
    with open(file_path, "wb") as fh:
        pickle.dump(data_dict, file=fh)


N_LEAVES = 100


def main():
    create_training_data(
        file_path=f"{N_LEAVES}leaf_perfect.p", n_trees=32, n_leaves=N_LEAVES
    )


if __name__ == "__main__":
    main()
