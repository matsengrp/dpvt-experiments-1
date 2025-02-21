import os, sys
import argparse
from pathlib import Path
from collections import defaultdict
import itertools as it

import pickle
import torch
from multiprocessing import Pool
from ete3 import Tree
from sklearn.model_selection import train_test_split

from dpvtex.perfect_phylogenies.utils import populate
from dpvtex.perfect_phylogenies.random_perfect_phylogeny import RandomPerfectPhylogeny
from dpvtex.perfect_phylogenies.perturb_phylogeny import (
    perturb_tree,
    make_worse_tree,
    make_worse_spr,
    sankoff_for_missing_sequences,
)


def print_leaf_sequences(phylo):
    print(f'features: {phylo.features}')
    leaf_seqs = [x.sequence for x in phylo.get_leaves()]
    leaf_lens = [len(x) for x in leaf_seqs]
    print(f'leaf_seqs: {len(leaf_seqs)} {leaf_seqs}')
    print(f'leaf_lens: {len(leaf_lens)} {leaf_lens}')


def make_phylogeny_data(n_leaves, n_phylos_per_tree, depth, n_sites=None, spr=False):
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
        phylo = rpp.make_random_perfect_phylogeny(n_sites=n_sites)
        print_leaf_sequences(phylo)
        print(f'mut_selections: {rpp.mut_selections}')
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
    n_trees, n_phylos_per_tree, n_leaves, depth=4, n_sites=None, n_threads=1, spr=False
):
    """
    Create a collection of random trees and phylogenies obtained by randomly
    mixing a subtree of a perfect phylogeny.

    """
    if n_threads > 1:
        process_pool = Pool(n_threads)
        pooled_results = process_pool.starmap(
            make_phylogeny_data, [(n_leaves, n_phylos_per_tree, depth, n_sites, spr)] * n_trees
        )
    else:
        # Skip the performance hit from using multiprocessing with one process.
        pooled_results = [
            make_phylogeny_data(n_leaves, n_phylos_per_tree, depth, n_sites, spr)
            for _ in range(n_trees)
        ]
    tree_data_dict = {
        mixed_phylo: edge_classifier
        for paired_data in pooled_results
        for mixed_phylo, edge_classifier in paired_data
    }
    return tree_data_dict


def get_output_path(output_dir, prefix, n_trees, n_phylos, n_leaves, n_sites, depth, spr):
    path = f"{output_dir}/{prefix}_{n_leaves}_leaves_{n_trees}_trees"
    if n_sites:
        path += f"_{n_sites}_sites"
    if spr:
        path += f"_spr"
    return f"{path}.p"


def split_train_test_data(data_dict, test_size=0.2, random_state=42):
    x_data = list(data_dict.keys())
    y_data = list(data_dict.values())
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size, random_state)
    train_dict = dict(zip(x_train, y_train))
    test_dict = dict(zip(x_test, y_test))
    return train_dict, test_dict


def save_data(data_dict, data_path):
    with open(data_path, "wb") as file:
        pickle.dump(data_dict, file)



class Parser:
    args_default = {
      "n_sites": [None],
      "n_phylos": 1,
      "depth": 2,
      "spr": False,
      "n_threads": 4,

      "split_data": False,
      "output_dir": "./",
      "prefix": "perfect",
    }

    args_help = {
      "n_trees": "Number of unique trees.",
      "n_leaves": "Number of leaves in each tree.",
      "n_sites": "Number of sites in each tree. If none, determined by number of random mutations required to cover all internal nodes.",
      "n_phylos": "Number of phylo trees per unique tree.",
      "depth": "Depth of each tree.",
      "spr": "SPR moves for creating sub-optimal edges. Otherwise, uses subtree replacement.",
      "n_threads": "Number of threads for building datasets.",

      "output_dir": "Output directory for datasets.",
      "prefix": "Prefix to dataset names.",
      "split_data": "Split datasets into train and test sets.",
    }

    def __init__(self, commandline_args):
        if isinstance(commandline_args, str):
            self.commandline_args = commandline_args.split()
        assert isinstance(commandline_args, list)
        self.commandline_args = commandline_args

    def parse(self):
        parser = argparse.ArgumentParser(
            description="Create perfect phylogenies with some non-MP edges for the provided number of leaves"
        )
        parser.add_argument("--n_trees", type=Parser.parse_int_list, required=True)
        parser.add_argument("--n_leaves", type=Parser.parse_int_list, required=True)
        parser.add_argument("--n_sites", type=Parser.parse_int_list)
        parser.add_argument("--n_phylos", type=Parser.int)
        parser.add_argument("--depth", type=Parser.int)
        parser.add_argument("--spr", type=Parser.parse_flag, nargs="?", const=True)
        parser.add_argument("--n_threads", type=int)

        parser.add_argument("--output_dir", type=Parser.parse_output_path)
        parser.add_argument("--prefix", type=str)
        parser.add_argument("--split_data", type=Parser.parse_flag, nargs="?", const=True)
        parser.set_defaults(**self.args_default)

        for action in parser._actions:
            if action.help is None:
                action.help = ""
            if action.dest in self.args_help.keys():
                action.help += f"{self.args_help[action.dest]} "
            if action.dest in self.args_default.keys():
                action.help += f"(default: '{self.args_default[action.dest]}') "

        args = parser.parse_args(self.commandline_args)
        args = defaultdict(lambda: None, vars(args))
        print(args)
        return args

    @staticmethod
    def parse_list(args, type):
        output = []
        for arg in args.split(","):
            output.append(type(arg))
        return output

    parse_str_list = staticmethod(lambda x: Parser.parse_list(x, type=str))
    parse_int_list = staticmethod(lambda x: Parser.parse_list(x, type=int))

    @staticmethod
    def parse_output_path(arg):
        path_dir = os.path.dirname(os.path.abspath(arg))
        print(f"path_dir: {path_dir}")
        if not os.path.isdir(os.path.dirname(os.path.abspath(arg))):
            raise argparse.ArgumentTypeError(f"Directory for input path '{arg}' does not exist.")
        return arg

    @staticmethod
    def parse_input_path(arg):
        if not os.path.exists(arg):
            raise argparse.ArgumentTypeError(f"Output path '{arg}' does not exist.")
        return arg

    @staticmethod
    def parse_flag(arg):
        if isinstance(arg, bool):
            return arg
        arg = arg.lower()
        if arg in {"true", "yes", "1"}:
            return True
        elif arg in {"false", "no", "0"}:
            return False
        raise argparse.ArgumentTypeError(f"Invalid boolean arg: '{arg}' (expected true/false)")


def main():
    args = Parser(sys.argv[1:]).parse()

    data_settings = it.product(
        args["n_trees"],
        args["n_leaves"],
        args["n_sites"],
    )
    for i,(n_trees,n_leaves,n_sites) in enumerate(data_settings):
        output_path = get_output_path(
            output_dir=args["output_dir"],
            prefix=args["prefix"],
            n_trees=n_trees,
            n_phylos=args["phylos"],
            n_leaves=n_leaves,
            n_sites=n_sites,
            depth=args["depth"],
            spr=args["spr"],
        )

        print(f"# building {output_path}...")
        data_dict = create_training_data(
            n_trees=n_trees,
            n_phylos_per_tree=args["n_phylos"],
            n_leaves=n_leaves,
            n_sites=n_sites,
            depth=args["depth"],
            n_threads=4,
            spr=args["spr"],
        )
        print(data_dict)
        save_data(data_dict, output_path)

        if args["split_data"]:
            train_data,test_data = split_train_test_data(data_dict, train_percentage=0.8)
            save_data(train_data, output_path.replace(".p", "_train.p"))
            save_data(test_data, output_path.replace(".p", "_test.p"))

        if args[""]


if __name__ == "__main__":
    main()
