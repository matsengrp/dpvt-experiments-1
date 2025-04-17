import historydag as hdag
import pickle
from ete3 import Tree
import os
from Bio import SeqIO
from dpvtex.perfect_phylogenies.perturb_phylogeny import (
    sankoff_for_missing_sequences,
)
from dpvtex.larch.scripts.extract_data_from_hdag import (
    assign_edge_labels,
    extract_hdag_clade_child_clades,
)
import sys


def read_trees(filename, fasta_path):
    """
    Read trees from filename and return them as ete3 trees with sequences of fasta assigned to leaves.
    Args:
        filename (str): Path to the file containing trees in newick format.
        fasta_path (str): Path to the fasta file containing sequences.
    """
    trees = []  # list of trees from iqtree run
    with open(filename, "r") as f:
        print(filename)
        # Get the basename to construct path to fasta file
        basename = os.path.basename(os.path.splitext(filename)[0])

        # Read sequences from fasta file
        sequences = {}
        try:
            for record in SeqIO.parse(fasta_path, "fasta"):
                sequences[record.id] = str(record.seq)
            print(f"Loaded {len(sequences)} sequences from {fasta_path}")
        except Exception as e:
            print(f"Error loading sequences from {fasta_path}: {str(e)}")
            return []

        for line in f.readlines():
            tree = Tree(line.rstrip())
            # Assign sequences to leaf nodes
            for leaf in tree.get_leaves():
                if "_<unknown_description>" in leaf.name:
                    leaf.name = leaf.name.split("_<unknown_description>")[0]
                if leaf.name in sequences:
                    leaf.add_feature("sequence", sequences[leaf.name])
                else:
                    print(f"Warning: No sequence found for leaf {leaf.name}")
            trees.append(tree)

    print(f"Loaded {len(trees)} trees from {filename}")
    return trees


def main():
    """
    Read a file containing pickled hDAG or protobuf, a file containing a list of trees to compare to DAG, and a fasta file with leaf sequences.
    The script will assign edge labels (MP vs non-MP edges) to the trees based on the hDAG and save the results in a pickle file.
    """
    if len(sys.argv) < 3:
        print(
            "Error: Please provide (1) file containing pickled hDAG or protobuf, (2) filename for dpvt data, and (3) files of trees to compare to DAG."
        )
        sys.exit(1)
    else:
        dag_file = sys.argv[1]
        dpvt_data_file = sys.argv[2]
        tree_file = sys.argv[3]
        fasta_path = sys.argv[4]

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
    dag_clades = extract_hdag_clade_child_clades(dag)
    # take a random tree from dag to be able to apply function assign_edge_labels()
    dag_tree = next(dag.get_histories()).to_ete()

    trees = read_trees(tree_file, fasta_path)
    # assign edge labels to the tree
    tree_label_dict = {}
    for tree in trees:
        sankoff_for_missing_sequences(tree)
        labels = assign_edge_labels(tree, dag_tree, dag_clades)
        tree_label_dict[tree] = labels

    with open(dpvt_data_file, "wb") as f:
        pickle.dump(tree_label_dict, f)


if __name__ == "__main__":
    main()
