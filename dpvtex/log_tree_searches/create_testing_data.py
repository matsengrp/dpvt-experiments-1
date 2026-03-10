import historydag as hdag
import pickle
from ete3 import Tree
import os
from Bio import SeqIO
from dpvtex.larch.scripts.tree_perturbation import (
    sankoff_for_missing_sequences,
)
from dpvtex.larch.scripts.extract_data_from_hdag import (
    assign_edge_labels,
    extract_hdag_clade_child_clades,
)
import sys

# Suffix added by some tree software that needs to be stripped from leaf names
UNKNOWN_DESCRIPTION_SUFFIX = "_<unknown_description>"


def build_dag_to_fasta_name_map(dag, fasta_path):
    """Build a mapping from DAG node_id to FASTA leaf names.

    Larch's VCF pipeline replaces "-" with "N" in sample names, so DAG
    node_ids don't match FASTA IDs directly. We build the mapping by
    replacing "-" with "N" in FASTA names to find corresponding DAG node_ids.

    Args:
        dag: historydag.sequence_dag.SequenceHistoryDag (trimmed to MP)
        fasta_path: Path to the FASTA file with original leaf names.

    Returns:
        dict: mapping from DAG node_id strings to FASTA ID strings.
    """
    fasta_ids = {record.id for record in SeqIO.parse(fasta_path, "fasta")}
    dag_leaf_ids = set()
    for node in dag.postorder():
        if node.is_leaf():
            dag_leaf_ids.add(node.label.node_id)

    # Build map: FASTA name with "-" replaced by "N" should match DAG node_id
    dag_to_fasta = {}
    for fasta_id in fasta_ids:
        dag_id = fasta_id.replace("-", "N")
        if dag_id in dag_leaf_ids:
            dag_to_fasta[dag_id] = fasta_id
        elif fasta_id in dag_leaf_ids:
            # No renaming needed for this leaf
            dag_to_fasta[fasta_id] = fasta_id

    unmapped = dag_leaf_ids - set(dag_to_fasta.keys())
    if unmapped:
        print(f"Warning: {len(unmapped)} DAG leaves could not be mapped to FASTA IDs")

    return dag_to_fasta


def extract_dag_clades_with_fasta_names(dag, dag_to_fasta):
    """Extract DAG clades remapped to use FASTA leaf names.

    Args:
        dag: historydag.sequence_dag.SequenceHistoryDag
        dag_to_fasta: mapping from DAG node_id to FASTA ID

    Returns:
        dict: clade -> child_clades, using FASTA leaf names
    """
    dag_clades_raw = extract_hdag_clade_child_clades(dag)
    dag_clades = {}
    for clade, child_clades in dag_clades_raw.items():
        mapped_clade = frozenset(dag_to_fasta.get(n, n) for n in clade)
        mapped_children = frozenset(
            frozenset(dag_to_fasta.get(n, n) for n in cc) for cc in child_clades
        )
        dag_clades[mapped_clade] = mapped_children
    return dag_clades


def read_trees(filename, fasta_path):
    """
    Read trees from filename and return them as ete3 trees with sequences
    assigned to leaves.

    Uses ete3's resolve_polytomy() to ensure output trees are binary.

    Args:
        filename (str): Path to the file containing trees in newick format.
        fasta_path (str): Path to the fasta file containing sequences.

    Returns:
        list: List of ete3 Tree objects with sequences attached to leaves.
              Returns empty list if sequences cannot be loaded.
    """
    trees = []
    with open(filename, "r") as f:
        # Read sequences from fasta file
        sequences = {}
        try:
            for record in SeqIO.parse(fasta_path, "fasta"):
                sequences[record.id] = str(record.seq).upper()
            print(f"Loaded {len(sequences)} sequences from {fasta_path}")
        except Exception as e:
            print(f"Error loading sequences from {fasta_path}: {str(e)}")
            return []

        for line in f.readlines():
            tree = Tree(line.rstrip(), format=8)
            tree.resolve_polytomy(recursive=True)
            # Assign sequences to leaf nodes
            for leaf in tree.get_leaves():
                # Strip software-added suffixes from leaf names
                if UNKNOWN_DESCRIPTION_SUFFIX in leaf.name:
                    leaf.name = leaf.name.split(UNKNOWN_DESCRIPTION_SUFFIX)[0]
                leaf.add_feature("sequence", sequences[leaf.name])
            trees.append(tree)

    print(f"Loaded {len(trees)} trees from {filename}")
    return trees


def main():
    """
    Create DPVT testing data from tree search logs and an hDAG.

    Reads a pickled hDAG or protobuf file, a file containing logged trees,
    and a FASTA file with leaf sequences. Assigns edge labels (MP vs non-MP)
    to trees based on the hDAG and saves results as a pickle file.

    Usage:
        python create_testing_data.py <dag_file> <output_file> <tree_file> <fasta_file>

    Args:
        dag_file: Path to pickled hDAG (.p) or protobuf (.pb) file.
        output_file: Path for output pickle file with labeled trees.
        tree_file: Path to file containing trees in newick format.
        fasta_file: Path to FASTA file with leaf sequences.
    """
    if len(sys.argv) < 5:
        print(
            "Usage: python create_testing_data.py <dag_file> <output_file> <tree_file> <fasta_file>"
        )
        print("  dag_file:    Pickled hDAG (.p) or protobuf (.pb)")
        print("  output_file: Output pickle file for labeled trees")
        print("  tree_file:   File containing trees in newick format")
        print("  fasta_file:  FASTA file with leaf sequences")
        sys.exit(1)

    dag_file = sys.argv[1]
    dpvt_data_file = sys.argv[2]
    tree_file = sys.argv[3]
    fasta_path = sys.argv[4]

    for path, label in [
        (dag_file, "DAG file"),
        (tree_file, "Tree file"),
        (fasta_path, "FASTA file"),
    ]:
        if not os.path.exists(path):
            print(f"Error: {label} not found: {path}")
            sys.exit(1)

    # Load DAG based on file extension
    if dag_file.endswith(".p"):
        with open(dag_file, "rb") as f:
            dag = pickle.load(f)
    elif dag_file.endswith(".pb"):
        dag = hdag.mutation_annotated_dag.load_MAD_protobuf_file(
            dag_file, compact_genomes=True
        )
    else:
        print("Error: DAG file must be .p (pickle) or .pb (protobuf)")
        sys.exit(1)

    # Trim to only MP topologies and convert to sequence_dag
    dag.trim_optimal_weight()
    dag = hdag.sequence_dag.SequenceHistoryDag.from_history_dag(dag)

    # Map DAG node_ids to FASTA names and extract clades with FASTA names
    dag_to_fasta = build_dag_to_fasta_name_map(dag, fasta_path)
    dag_clades = extract_dag_clades_with_fasta_names(dag, dag_to_fasta)
    dag_tree = next(dag.get_histories()).to_ete(
        name_func=lambda n: (
            dag_to_fasta.get(n.label.node_id, n.label.node_id) if n.is_leaf() else ""
        )
    )

    all_trees = read_trees(tree_file, fasta_path)

    # Deduplicate trees: phangorn logs every iteration, including ones where
    # the tree didn't change. Sort descendants for canonical child ordering
    # before comparing newick strings.
    seen_newicks = set()
    trees = []
    for tree in all_trees:
        tree.sort_descendants()
        nw = tree.write(format=9)
        if nw not in seen_newicks:
            seen_newicks.add(nw)
            trees.append(tree)
    print(f"Kept {len(trees)} unique trees out of {len(all_trees)}")

    # Assign edge labels to each tree
    tree_label_dict = {}
    for tree in trees:
        sankoff_for_missing_sequences(tree)
        labels = assign_edge_labels(tree, dag_tree, dag_clades)
        tree_label_dict[tree] = labels

    with open(dpvt_data_file, "wb") as f:
        pickle.dump(tree_label_dict, f)


if __name__ == "__main__":
    main()
