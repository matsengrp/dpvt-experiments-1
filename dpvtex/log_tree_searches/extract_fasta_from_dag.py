from Bio import SeqIO
import os
import historydag
import argparse

def save_dict_to_fasta(sequence_dict, output_file):
    """
    Write sequences from a dictionary to a fasta file.
    Args:
        sequence_dict (dict): Dictionary with sequence IDs as keys and sequences as values.
        output_file (str): Path to the output fasta file.
    """
    with open(output_file, 'w') as f:
        for seq_id, sequence in sequence_dict.items():
            f.write(f">{seq_id}\n")
            f.write(f"{sequence}\n")


def main():

    # Parse arguments
    parser = argparse.ArgumentParser(description='Take a DAG from protobuf and create fasta containing DAG leaf sequences.')

    # Add arguments
    parser.add_argument('-d', '--dag_file', help='File path to dag protobuf', required=True)
    parser.add_argument('-o', '--output', help='File to fasta output', required=True)
    parser.add_argument('-r', '--root_sequence', action='store_true', help='Set this flag to indicate whether you want to add DAG root sequence to fasta. Default is False')
    args = parser.parse_args()
    dag_file = args.dag_file
    fasta_file = args.output
    
    # Load DAG from .pb file
    dag = historydag.mutation_annotated_dag.load_MAD_protobuf_file(
            dag_file, compact_genomes=True
        )
    dag = historydag.sequence_dag.SequenceHistoryDag.from_history_dag(dag)
    
    sequences = {}
    for leaf in dag.get_leaves():
        sequences[leaf.label.node_id] = leaf.label.sequence
    ua_node = next(dag.preorder())
    if args.root_sequence:
        if len(list(ua_node.children())) > 1:
            print("Warning: More than one child for UA node")
        else:
            root_node = list(ua_node.children())[0]
            sequences["root"] = root_node.label.sequence
            fasta_file = fasta_file.replace(".fasta", "_root.fasta")

    save_dict_to_fasta(sequences, fasta_file)


if __name__ == "__main__":
    main()