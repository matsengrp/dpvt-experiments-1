from Bio import AlignIO
from Bio.Align import MultipleSeqAlignment
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from collections import Counter


def is_site_informative(column):
    """Check if a site is phylogenetically informative.

    A site is informative if it has at least two different nucleotides,
    and each occurs at least twice in the column.
    """
    column = column.upper()
    if not all(char in "ACGTacgt" for char in column):
        return False

    counter = Counter(column)
    # At least two different nucleotides
    if len(counter) < 2:
        return False

    # Check if it's a singleton (only one sequence has a different nucleotide)
    if len(counter) == 2 and 1 in counter.values():
        return False

    return True


def get_informative_site_indices(alignment):
    """Get indices of all informative sites in the alignment."""
    informative_indices = []
    for i in range(alignment.get_alignment_length()):
        column = alignment[:, i]
        if is_site_informative(column):
            informative_indices.append(i)
    return informative_indices


def create_trimmed_informative_alignment(alignment, target_length, target_seqs):
    """
    Create an alignment with informative sites and target dimensions.

    Parameters:
    -----------
    alignment : MultipleSeqAlignment
        The original alignment
    target_length : int
        The desired number of sites
    target_seqs : int
        The desired number of sequences

    Returns:
    --------
    MultipleSeqAlignment
        A new alignment with exactly target_length informative sites and target_seqs sequences
    """
    # trim number of sequences if needed
    if len(alignment) > target_seqs:
        alignment = MultipleSeqAlignment(alignment[:target_seqs])

    # Find all informative sites
    informative_indices = get_informative_site_indices(alignment)

    # Make sure we have enough informative sites
    if len(informative_indices) < target_length:
        print(
            f"Not enough informative sites ({len(informative_indices)}) to meet target length ({target_length})"
        )

    # Take only the first target_length informative sites
    selected_indices = sorted(informative_indices[:target_length])

    # Create a new alignment with only the selected informative sites
    new_alignment = MultipleSeqAlignment([])
    for record in alignment:
        new_seq = Seq("".join([record.seq[i] for i in selected_indices]))
        new_record = SeqRecord(new_seq, id=record.id, description="")
        new_alignment.append(new_record)

    return new_alignment


def remove_duplicate_sequences(multiple_seq_alignment):
    """
    Remove duplicate sequences from the alignment.

    Parameters:
    -----------
    multiple_seq_alignment : MultipleSeqAlignment
        The original alignment

    Returns:
    --------
    MultipleSeqAlignment
        A new alignment with duplicate sequences removed
    int
        The number of unique sequences in the new alignment
    """
    seen_sequences = set()  # To keep track of unique sequences
    unique_sequences = []  # List to hold unique SeqRecord objects

    for record in multiple_seq_alignment:
        # Convert sequence to a string (to make it hashable)
        seq_str = str(record.seq)
        if seq_str not in seen_sequences:
            seen_sequences.add(seq_str)
            unique_sequences.append(record)

    # Create a new MultipleSeqAlignment with unique sequences
    return MultipleSeqAlignment(unique_sequences), len(unique_sequences)


if __name__ == "__main__":
    import sys

    # Simple command-line argument parsing
    if len(sys.argv) < 4:
        print(
            "Usage: python clean_data.py <input_file> <output_file> <length_file> [target_length] [target_seqs]"
        )
        sys.exit(1)

    input_filename = sys.argv[1]
    output_filename = sys.argv[2]
    algn_length_filename = sys.argv[3]

    # Get target dimensions if provided
    target_length = None
    target_seqs = None
    if len(sys.argv) > 4:
        target_length = int(sys.argv[4])
    if len(sys.argv) > 5:
        target_seqs = int(sys.argv[5])

    # Read the original alignment
    alignment = AlignIO.read(input_filename, "fasta")
    original_num_seqs = len(alignment)
    original_num_sites = alignment.get_alignment_length()
    print(
        f"Original alignment: {original_num_seqs} sequences, {original_num_sites} sites"
    )

    # Remove duplicate sequences first
    clean_alignment, num_unique_seqs = remove_duplicate_sequences(alignment)

    # If we need to trim to target dimensions
    if target_length is not None and target_seqs is not None:
        print("Trimming alignment to target dimensions")
        # First trim down to target number of sequences
        if num_unique_seqs < target_seqs:
            print(
                f"Warning: Not enough unique sequences ({num_unique_seqs}) available to meet target ({target_seqs})"
            )
            target_seqs = num_unique_seqs
        else:
            clean_alignment = MultipleSeqAlignment(clean_alignment[:target_seqs])
            # Using create_trimmed_informative_alignment which filters for informative sites
            clean_alignment = create_trimmed_informative_alignment(clean_alignment, target_length, target_seqs)
            # Again remove duplicates after trimming
            clean_alignment, num_unique_seqs = remove_duplicate_sequences(clean_alignment)

    final_num_sites = clean_alignment.get_alignment_length()
    final_num_seqs = len(clean_alignment)

    AlignIO.write(clean_alignment, output_filename, "fasta")
    with open(algn_length_filename, "w") as f:
        f.write(str(final_num_sites) + "," + str(final_num_seqs))

    print(f"Final alignment: {final_num_seqs} sequences, {final_num_sites} sites")
