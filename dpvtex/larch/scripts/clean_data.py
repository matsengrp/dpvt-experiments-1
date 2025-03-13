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


def remove_identical_site_patterns(multiple_seq_alignment):
    """
    Remove identical site patterns from the alignment.
    Parameters:
    -----------
    multiple_seq_alignment : MultipleSeqAlignment
        The original alignment

    Returns:
    --------
    MultipleSeqAlignment
        A new alignment with identical site patterns removed
    int
        The number of unique site patterns in the new alignment
    """
    if len(multiple_seq_alignment) == 0:
        return multiple_seq_alignment, 0

    # Get the alignment length
    alignment_length = multiple_seq_alignment.get_alignment_length()

    # Dictionary to store unique patterns and their first occurrence index
    unique_patterns = {}
    unique_site_indices = []

    # Iterate through each site (column) in the alignment
    for site_idx in range(alignment_length):
        # Extract the site pattern (column) as a tuple (hashable)
        site_pattern = tuple(
            str(record.seq[site_idx]) for record in multiple_seq_alignment            
        )
        # Check if we've seen this pattern before
        if site_pattern not in unique_patterns:
            unique_patterns[site_pattern] = site_idx
            unique_site_indices.append(site_idx)

    # Create new sequences with only unique site patterns
    new_sequences = []
    for record in multiple_seq_alignment:
        # Extract only the sites at unique indices
        new_seq = "".join(str(record.seq[i]) for i in unique_site_indices)
        new_record = SeqRecord(
            Seq(new_seq), id=record.id, name=record.name, description=record.description
        )
        new_sequences.append(new_record)

    # Create a new MultipleSeqAlignment with reduced site patterns
    return MultipleSeqAlignment(new_sequences), len(unique_site_indices)


def clean_alignment(
    input_filename,
    output_filename,
    algn_length_filename,
    remove_site_patterns=False,
    target_length=None,
    target_seqs=None,
):
    """
    Clean an alignment by removing duplicates and optionally trimming to target dimensions.

    Parameters:
    -----------
    input_filename : str
        Path to the input FASTA alignment file
    output_filename : str
        Path to write the cleaned alignment
    algn_length_filename : str
        Path to write the final alignment dimensions (format: "length,num_seqs")
    remove_site_patterns : bool, optional
        Whether to remove duplicate site patterns (default: False)
    target_length : int, optional
        Target number of sites (informative sites will be selected)
    target_seqs : int, optional
        Target number of sequences

    Returns:
    --------
    tuple
        (final_num_seqs, final_num_sites, original_num_seqs, original_num_sites)
    """
    # Read the original alignment
    alignment = AlignIO.read(input_filename, "fasta")
    original_num_seqs = len(alignment)
    original_num_sites = alignment.get_alignment_length()
    print(
        f"Original alignment: {original_num_seqs} sequences, {original_num_sites} sites"
    )

    # Remove duplicate sequences first
    clean_alignment, num_unique_seqs = remove_duplicate_sequences(alignment)

    # Remove duplicate site patterns if requested
    if remove_site_patterns:
        print("Removing duplicate site patterns...")
        clean_alignment, num_unique_sites = remove_identical_site_patterns(
            clean_alignment
        )
        print(f"Unique site patterns: {num_unique_sites}")

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
            clean_alignment = create_trimmed_informative_alignment(
                clean_alignment, target_length, target_seqs
            )
            # Again remove duplicates after trimming
            clean_alignment, num_unique_seqs = remove_duplicate_sequences(
                clean_alignment
            )

    final_num_sites = clean_alignment.get_alignment_length()
    final_num_seqs = len(clean_alignment)

    AlignIO.write(clean_alignment, output_filename, "fasta")
    with open(algn_length_filename, "w") as f:
        f.write(str(final_num_sites) + "," + str(final_num_seqs))

    print(f"Final alignment: {final_num_seqs} sequences, {final_num_sites} sites")

    return final_num_seqs, final_num_sites, original_num_seqs, original_num_sites
