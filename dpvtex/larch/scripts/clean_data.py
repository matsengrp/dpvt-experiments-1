from Bio import AlignIO
from Bio.Align import MultipleSeqAlignment
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from collections import Counter
import csv
import os


def calculate_sequence_quality(sequence):
    """
    Calculate quality metrics for a sequence.

    Parameters:
    -----------
    sequence : str or Seq
        The sequence to analyze

    Returns:
    --------
    dict
        Dictionary with quality metrics:
        - 'gap_fraction': proportion of gaps (-, .)
        - 'ambiguous_fraction': proportion of ambiguous bases (N, ?, R, Y, etc.)
        - 'valid_fraction': proportion of valid bases (A, C, G, T)
        - 'length': sequence length
    """
    seq_str = str(sequence).upper()
    length = len(seq_str)

    if length == 0:
        return {
            'gap_fraction': 0.0,
            'ambiguous_fraction': 0.0,
            'valid_fraction': 0.0,
            'length': 0
        }

    # Standard nucleotides
    valid_bases = set('ACGT')
    # Gap characters
    gap_chars = set('-.')
    # Ambiguous IUPAC codes
    ambiguous_chars = set('NRYKMSWBDHV?')

    gap_count = sum(1 for c in seq_str if c in gap_chars)
    ambiguous_count = sum(1 for c in seq_str if c in ambiguous_chars)
    valid_count = sum(1 for c in seq_str if c in valid_bases)

    return {
        'gap_fraction': gap_count / length,
        'ambiguous_fraction': ambiguous_count / length,
        'valid_fraction': valid_count / length,
        'length': length
    }


def filter_low_quality_sequences(alignment, max_ambiguous_fraction=0.2):
    """
    Remove sequences with too many gaps or ambiguous characters.

    Parameters:
    -----------
    alignment : MultipleSeqAlignment
        The original alignment
    max_ambiguous_fraction : float
        Maximum allowed combined fraction of gaps and ambiguous bases (default: 0.2, i.e., 20%)
        This includes gap characters (-, .) and ambiguous IUPAC codes (N, ?, R, Y, etc.)

    Returns:
    --------
    tuple
        (filtered_alignment, num_kept, num_removed, removed_sequences)
        - filtered_alignment: MultipleSeqAlignment with low-quality sequences removed
        - num_kept: number of sequences kept
        - num_removed: number of sequences removed
        - removed_sequences: list of dicts with info about removed sequences
    """
    kept_sequences = []
    removed_sequences = []

    for record in alignment:
        quality = calculate_sequence_quality(record.seq)

        # Combined fraction of gaps and ambiguous bases
        combined_fraction = quality['gap_fraction'] + quality['ambiguous_fraction']

        if combined_fraction > max_ambiguous_fraction:
            removed_sequences.append({
                'id': record.id,
                'reason': 'low_quality',
                'gap_fraction': quality['gap_fraction'],
                'ambiguous_fraction': quality['ambiguous_fraction'],
                'combined_fraction': combined_fraction
            })
        else:
            kept_sequences.append(record)

    filtered_alignment = MultipleSeqAlignment(kept_sequences)

    return filtered_alignment, len(kept_sequences), len(removed_sequences), removed_sequences


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


def remove_uninformative_sites(alignment):
    """
    Remove uninformative sites from an alignment.

    A site is uninformative if it is either constant (all sequences have the same
    nucleotide) or is a singleton (only one sequence differs from the others).

    Parameters:
    -----------
    alignment : MultipleSeqAlignment
        The original alignment

    Returns:
    --------
    MultipleSeqAlignment
        A new alignment with only informative sites
    int
        The number of informative sites retained
    """
    # Find all informative sites
    informative_indices = get_informative_site_indices(alignment)

    if len(informative_indices) == 0:
        return MultipleSeqAlignment([]), 0

    # Create a new alignment with only the informative sites
    new_alignment = MultipleSeqAlignment([])
    for record in alignment:
        new_seq = Seq("".join([record.seq[i] for i in informative_indices]))
        new_record = SeqRecord(new_seq, id=record.id, description="")
        new_alignment.append(new_record)

    return new_alignment, len(informative_indices)


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
    # If there aren't enough informative sites, just use what we have
    if len(informative_indices) < target_length:
        target_length = len(informative_indices)

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


def write_alignment_size_stats(
    csv_path,
    alignment_name,
    original_num_seqs,
    original_num_sites,
    final_num_seqs,
    final_num_sites,
):
    """
    Write or append alignment size statistics to a CSV file.

    Parameters:
    -----------
    csv_path : str
        Path to the CSV file
    alignment_name : str
        Name/identifier of the alignment
    original_num_seqs : int
        Number of sequences before cleaning
    original_num_sites : int
        Number of sites before cleaning
    final_num_seqs : int
        Number of sequences after cleaning
    final_num_sites : int
        Number of sites after cleaning
    """
    # Check if CSV file exists to determine if we need to write header
    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # Write header if this is a new file
        if not file_exists:
            writer.writerow([
                "alignment_name",
                "original_num_seqs",
                "original_num_sites",
                "cleaned_num_seqs",
                "cleaned_num_sites",
                "seq_ratio",
                "site_ratio",
            ])

        # Calculate ratios
        seq_ratio = final_num_seqs / original_num_seqs if original_num_seqs > 0 else 0
        site_ratio = final_num_sites / original_num_sites if original_num_sites > 0 else 0

        # Write data row
        writer.writerow([
            alignment_name,
            original_num_seqs,
            original_num_sites,
            final_num_seqs,
            final_num_sites,
            seq_ratio,
            site_ratio,
        ])


def clean_alignment(
    input_filename,
    output_filename,
    algn_length_filename=None,
    remove_site_patterns=False,
    target_length=None,
    target_seqs=None,
    size_stats_csv=None,
    max_ambiguous_fraction=None,
):
    """
    Clean an alignment by removing low-quality sequences, duplicates, and uninformative sites.

    Parameters:
    -----------
    input_filename : str
        Path to the input alignment file (supports FASTA and NEXUS formats)
    output_filename : str
        Path to write the cleaned alignment
    algn_length_filename : str, optional
        Path to write the final alignment dimensions (format: "length,num_seqs")
    remove_site_patterns : bool, optional
        Whether to remove duplicate site patterns (default: False)
    target_length : int, optional
        Target number of sites (informative sites will be selected)
    target_seqs : int, optional
        Target number of sequences
    size_stats_csv : str, optional
        Path to CSV file where alignment size statistics will be appended
    max_ambiguous_fraction : float, optional
        Maximum allowed combined fraction of gaps and ambiguous bases per sequence (default: None, no filtering)
        Example: 0.5 means sequences with >50% gaps+ambiguous bases will be removed

    Returns:
    --------
    tuple
        (final_num_seqs, final_num_sites, original_num_seqs, original_num_sites)
    """
    # Determine input format based on file extension
    if input_filename.endswith(('.nex', '.nexus')):
        input_format = 'nexus'
    else:
        input_format = 'fasta'

    # Read the original alignment
    alignment = AlignIO.read(input_filename, input_format)
    original_num_seqs = len(alignment)
    original_num_sites = alignment.get_alignment_length()

    # Filter low-quality sequences if threshold is provided
    if max_ambiguous_fraction is not None:
        alignment, num_kept, num_removed, removed_seqs = filter_low_quality_sequences(
            alignment, max_ambiguous_fraction
        )

    # Remove uninformative sites and duplicate sequences
    clean_alignment, num_unique_seqs = remove_uninformative_sites(alignment)
    clean_alignment, num_unique_seqs = remove_duplicate_sequences(clean_alignment)

    # Remove duplicate site patterns if requested
    if remove_site_patterns:
        clean_alignment, num_unique_sites = remove_identical_site_patterns(
            clean_alignment
        )

    # If we need to trim to target dimensions
    if target_length is not None and target_seqs is not None:
        # First trim down to target number of sequences
        if num_unique_seqs < target_seqs:
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

    if algn_length_filename is not None:
        with open(algn_length_filename, "w") as f:
            f.write(str(final_num_sites) + "," + str(final_num_seqs))

    # Write alignment size statistics to CSV if requested
    if size_stats_csv is not None:
        alignment_name = os.path.basename(os.path.dirname(input_filename))
        write_alignment_size_stats(
            size_stats_csv,
            alignment_name,
            original_num_seqs,
            original_num_sites,
            final_num_seqs,
            final_num_sites,
        )

    return final_num_seqs, final_num_sites, original_num_seqs, original_num_sites
