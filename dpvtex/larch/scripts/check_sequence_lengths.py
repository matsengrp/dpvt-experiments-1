from Bio import SeqIO
from pathlib import Path


def check_alignment_has_equal_lengths(alignment_file):
    """
    Check if all sequences in an alignment file have the same length.

    Parameters:
    -----------
    alignment_file : str
        Path to the alignment file (FASTA or NEXUS format)

    Returns:
    --------
    tuple
        (all_equal: bool, details: dict)
        - all_equal: True if all sequences have the same length
        - details: dictionary with sequence length information
    """
    # Convert to string if it's a Path object
    alignment_file = str(alignment_file)

    # Determine format based on file extension
    if alignment_file.endswith((".nex", ".nexus")):
        file_format = "nexus"
    else:
        file_format = "fasta"

    try:
        sequences = list(SeqIO.parse(alignment_file, file_format))

        if not sequences:
            return False, {"error": "No sequences found", "num_sequences": 0}

        lengths = [len(seq.seq) for seq in sequences]
        unique_lengths = set(lengths)

        details = {
            "min_length": min(lengths),
            "max_length": max(lengths),
            "num_sequences": len(sequences),
            "num_unique_lengths": len(unique_lengths),
            "unique_lengths": sorted(unique_lengths),
        }

        # Add distribution if lengths vary
        if len(unique_lengths) > 1:
            details["length_distribution"] = {
                length: lengths.count(length) for length in unique_lengths
            }

        all_equal = len(unique_lengths) == 1

        return all_equal, details

    except Exception as e:
        return False, {"error": str(e), "num_sequences": 0}


def scan_directory(base_dir, output_file=None):
    """
    Scan directory structure for alignments with unequal sequence lengths.

    Parameters:
    -----------
    base_dir : str
        Base directory containing subdirectories with alignment files
    output_file : str, optional
        Path to output file where problematic directories will be saved

    Returns:
    --------
    dict
        Dictionary with results:
        - "problematic": list of directory names with unequal sequences
        - "uniform": list of directory names with equal sequences
        - "errors": list of directory names with errors
    """
    base_path = Path(base_dir)

    results = {"problematic": [], "uniform": [], "errors": []}

    # Find all subdirectories
    subdirs = [
        d for d in base_path.iterdir() if d.is_dir() and ".snakemake" not in str(d)
    ]

    for subdir in sorted(subdirs):
        # Look for alignment files in the subdirectory
        alignment_files = (
            list(subdir.glob("alignment.nex"))
            + list(subdir.glob(f"{subdir.name}.nex"))
            + list(subdir.glob(f"{subdir.name}.fasta"))
            + list(subdir.glob("*.fasta"))
            + list(subdir.glob("*.fa"))
            + list(subdir.glob("*.fna"))
        )

        if not alignment_files:
            continue

        # Check the first alignment file found
        alignment_file = alignment_files[0]
        all_equal, details = check_alignment_has_equal_lengths(alignment_file)

        if "error" in details:
            results["errors"].append(str(subdir.name))
        elif not all_equal:
            results["problematic"].append(str(subdir.name))
        else:
            results["uniform"].append(str(subdir.name))

    # Save problematic directories to file (always create the file, even if empty)
    if output_file:
        with open(output_file, "w") as f:
            f.write("# Directories with sequences of unequal lengths\n")
            f.write("# These alignments should be reviewed before processing\n\n")
            if results["problematic"]:
                for directory in results["problematic"]:
                    f.write(f"{directory}\n")
            else:
                f.write(
                    "# No problematic alignments found - all sequences have equal lengths!\n"
                )

        print(f"\nResults saved to: {output_file}")

    return results
