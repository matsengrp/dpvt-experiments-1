from shutil import rmtree


def check_alignment_size(algn_length_file, output_flag_file, min_sites=5, min_seqs=5):
    """
    Check whether the alignment has sufficient sites and sequences.

    If the alignment meets the minimum requirements, create a flag file.
    Otherwise, print a warning message.

    Parameters:
    -----------
    algn_length_file : str
        Path to file containing alignment dimensions (format: "length,num_seqs")
    output_flag_file : str
        Path to output flag file to create if alignment is sufficient
    min_sites : int, optional
        Minimum number of sites required (default: 5)
    min_seqs : int, optional
        Minimum number of sequences required (default: 5)

    Returns:
    --------
    bool
        True if alignment meets requirements, False otherwise
    """
    with open(algn_length_file, "r") as f:
        [msa_length, msa_size] = f.readline().split(",")

    if int(msa_length) >= min_sites and int(msa_size) >= min_seqs:
        with open(output_flag_file, "w") as flag:
            flag.write("NOT_EMPTY")
        return True
    else:
        data_name = algn_length_file.split("/")[-2]
        print(
            f"Alignment {data_name} contains less than {min_seqs} sequences or less than {min_sites} sites after removing uninformative sites. Skip this dataset."
        )
        return False
