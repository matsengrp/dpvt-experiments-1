import os
import sys
import pandas as pd
import json

# Add scripts directory to Python path for imports
snakefile_dir = workflow.basedir
scripts_dir = os.path.join(snakefile_dir, "scripts")
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from clean_data import clean_alignment
from check_size_fasta import check_alignment_size
from check_sequence_lengths import scan_directory

# Config file can be specified via --configfile on command line
# If not specified, falls back to config.yaml in the snakefile directory

input_data=os.path.realpath(config["input_data"])
remove_site_patterns = config.get("remove_duplicate_site_patterns", False)
max_ambiguous_site_frac_per_seq = config.get("max_ambiguous_site_frac_per_seq", None)


def get_subdirs(data_dir):
    """
    Get subdirectories excluding those with unequal sequence lengths.

    Alignments with unequal sequence lengths cannot be processed, so they are automatically
    filtered out if the exclusion list exists.
    """
    all_subdirs = [
        f.path.split("/")[-1] for f in os.scandir(data_dir) if f.is_dir() and ".snakemake" not in f.path
    ]

    # Path to the exclusion list
    exclusion_file = os.path.join(data_dir, "unequal_length_alignments.txt")

    # If the exclusion file doesn't exist yet, return all subdirs
    if not os.path.exists(exclusion_file):
        return all_subdirs

    # Read the list of directories to exclude
    excluded_dirs = set()
    with open(exclusion_file, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if line and not line.startswith('#'):
                excluded_dirs.add(line)

    # Filter out excluded directories
    filtered_subdirs = [d for d in all_subdirs if d not in excluded_dirs]
    return filtered_subdirs

def get_input_alignment(wildcards):
    """
    Find the input alignment file for a given subdirectory.
    Supports both:
    - alignment.nex (new format)
    - {subdir}.nex (directory-named nexus)
    - {subdir}.fasta (directory-named fasta, original format)
    """
    subdir = wildcards.subdir
    base_path = os.path.join(input_data, subdir)

    # Priority order: alignment.nex > {subdir}.nex > {subdir}.fasta
    candidates = [
        os.path.join(base_path, "alignment.nex"),
        os.path.join(base_path, f"{subdir}.nex"),
        os.path.join(base_path, f"{subdir}.fasta"),
    ]

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    # If none found, return the expected fasta path (will trigger error)
    return os.path.join(base_path, f"{subdir}.fasta")

dup_sites_suffix = ""
if remove_site_patterns in [True, "True", "true"]:
    dup_sites_suffix = "_no_dup_sites"

# First rule: check for unequal lengths before processing
rule check_unequal_lengths:
    output:
        done=input_data+"/unequal_length_check.done",
        report=input_data+"/unequal_length_alignments.txt"
    run:
        # Scan directory for alignments with unequal sequence lengths
        results = scan_directory(input_data, output.report)

        # Create done file
        with open(output.done, "w") as f:
            f.write("Unequal length check completed\n")
            f.write(f"Problematic alignments: {len(results['problematic'])}\n")
            f.write(f"Uniform alignments: {len(results['uniform'])}\n")
            f.write(f"Errors: {len(results['errors'])}\n")

        print(f"\n{'='*60}")
        print(f"Unequal length check completed:")
        print(f"  Problematic alignments: {len(results['problematic'])}")
        print(f"  Uniform alignments: {len(results['uniform'])}")
        print(f"  Errors: {len(results['errors'])}")
        print(f"{'='*60}\n")


rule all:
    input:
        input_data+"/unequal_length_check.done",
        expand(input_data+"/{subdir}/input" + dup_sites_suffix + ".fasta", subdir=get_subdirs(input_data)),
        input_data+"/alignment_size_stats" + dup_sites_suffix + ".csv"


rule clean_data:
    input:
        alignment_file=get_input_alignment,
        check_done=input_data+"/unequal_length_check.done"
    output:
        input_fasta=input_data+"/{subdir}/input" + dup_sites_suffix + ".fasta",
        size_stats_csv=input_data+"/{subdir}/size_stats" + dup_sites_suffix + ".csv"
    params:
        remove_site_patterns=remove_site_patterns,
        max_ambiguous_site_frac_per_seq=max_ambiguous_site_frac_per_seq
    run:
        clean_alignment(
            input.alignment_file,
            output.input_fasta,
            algn_length_filename=None,
            remove_site_patterns=params.remove_site_patterns,
            size_stats_csv=output.size_stats_csv,
            max_ambiguous_site_frac_per_seq=params.max_ambiguous_site_frac_per_seq
        )


rule aggregate_alignment_stats:
    input:
        check_done=input_data+"/unequal_length_check.done",
        individual_csvs=expand(input_data+"/{subdir}/size_stats" + dup_sites_suffix + ".csv", subdir=get_subdirs(input_data))
    output:
        combined_csv=input_data+"/alignment_size_stats" + dup_sites_suffix + ".csv"
    run:
        # Read all individual CSV files
        dfs = []
        for csv_file in input.individual_csvs:
            df = pd.read_csv(csv_file)
            dfs.append(df)

        # Concatenate all dataframes
        combined_df = pd.concat(dfs, ignore_index=True)

        # Write to output file
        combined_df.to_csv(output.combined_csv, index=False)

        print(f"\nAggregated stats from {len(dfs)} alignments")
        print(f"Output saved to: {output.combined_csv}\n")
