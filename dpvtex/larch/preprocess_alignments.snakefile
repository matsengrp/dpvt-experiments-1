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
from pipeline_logger import get_logger

# Config file can be specified via --configfile on command line
# If not specified, falls back to config.yaml in the snakefile directory

input_data=os.path.realpath(config["input_data"])
remove_site_patterns = config.get("remove_duplicate_site_patterns", False)
max_ambiguous_site_frac_per_seq = config.get("max_ambiguous_site_frac_per_seq", None)


def get_all_subdirs(data_dir):
    """
    Get all subdirectories that contain alignment files.
    Excludes .snakemake and directories without any alignment files.
    """
    subdirs = []
    for f in os.scandir(data_dir):
        if not f.is_dir() or ".snakemake" in f.path:
            continue

        subdir_name = f.path.split("/")[-1]
        subdir_path = os.path.join(data_dir, subdir_name)

        # Check if directory contains any alignment file
        has_alignment = any([
            os.path.exists(os.path.join(subdir_path, "alignment.nex")),
            os.path.exists(os.path.join(subdir_path, f"{subdir_name}.nex")),
            os.path.exists(os.path.join(subdir_path, f"{subdir_name}.fasta"))
        ])

        if has_alignment:
            subdirs.append(subdir_name)
    return subdirs


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

# Get all subdirectories at DAG build time
ALL_SUBDIRS = get_all_subdirs(input_data)

dup_sites_suffix = ""
if remove_site_patterns in [True, "True", "true"]:
    dup_sites_suffix = "_no_dup_sites"

rule all:
    input:
        input_data+"/unequal_length_check.done",
        input_data+"/alignment_size_stats" + dup_sites_suffix + ".csv",
        input_fasta=expand(input_data+"/{subdir}/input" + dup_sites_suffix + ".fasta", subdir=ALL_SUBDIRS),

# First rule: check for unequal lengths and report them
# This doesn't block processing - it just creates a report
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
        if results['problematic']:
            print(f"\nWARNING: The following alignments have unequal sequence lengths and will fail:")
            for subdir in results['problematic']:
                print(f"  - {subdir}")
        print(f"{'='*60}\n")


rule clean_data:
    input:
        alignment_file=get_input_alignment
    output:
        input_fasta=input_data+"/{subdir}/input" + dup_sites_suffix + ".fasta",
        size_stats_csv=input_data+"/{subdir}/size_stats" + dup_sites_suffix + ".csv"
    params:
        remove_site_patterns=remove_site_patterns,
        max_ambiguous_site_frac_per_seq=max_ambiguous_site_frac_per_seq
    run:
        logger = get_logger(input_data)
        clean_alignment(
            input.alignment_file,
            output.input_fasta,
            algn_length_filename=None,
            remove_site_patterns=params.remove_site_patterns,
            size_stats_csv=output.size_stats_csv,
            max_ambiguous_site_frac_per_seq=params.max_ambiguous_site_frac_per_seq,
            logger=logger
        )


def get_existing_stats_files(wildcards):
    """
    Find all size_stats CSV files that exist after clean_data attempts.
    This is called after clean_data rules have been attempted.
    """
    all_csvs = []
    for subdir in ALL_SUBDIRS:
        csv_path = os.path.join(input_data, subdir, f"size_stats{dup_sites_suffix}.csv")
        if os.path.exists(csv_path):
            all_csvs.append(csv_path)
    return all_csvs


rule aggregate_alignment_stats:
    input:
        check_done=input_data+"/unequal_length_check.done",
        # Try to create all stats files, but allow some to fail
        stats_files=expand(input_data+"/{subdir}/size_stats" + dup_sites_suffix + ".csv", subdir=ALL_SUBDIRS)
    output:
        combined_csv=input_data+"/alignment_size_stats" + dup_sites_suffix + ".csv"
    run:
        # Find all existing size_stats CSV files (some may have failed)
        all_csvs = []
        for csv_path in input.stats_files:
            if os.path.exists(csv_path):
                all_csvs.append(csv_path)

        if len(all_csvs) == 0:
            raise ValueError("No size_stats.csv files found. All alignments may have failed.")

        # Read all individual CSV files
        dfs = []
        for csv_file in all_csvs:
            df = pd.read_csv(csv_file)
            dfs.append(df)

        # Concatenate all dataframes
        combined_df = pd.concat(dfs, ignore_index=True)

        # Write to output file
        combined_df.to_csv(output.combined_csv, index=False)

        failed_count = len(ALL_SUBDIRS) - len(all_csvs)
        print(f"\nAggregated stats from {len(dfs)} alignments")
        if failed_count > 0:
            print(f"  WARNING: {failed_count} alignments failed and were skipped")
        print(f"Output saved to: {output.combined_csv}\n")
