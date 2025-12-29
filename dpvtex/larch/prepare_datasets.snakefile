import os
import sys

snakefile_dir = workflow.basedir
scripts_dir = os.path.join(snakefile_dir, "scripts")
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from prepare_dataset import create_filtered_dataset, create_train_test_split
from plot_alignment_size_ratios import plot_filtered_alignment_stats

# Config file can be specified via --configfile on command line
# If not specified, falls back to config.yaml in the snakefile directory

# Read config parameters
source_data = os.path.realpath(config["input_data"])
output_base = config["output_datasets"]
min_frac_sites_retained = config["min_frac_sites_retained"]
create_split = config.get("create_train_test_split", False)
test_fraction = config.get("test_fraction", 0.2)
dataset_name = config["dataset_name"]

# Derive output directory names
filtered_dir = f"{output_base}/{dataset_name}_filtered_{min_frac_sites_retained}"
train_dir = f"{output_base}/{dataset_name}_train_{min_frac_sites_retained}"
test_dir = f"{output_base}/{dataset_name}_test_{min_frac_sites_retained}"


rule all:
    input:
        f"{filtered_dir}/manifest.txt",
        f"{filtered_dir}/alignment_size_ratios.pdf",
        f"{filtered_dir}/cleaned_alignment_sizes.pdf",
        f"{train_dir}/manifest.txt" if create_split else [],
        f"{test_dir}/manifest.txt" if create_split else []


rule create_filtered_dataset:
    input:
        stats=f"{source_data}/alignment_size_stats.csv",
        check_done=f"{source_data}/unequal_length_check.done"
    output:
        manifest=f"{filtered_dir}/manifest.txt"
    params:
        source_dir=source_data,
        output_dir=filtered_dir,
        min_frac_sites_retained=min_frac_sites_retained
    run:
        created_alignments = create_filtered_dataset(
            source_dir=params.source_dir,
            stats_file=input.stats,
            output_dir=params.output_dir,
            min_frac_sites_retained=params.min_frac_sites_retained
        )


rule plot_filtered_stats:
    """Generate plots for the filtered dataset."""
    input:
        manifest=f"{filtered_dir}/manifest.txt",
        stats=f"{source_data}/alignment_size_stats.csv"
    output:
        ratios_plot=f"{filtered_dir}/alignment_size_ratios.pdf",
        sizes_plot=f"{filtered_dir}/cleaned_alignment_sizes.pdf"
    params:
        output_dir=filtered_dir
    run:
        plot_filtered_alignment_stats(input.manifest, input.stats, params.output_dir)


rule create_train_test_split:
    input:
        manifest=f"{filtered_dir}/manifest.txt"
    output:
        train_manifest=f"{train_dir}/manifest.txt",
        test_manifest=f"{test_dir}/manifest.txt"
    params:
        source_dir=source_data,
        train_dir=train_dir,
        test_dir=test_dir,
        test_fraction=test_fraction
    run:
        train_alignments, test_alignments = create_train_test_split(
            source_dir=params.source_dir,
            filtered_manifest=input.manifest,
            train_dir=params.train_dir,
            test_dir=params.test_dir,
            test_fraction=params.test_fraction
        )
