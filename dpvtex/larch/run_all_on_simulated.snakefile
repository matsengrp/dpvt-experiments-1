"""
All-in-one Snakefile for running the complete DPVT pipeline on simulated data.

This snakefile runs all three phases without manual checkpoints:
    Phase 1: Preprocessing (clean alignments)
    Phase 2: Dataset preparation (filter alignments, no train/test split)
    Phase 3: Training data generation (run larch and extract DPVT data)

For simulated data without gaps or ambiguous characters, no manual review of
filtering results is needed, so all phases can run automatically.

Usage:
    1. Generate config with --no-split flag:
       python scripts/generate_configs.py -i <input_data> -d <dataset_name> -l larch --no-split

    2. Run the pipeline:
       snakemake --snakefile run_all_on_simulated.snakefile \\
           --configfile configs/<dataset_name>_prepare.yaml \\
           --cores 8

Note: This snakefile uses the _prepare.yaml config which contains all necessary
parameters for all three phases. The _generate.yaml config is only needed if
running Phase 3 separately.
"""

import os
import sys

snakefile_dir = workflow.basedir
scripts_dir = os.path.join(snakefile_dir, "scripts")
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from utils import EDGE_DIST_TO_SUFFIX, SUFFIX_TO_EDGE_DIST, get_dup_sites_suffix

# Config file can be specified via --configfile on command line
default_config_path = os.path.join(snakefile_dir, "config.yaml")
args = sys.argv
try:
    config_path = os.path.join(snakefile_dir, args[args.index("--configfile") + 1])
except:
    config_path = default_config_path

configfile: config_path

# Read config parameters
input_data = os.path.realpath(config["input_data"])
dataset_name = config["dataset_name"]
remove_site_patterns = config.get("remove_duplicate_site_patterns", False)
min_frac_sites_retained = config.get("min_frac_sites_retained", 0.8)
output_datasets = config.get("output_datasets", "../../data")
output_data = config.get("larch_output", config.get("output_data", "../../data"))
num_cores = config.get("num_cores", 8)
larch_command = config.get("larch_command", "larch")

# Support both single edge_distribution (string) and multiple (list)
edge_distributions = config.get("edge_distributions", config.get("edge_distribution", "constant"))
if isinstance(edge_distributions, str):
    edge_distributions = [edge_distributions]

# New SPR parameters (for constant/uniform edge distributions)
spr_radius = config.get("spr_radius", None)  # None means unlimited
spr_target_non_mp_proportion = config.get("spr_target_non_mp_proportion", 0.1)
max_spr_attempts = config.get("max_spr_attempts", 100)

# Subtree replacement parameters
subtree_max_attempts = config.get("subtree_max_attempts", 100)
subtree_target_non_mp_proportion = config.get("subtree_target_non_mp_proportion", 1/6)

# Legacy SPR parameters (for treesearch_mimic only)
max_spr_moves = config.get("max_spr_moves", 100)
spr_move_divisor = config.get("spr_move_divisor", 10)

# Derived paths - use relative paths to match what prepare_datasets.snakefile produces
filtered_dir = f"{output_datasets}/{dataset_name}_filtered_{min_frac_sites_retained}"
# Absolute path for shell command in Phase 3
filtered_dir_abs = os.path.realpath(os.path.join(snakefile_dir, filtered_dir))
output_data_abs = os.path.realpath(os.path.join(snakefile_dir, output_data))

dup_sites_suffix = get_dup_sites_suffix(remove_site_patterns)

# Final output dataset name (matches what generate_dpvt_input.snakefile expects)
final_dataset_name = f"{dataset_name}_filtered_{min_frac_sites_retained}"

# Path to Phase 3 snakefile
generate_dpvt_snakefile = os.path.join(snakefile_dir, "generate_dpvt_input.snakefile")


rule all:
    """
    Final targets for the complete pipeline:
    - Aggregated DPVT training data pickle files (one per edge distribution)
    - Data properties CSVs (one per edge distribution)
    """
    input:
        # Phase 3 outputs for all edge distributions
        expand(f"{filtered_dir}/data_properties_{final_dataset_name}{{edge_suffix}}" + dup_sites_suffix + ".csv",
               edge_suffix=[EDGE_DIST_TO_SUFFIX[ed] for ed in edge_distributions]),
        expand(f"{output_data}/{final_dataset_name}{{edge_suffix}}" + dup_sites_suffix + ".p",
               edge_suffix=[EDGE_DIST_TO_SUFFIX[ed] for ed in edge_distributions]),


# =============================================================================
# Phase 1: Preprocessing
# =============================================================================
module preprocessing:
    snakefile:
        "preprocess_alignments.snakefile"
    config:
        config


use rule * from preprocessing as preprocessing_*


# =============================================================================
# Phase 2: Dataset Preparation (filtering only, no train/test split)
# =============================================================================
module prepare_datasets:
    snakefile:
        "prepare_datasets.snakefile"
    config:
        config


use rule * from prepare_datasets as prepare_*


# =============================================================================
# Phase 3: Training Data Generation
# =============================================================================
# Phase 3 must be run via shell command (not as a module) because
# generate_dpvt_input.snakefile scans directories at DAG build time,
# and the filtered directory doesn't exist until Phase 2 completes.

rule generate_dpvt_data:
    """Run Phase 3 after Phases 1 and 2 complete (once per edge distribution)."""
    input:
        # Depend on Phase 2 output to ensure it runs first
        manifest=f"{filtered_dir}/manifest.txt"
    output:
        data_props=f"{filtered_dir}/data_properties_{final_dataset_name}{{edge_suffix}}" + dup_sites_suffix + ".csv",
        dpvt_data=f"{output_data}/{final_dataset_name}{{edge_suffix}}" + dup_sites_suffix + ".p",
    wildcard_constraints:
        edge_suffix="|".join(EDGE_DIST_TO_SUFFIX.values())
    threads: num_cores  # Claim all cores to prevent parallel execution (avoids directory lock conflicts)
    params:
        snakefile=generate_dpvt_snakefile,
        # Use absolute paths for shell command
        input_data=filtered_dir_abs,
        output_data=output_data_abs,
        dataset_name=final_dataset_name,
        num_cores=num_cores,
        remove_dup_sites=remove_site_patterns,
        larch_command=larch_command,
        # SPR parameters
        spr_radius=spr_radius,
        spr_target_non_mp_proportion=spr_target_non_mp_proportion,
        max_spr_attempts=max_spr_attempts,
        # Subtree parameters
        subtree_max_attempts=subtree_max_attempts,
        subtree_target_non_mp_proportion=subtree_target_non_mp_proportion,
        # Legacy SPR parameters
        max_spr_moves=max_spr_moves,
        spr_move_divisor=spr_move_divisor
    run:
        edge_dist = SUFFIX_TO_EDGE_DIST[wildcards.edge_suffix]
        # Handle None for spr_radius (convert to "null" for YAML)
        spr_radius_str = "null" if params.spr_radius is None else params.spr_radius
        shell(f"""
        snakemake --snakefile {params.snakefile} \
            --cores {params.num_cores} \
            --config \
                input_data="{params.input_data}" \
                output_data="{params.output_data}" \
                dataset_name="{params.dataset_name}" \
                edge_distribution="{edge_dist}" \
                remove_duplicate_site_patterns={params.remove_dup_sites} \
                larch_command="{params.larch_command}" \
                spr_radius={spr_radius_str} \
                spr_target_non_mp_proportion={params.spr_target_non_mp_proportion} \
                max_spr_attempts={params.max_spr_attempts} \
                subtree_max_attempts={params.subtree_max_attempts} \
                subtree_target_non_mp_proportion={params.subtree_target_non_mp_proportion} \
                max_spr_moves={params.max_spr_moves} \
                spr_move_divisor={params.spr_move_divisor} \
            --rerun-incomplete
        """)
