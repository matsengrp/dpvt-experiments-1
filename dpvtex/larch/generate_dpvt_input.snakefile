import glob
import os
import datetime
from dpvtex.larch.scripts.extract_data_from_hdag import extract_data_from_hdag
from dpvtex.larch.scripts.aggregate_training_data import aggregate_data
from dpvtex.larch.scripts.pipeline_logger import get_logger

snakefile_dir = workflow.basedir

# Config file can be specified via --configfile on command line
# If not specified, falls back to config.yaml in the snakefile directory

input_data=os.path.realpath(config["input_data"])
output_data=config.get("output_data", config.get("larch_output"))  # Support both names
larch_command=config.get("larch_command", "larch")  # Default to "larch" command
dataset_name=config["dataset_name"]
remove_site_patterns = config.get("remove_duplicate_site_patterns", False)
balance_by_median_num_MP_trees = config.get("balance_by_median_num_MP_trees", True)  # Default to True - balance trees per alignment
larch_timeout = config.get("larch_timeout", 1800)  # Default timeout: 1800 seconds (30 minutes)

# Support both single edge_distribution (string) and multiple (list)
edge_distributions = config.get("edge_distributions", config.get("edge_distribution", "constant"))
if isinstance(edge_distributions, str):
    edge_distributions = [edge_distributions]

# Tree extraction parameters
max_trees = config.get("max_trees", 200)  # Max trees to extract per alignment
max_spr_moves = config.get("max_spr_moves", 100)  # Max SPR moves per tree
spr_move_divisor = config.get("spr_move_divisor", 10)  # Divisor for constant SPR distribution
subtree_max_attempts = config.get("subtree_max_attempts", 100)  # Max attempts for subtree replacement
subtree_target_non_mp_proportion = config.get("subtree_target_non_mp_proportion", 1/6)  # Target non-MP edge proportion


# Suffix mapping for edge distributions
# Note: "constant" -> "_spr" and "random_subtree" -> "_subtree" for historical reasons
EDGE_DIST_TO_SUFFIX = {
    "constant": "_spr",
    "uniform": "_uniform",
    "treesearch_mimic": "_treesearch_mimic",
    "random_subtree": "_subtree",
}
SUFFIX_TO_EDGE_DIST = {suffix: dist_id for dist_id, suffix in EDGE_DIST_TO_SUFFIX.items()}

dup_sites_suffix = ""
if remove_site_patterns in [True, "True", "true"]:
    dup_sites_suffix = "_no_dup_sites"


def get_subdirs(data_dir):
    """
    Get subdirectories with input fasta files, excluding those with unequal sequence lengths
    and those that timed out during larch processing.

    Checks for the presence of either input.fasta or input_no_dup_sites.fasta to determine
    if preprocessing has completed for a subdirectory.

    Alignments with unequal sequence lengths or larch timeouts cannot be processed,
    so they are automatically filtered out if the exclusion lists exist.
    """
    subdirs_with_input = []
    for entry in os.scandir(data_dir):
        if entry.is_dir() and ".snakemake" not in entry.path:
            subdir_path = entry.path
            # Check for either input.fasta or input_no_dup_sites.fasta
            input_fasta = os.path.join(subdir_path, 'input.fasta')
            input_fasta_no_dup = os.path.join(subdir_path, 'input_no_dup_sites.fasta')
            if os.path.isfile(input_fasta) or os.path.isfile(input_fasta_no_dup):
                subdirs_with_input.append(entry.name)

    # Collect excluded directories from multiple sources
    excluded_dirs = set()

    # Path to the unequal length exclusion list
    exclusion_file = os.path.join(data_dir, "unequal_length_alignments.txt")
    if os.path.exists(exclusion_file):
        with open(exclusion_file, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if line and not line.startswith('#'):
                    excluded_dirs.add(line)

    # Path to the timeout exclusion list
    timeout_file = os.path.join(data_dir, "larch_timeout_alignments.txt")
    if os.path.exists(timeout_file):
        with open(timeout_file, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if line and not line.startswith('#'):
                    excluded_dirs.add(line)

    # Filter out excluded directories
    filtered_subdirs = [d for d in subdirs_with_input if d not in excluded_dirs]

    return filtered_subdirs


rule all:
    input:
        expand(input_data+"/data_properties_"+dataset_name+"{edge_suffix}" + dup_sites_suffix + ".csv",
               edge_suffix=[EDGE_DIST_TO_SUFFIX[ed] for ed in edge_distributions]),
        expand(output_data+"/"+dataset_name+"{edge_suffix}" + dup_sites_suffix + ".p",
               edge_suffix=[EDGE_DIST_TO_SUFFIX[ed] for ed in edge_distributions]),


rule preprocessing:
    input:
        input_data+"/{subdir}/input" + dup_sites_suffix + ".fasta",
    output:
        pb=input_data+"/{subdir}/output" + dup_sites_suffix + ".pb",
        txt=input_data+"/{subdir}/output" + dup_sites_suffix + ".txt",
        vcf=input_data+"/{subdir}/output" + dup_sites_suffix + ".vcf",
    params:
        input_dir=input_data+"/{subdir}/",
        dup_sites_suffix=dup_sites_suffix
    shell:
        """
        cd {snakefile_dir}/setup_larch_inputs
        snakemake --snakefile convert_fasta_to_larch_input.snakefile -d {params.input_dir} --cores 1 --config dup_sites_suffix="{params.dup_sites_suffix}" --rerun-incomplete --unlock
        snakemake --snakefile convert_fasta_to_larch_input.snakefile -d {params.input_dir} --cores 1 --config dup_sites_suffix="{params.dup_sites_suffix}" --rerun-incomplete
        cd {snakefile_dir}
        """


rule run_larch:
    input:
        pb=input_data+"/{subdir}/output" + dup_sites_suffix + ".pb",
        txt=input_data+"/{subdir}/output" + dup_sites_suffix + ".txt",
        vcf=input_data+"/{subdir}/output" + dup_sites_suffix + ".vcf",
    output:
        pb=input_data+"/{subdir}/larch-output" + dup_sites_suffix + ".pb"
    params:
        log=input_data+"/{subdir}/log" + dup_sites_suffix,
        timeout=larch_timeout,
        timeout_file=input_data+"/larch_timeout_alignments.txt",
        subdir="{subdir}"
    run:
        import subprocess
        import sys
        from dpvtex.larch.scripts.pipeline_logger import get_logger

        logger = get_logger(input_data)

        # Build the shell command
        shell_cmd = f"""
        set -e
        # Run larch with timeout
        timeout {params.timeout} {larch_command} -i {input.pb} -r {input.txt} -v {input.vcf} -o {output.pb} -l {params.log} -S
        """

        try:
            result = subprocess.run(
                shell_cmd,
                shell=True,
                executable='/bin/bash',
                capture_output=True,
                text=True
            )

            if result.returncode == 124:  # timeout command returns 124 on timeout
                # Log the timeout
                logger.log("LARCH_TIMEOUT", f"Larch timed out after {params.timeout} seconds for alignment: {params.subdir}")
                print(f"WARNING: Larch timed out for {params.subdir}. Excluding from pipeline.", file=sys.stderr)

                # Append to timeout file so it's excluded from downstream processing
                with open(params.timeout_file, 'a') as f:
                    f.write(f"{params.subdir}\n")

                # Create an empty output file so the rule succeeds (but downstream will skip it)
                with open(output.pb, 'w') as f:
                    f.write("")  # Empty file as marker

            elif result.returncode != 0:
                # Other error - log it
                logger.log("LARCH_ERROR", f"Larch failed with return code {result.returncode} for alignment: {params.subdir}")
                logger.log("LARCH_ERROR", f"STDERR: {result.stderr}")
                print(f"WARNING: Larch failed for {params.subdir}. Excluding from pipeline.", file=sys.stderr)

                # Create empty output to allow pipeline to continue
                with open(output.pb, 'w') as f:
                    f.write("")
        except Exception as e:
            # Log unexpected errors but don't crash
            logger.log("LARCH_ERROR", f"Unexpected error for {params.subdir}: {str(e)}")
            print(f"WARNING: Unexpected error for {params.subdir}: {str(e)}", file=sys.stderr)
            # Create empty output to allow pipeline to continue
            with open(output.pb, 'w') as f:
                f.write("")


rule extract_dpvt_data:
    input:
        pb=input_data+"/{subdir}/larch-output" + dup_sites_suffix + ".pb",
    output:
        data=input_data+"/{subdir}/{subdir}{edge_suffix}" + dup_sites_suffix + ".p",
        num_children_file=input_data+"/{subdir}/num_children_dag_trees{edge_suffix}" + dup_sites_suffix + ".csv"
    wildcard_constraints:
        edge_suffix="|".join(EDGE_DIST_TO_SUFFIX.values())
    run:
        logger = get_logger(input_data)
        edge_dist = SUFFIX_TO_EDGE_DIST[wildcards.edge_suffix]
        extract_data_from_hdag(
            input.pb,
            output.data,
            output.num_children_file,
            edge_distribution=edge_dist,
            logger=logger,
            max_trees=max_trees,
            max_spr_moves=max_spr_moves,
            spr_move_divisor=spr_move_divisor,
            subtree_max_attempts=subtree_max_attempts,
            subtree_target_non_mp_proportion=subtree_target_non_mp_proportion,
        )


def get_extract_inputs(wildcards):
    """Get all per-subdir pickle files for a given edge suffix."""
    subdirs = get_subdirs(input_data)
    return expand(
        input_data+"/{subdir}/{subdir}{edge_suffix}" + dup_sites_suffix + ".p",
        subdir=subdirs,
        edge_suffix=wildcards.edge_suffix
    )

rule aggregate_training_data:
    input:
        pickles=get_extract_inputs,
        size_stats=expand(input_data+"/{subdir}/size_stats" + dup_sites_suffix + ".csv", subdir=get_subdirs(input_data)),
    output:
        data_props=input_data+"/data_properties_"+dataset_name+"{edge_suffix}" + dup_sites_suffix + ".csv",
        dpvt_data=output_data+"/"+dataset_name+"{edge_suffix}" + dup_sites_suffix + ".p",
    wildcard_constraints:
        edge_suffix="|".join(EDGE_DIST_TO_SUFFIX.values())
    run:
        edge_dist = SUFFIX_TO_EDGE_DIST[wildcards.edge_suffix]
        aggregate_data(data_dir=input_data, data_props_file=output.data_props, dpvt_train_data=output.dpvt_data, edge_distribution=edge_dist, dpvt_test_data=None, balance_by_median_num_MP_trees=balance_by_median_num_MP_trees)

