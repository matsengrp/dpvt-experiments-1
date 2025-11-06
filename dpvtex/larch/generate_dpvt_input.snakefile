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
larch_build=config["larch_build"]
dataset_name=config["dataset_name"]
edge_distribution=config.get("edge_distribution", "constant")
remove_site_patterns = config.get("remove_duplicate_site_patterns", False)

# Tree extraction parameters
max_trees = config.get("max_trees", 200)  # Max trees to extract per alignment
max_spr_moves = config.get("max_spr_moves", 100)  # Max SPR moves per tree
spr_move_divisor = config.get("spr_move_divisor", 10)  # Divisor for constant SPR distribution
subtree_max_attempts = config.get("subtree_max_attempts", 100)  # Max attempts for subtree replacement
subtree_target_non_mp_proportion = config.get("subtree_target_non_mp_proportion", 1/6)  # Target non-MP edge proportion


# Define suffixes based on edge_distribution
if edge_distribution == "constant":
    pickle_suffix = "_spr.p"
    csv_suffix = "_spr.csv"
elif edge_distribution == "uniform":
    pickle_suffix = "_uniform.p"
    csv_suffix = "_uniform.csv"
elif edge_distribution == "treesearch_mimic":
    pickle_suffix = "_treesearch_mimic.p"
    csv_suffix = "_treesearch_mimic.csv"
elif edge_distribution == "random_subtree":
    pickle_suffix = "_subtree.p"
    csv_suffix = "_subtree.csv"
else:
    pickle_suffix = ".p"
    csv_suffix = ".csv"


dup_sites_suffix = ""
if remove_site_patterns in [True, "True", "true"]:
    dup_sites_suffix = "_no_dup_sites"
    pickle_suffix = pickle_suffix.replace(".p", "_no_dup_sites.p")
    csv_suffix = csv_suffix.replace(".csv", "_no_dup_sites.csv")


def get_subdirs(data_dir):
    """
    Get subdirectories with input fasta files, excluding those with unequal sequence lengths.

    Checks for the presence of either input.fasta or input_no_dup_sites.fasta to determine
    if preprocessing has completed for a subdirectory.

    Alignments with unequal sequence lengths cannot be processed, so they are automatically
    filtered out if the exclusion list exists.
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

    # Path to the exclusion list
    exclusion_file = os.path.join(data_dir, "unequal_length_alignments.txt")

    # If the exclusion file doesn't exist yet, return all subdirs
    if not os.path.exists(exclusion_file):
        return subdirs_with_input

    # Read the list of directories to exclude
    excluded_dirs = set()
    with open(exclusion_file, 'r') as f:
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
        input_data+"/data_properties_"+dataset_name+csv_suffix,
        output_data+"/larch_"+dataset_name+pickle_suffix,


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
        snakemake --snakefile convert_fasta_to_larch_input.snakefile -d {params.input_dir} --cores 1 --config dup_sites_suffix="{params.dup_sites_suffix}"
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
        log=input_data+"/{subdir}/log" + dup_sites_suffix
    shell:
        """
        echo "All input files are present, processing..."
        # Run larch-usher from bin directory
        {larch_build}/larch-usher -i {input.pb} -r {input.txt} -v {input.vcf} -o {output.pb} -l {params.log} -S
        """


rule extract_dpvt_data:
    input:
        pb=input_data+"/{subdir}/larch-output" + dup_sites_suffix + ".pb",
    output:
        data=input_data+"/{subdir}/{subdir}" + dup_sites_suffix + pickle_suffix,
        num_children_file=input_data+"/{subdir}/num_children_dag_trees" + dup_sites_suffix + csv_suffix
    run:
        logger = get_logger(input_data)
        extract_data_from_hdag(
            input.pb,
            output.data,
            output.num_children_file,
            edge_distribution=edge_distribution,
            logger=logger,
            max_trees=max_trees,
            max_spr_moves=max_spr_moves,
            spr_move_divisor=spr_move_divisor,
            subtree_max_attempts=subtree_max_attempts,
            subtree_target_non_mp_proportion=subtree_target_non_mp_proportion,
        )


rule aggregate_training_data:
    input:
        expand(input_data+"/{subdir}/{subdir}" + dup_sites_suffix + pickle_suffix, subdir=get_subdirs(input_data)),
        size_stats=expand(input_data+"/{subdir}/size_stats" + dup_sites_suffix + ".csv", subdir=get_subdirs(input_data)),
    output:
        data_props=input_data+"/data_properties_"+dataset_name+csv_suffix,
        dpvt_data=output_data+"/larch_"+dataset_name+pickle_suffix,
    run:
        aggregate_data(data_dir = input_data, data_props_file = output.data_props, dpvt_train_data = output.dpvt_data, edge_distribution=edge_distribution, dpvt_test_data = None)

