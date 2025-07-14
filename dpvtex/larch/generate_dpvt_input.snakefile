import glob
import os
import datetime
from dpvtex.larch.scripts.extract_data_from_hdag import extract_data_from_hdag
from dpvtex.larch.scripts.aggregate_training_data import aggregate_data

snakefile_dir = workflow.basedir
default_config_path = os.path.join(snakefile_dir, "config.yaml")

args = sys.argv

try:
    config_path = os.path.join(snakefile_dir, args[args.index("--configfile") + 1])
except:
    config_path = default_config_path

configfile: config_path

input_data=os.path.realpath(config["input_data"])
output_data=config["output_data"]
larch_build=config["larch_build"]
dataset_name=config["dataset_name"]
edge_distribution=config.get("edge_distribution", "constant")
remove_site_patterns = config.get("remove_duplicate_site_patterns", False)


# Define suffixes based on edge_distribution
if edge_distribution == "constant":
    pickle_suffix = "_spr.p"
    csv_suffix = "_spr.csv"
elif edge_distribution == "uniform":
    pickle_suffix = "_uniform.p"
    csv_suffix = "_uniform.csv"
elif edge_distribution == "treesearch":
    pickle_suffix = "_treesearch.p"
    csv_suffix = "_treesearch.csv"
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
    subdirs_with_flag = []
    for entry in os.scandir(data_dir):
        if entry.is_dir():
            subdir_path = entry.path
            flag_file_path = os.path.join(subdir_path, 'checkpoint.flag')
            if os.path.isfile(flag_file_path):
                subdirs_with_flag.append(entry.name)
    return subdirs_with_flag


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
        {larch_build}/bin/larch-usher -i {input.pb} -r {input.txt} -v {input.vcf} -o {output.pb} -l {params.log} -S
        """


rule extract_dpvt_data:
    input:
        pb=input_data+"/{subdir}/larch-output" + dup_sites_suffix + ".pb",
    output:
        data=input_data+"/{subdir}/{subdir}" + dup_sites_suffix + pickle_suffix,
        num_children_file=input_data+"/{subdir}/num_children_dag_trees" + dup_sites_suffix + csv_suffix
    run:
        extract_data_from_hdag(input.pb, output.data, output.num_children_file, edge_distribution)


rule aggregate_training_data:
    input:
        expand(input_data+"/{subdir}/{subdir}" + dup_sites_suffix +pickle_suffix, subdir=get_subdirs(input_data)),
        length_files=expand(input_data+"/{subdir}/cleaned_alignment_length" + dup_sites_suffix + ".txt", subdir=get_subdirs(input_data)),
    output:
        data_props=input_data+"/data_properties_"+dataset_name+csv_suffix,
        dpvt_data=output_data+"/larch_"+dataset_name+pickle_suffix,
    run:
        aggregate_data(data_dir = input_data, data_props_file = output.data_props, dpvt_train_data = output.dpvt_data, edge_distribution=edge_distribution, dpvt_test_data = None)

