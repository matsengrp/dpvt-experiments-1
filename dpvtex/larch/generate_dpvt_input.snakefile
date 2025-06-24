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
make_worse_spr=config["make_worse_spr"]
balance=config["balance"]


# Convert make_worse_spr to boolean if it is a string
if type(make_worse_spr) is str:
    make_worse_spr = make_worse_spr.lower() == "true"
# special suffices if spr moves to introduce non-MP edges
# Convert to boolean if balance is a string
if type(balance) is str:
    unbalanced = balance.lower() != "true"
else:
    unbalanced = balance != True


# Define suffixes based on make_worse_spr and unbalanced flags
if unbalanced and make_worse_spr:
    pickle_suffix = "_spr_unbalanced.p"
    csv_suffix = "_spr_unbalanced.csv"
elif unbalanced and not make_worse_spr:
    pickle_suffix = "_unbalanced.p"
    csv_suffix = "_unbalanced.csv"
elif not unbalanced and make_worse_spr:
    pickle_suffix = "_spr.p"
    csv_suffix = "_spr.csv"
else:
    pickle_suffix = ".p"
    csv_suffix = ".csv"

print("pickle_suffix:", pickle_suffix)
print("csv_suffix:", csv_suffix)

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
        input_data+"/{subdir}/input.fasta",
    output:
        pb=input_data+"/{subdir}/output.pb",
        txt=input_data+"/{subdir}/output.txt",
        vcf=input_data+"/{subdir}/output.vcf",
    params:
        input_dir=input_data+"/{subdir}/"
    shell:
        """
        cd {snakefile_dir}/setup_larch_inputs
        snakemake --snakefile convert_fasta_to_larch_input.snakefile -d {params.input_dir} --cores 1
        cd {snakefile_dir}
        """


rule run_larch:
    input:
        pb=input_data+"/{subdir}/output.pb",
        txt=input_data+"/{subdir}/output.txt",
        vcf=input_data+"/{subdir}/output.vcf",
    output:
        pb=input_data+"/{subdir}/larch-output.pb"
    params:
        log=input_data+"/{subdir}/log"
    shell:
        """
        echo "All input files are present, processing..."
        cd {larch_build}
        # Run larch-usher
        ./larch-usher -i {input.pb} -r {input.txt} -v {input.vcf} -o {output.pb} -l {params.log} -S
        cd {snakefile_dir}
        """


rule extract_dpvt_data:
    input:
        pb=input_data+"/{subdir}/larch-output.pb",
    output:
        data=input_data+"/{subdir}/{subdir}"+pickle_suffix,
        num_children_file=input_data+"/{subdir}/num_children_dag_trees"+csv_suffix
    run:
        extract_data_from_hdag(input.pb, output.data, output.num_children_file, make_worse_spr)


rule aggregate_training_data:
    input:
        expand(input_data+"/{subdir}/{subdir}"+pickle_suffix, subdir=get_subdirs(input_data)),
        length_files=expand(input_data+"/{subdir}/cleaned_alignment_length.txt", subdir=get_subdirs(input_data)),
    output:
        data_props=input_data+"/data_properties_"+dataset_name+csv_suffix,
        dpvt_data=output_data+"/larch_"+dataset_name+pickle_suffix,
    run:
        aggregate_data(data_dir = input_data, data_props_file = output.data_props, dpvt_train_data = output.dpvt_data, spr=make_worse_spr, unbalanced=unbalanced, dpvt_test_data = None)

