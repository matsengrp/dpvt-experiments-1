import glob
import os
import datetime

snakefile_dir = workflow.basedir
config_path = os.path.join(snakefile_dir, "config.yaml")

configfile: config_path

input_data=os.path.realpath(config["input_data"])
output_data=config["output_data"]
larch_build=config["larch_build"]
num_larch_iterations=config["num_larch_iterations"]
dataset_name=config["dataset_name"]


current_date = datetime.datetime.now().strftime("%Y-%m-%d")


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
        input_data+"/data_properties_"+dataset_name+"_"+current_date+".csv",
        dpvt_data=output_data+"/larch_"+dataset_name+"_"+current_date+".p",


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

        # Define a function to check MaxParsimony
        cd {snakefile_dir}
        check_max_parsi() {{
            python scripts/check_max_parsimony.py {params.log}
            return $?
        }}
        cd {larch_build}

        # Run larch-usher
        ./larch-usher -i {input.pb} -r {input.txt} -v {input.vcf} -o {output.pb} -l {params.log} -c {num_larch_iterations}

        # Initialize retry count
        retries=0
        # Maximum of 10 additional runs -- that adds 100 iterations!
        while [ $retries -lt 10 ]; do
            if [ $(check_max_parsi) -eq 1 ]; then
                echo "MaxParsimony changed. Rerunning larch-usher..."
                ./larch-usher -i {output.pb} -o {output.pb} -l {params.log} -c 10
            else
                echo "MaxParsimony remains the same. Exiting loop."
                break
            fi
            retries=$((retries + 1))
        done
        cd {snakefile_dir}
        """


rule extract_dpvt_data:
    input:
        pb=input_data+"/{subdir}/larch-output.pb",
    output:
        data=output_data+"/{subdir}/{subdir}.p",
        num_children_file=output_data+"/{subdir}/num_children_dag_trees.csv"
    shell:
        """
        python {snakefile_dir}/scripts/extract_data_from_hdag.py {input.pb} {output.data} {output.num_children_file}
        """


rule aggregate_training_data:
    input:
        expand(output_data+"/{subdir}/{subdir}.p", subdir=get_subdirs(input_data)),
        length_files=expand(input_data+"/{subdir}/cleaned_alignment_length.txt", subdir=get_subdirs(input_data)),
    output:
        data_props=input_data+"/data_properties_"+dataset_name+"_"+current_date+".csv",
        dpvt_train_data=output_data+"/larch_"+dataset_name+"_"+current_date+"_train.p",
        dpvt_test_data=output_data+"/larch_"+dataset_name+"_"+current_date+"_test.p",
    shell:
        """
        python {snakefile_dir}/scripts/aggregate_training_data.py {output_data} {output.dpvt_train_data} {output.dpvt_test_data} {output.data_props} {input_data}
        """

