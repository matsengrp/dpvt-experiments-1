import glob
import os

snakefile_dir = workflow.basedir
config_path = os.path.join(snakefile_dir, "config.yaml")

configfile: config_path

input_data=os.path.realpath(config["input_data"])
output_data=config["output_data"]
larch_build=config["larch_build"]
num_iterations=config["num_iterations"]


def get_subdirs(data_dir):
    return [
        f.path.split("/")[-1] for f in os.scandir(data_dir) if f.is_dir()
    ]


rule all:
    input:
        expand(output_data+"/{subdir}/{subdir}.p", subdir=get_subdirs(input_data)),
        input_data+"/alignment_lengths.csv"


rule clean_data:
    input:
        fasta_file=input_data+"/{subdir}/{subdir}.fasta"
    output:
        input_fasta=input_data+"/{subdir}/input.fasta",
        algn_length=input_data+"/{subdir}/cleaned_alignment_length.txt"
    shell:
        """
        echo "Looking for input file at: {input.fasta_file}"
        echo "Will create output file at: {output.input_fasta}"
        if [ -f "{input.fasta_file}" ]; then
            python scripts/clean_data.py "{input.fasta_file}" "{output.input_fasta}" "{output.algn_length}"
        else
            echo "Input file not found."
        fi
        """


checkpoint check_alignment:
    input:
        fasta=input_data+"/{subdir}/input.fasta"
    output:
        fasta=input_data+"/{subdir}/checkpoint.flag"
    shell:
        """
        python scripts/check_size_fasta.py "{input.fasta}" "{output.fasta}"
        """


def get_next_input(wildcards):
    flag_path = input_data+f"/{wildcards.subdir}/checkpoint.flag"
    with open(flag_path, "r") as flag:
        status = flag.read().strip()
    if status == "EMPTY":
        return None
    else:
        return input_data + f"/{wildcards.subdir}/input.fasta"


rule preprocessing:
    input:
        input_flag=lambda wildcards: checkpoints.check_alignment.get(subdir=wildcards.subdir).output
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
        pb=input_data + "/{subdir}/output.pb",
        txt=input_data + "/{subdir}/output.txt",
        vcf=input_data + "/{subdir}/output.vcf",
    output:
        pb=input_data+"/{subdir}/larch-output.pb"
    params:
        log=input_data+"/{subdir}/log"
    shell:
        """
        echo "All input files are present, processing..."
        cd {larch_build}
        ./larch-usher -i {input.pb} -r {input.txt} -v {input.vcf} -o {output.pb} -c {num_iterations} -l params.log
        cd {snakefile_dir}
        """


rule extract_dpvt_data:
    input:
        pb=input_data+"/{subdir}/larch-output.pb"
    output:
        data=output_data+"/{subdir}/{subdir}.p"
    shell:
        """
        python scripts/extract_data_from_hdag.py {input.pb} {output.data}
        """


rule summarize_algn_lengths:
    input:
        length_files=expand(input_data+"/{subdir}/cleaned_alignment_length.txt", subdir=get_subdirs(input_data))
    output:
        all_algn_lengths=input_data+"/alignment_lengths.csv"
    script:
        "scripts/aggregate_alignment_lengths.py"