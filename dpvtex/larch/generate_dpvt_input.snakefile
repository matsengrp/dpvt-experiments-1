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
        input = expand(output_data+"/{subdir}/{subdir}.p", subdir=get_subdirs(input_data))


rule remove_ambiguities:
    input:
        fasta_file=input_data+"/{subdir}/{subdir}.fasta"
    output:
        input_fasta=input_data+"/{subdir}/input.fasta"
    shell:
        """
        echo "Looking for input file at: {input.fasta_file}"
        echo "Will create output file at: {output.input_fasta}"
        if [ -f "{input.fasta_file}" ]; then
            python scripts/remove_ambiguities.py "{input.fasta_file}" "{output.input_fasta}"
        else
            echo "Input file not found."
        fi
        """

rule preprocessing:
    input:
        input_fasta=input_data+"/{subdir}/input.fasta"
    output:
        output_pb=input_data+"/{subdir}/output.pb",
        output_txt=input_data+"/{subdir}/output.txt",
        output_vcf=input_data+"/{subdir}/output.vcf",
    shell:
        """
        snakemake_input=$(readlink -f {input_data})
        cd {snakefile_dir}/setup_larch_inputs
        snakemake --snakefile convert_fasta_to_larch_input.snakefile -d $snakemake_input/{wildcards.subdir} --cores 1
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