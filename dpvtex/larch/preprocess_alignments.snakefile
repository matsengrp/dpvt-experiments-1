import os

snakefile_dir = workflow.basedir
config_path = os.path.join(snakefile_dir, "config.yaml")

configfile: config_path

input_data=os.path.realpath(config["input_data"])


def get_subdirs(data_dir):
    return [
        f.path.split("/")[-1] for f in os.scandir(data_dir) if f.is_dir()
    ]


rule all:
    input:
        expand(input_data+"/{subdir}/checkpoint.done", subdir=get_subdirs(input_data))


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


checkpoint check_alignment_length:
    input:
        fasta=input_data+"/{subdir}/input.fasta",
        algn_length=input_data+"/{subdir}/cleaned_alignment_length.txt"
    output:
        touch(input_data+"/{subdir}/checkpoint.done"),
    params:
        fasta=input_data+"/{subdir}/checkpoint.flag",
    shell:
        """
        python scripts/check_size_fasta.py "{input.fasta}" "{params.fasta}"
        """