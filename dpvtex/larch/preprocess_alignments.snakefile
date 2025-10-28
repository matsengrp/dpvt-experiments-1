import os
import sys

# Add scripts directory to Python path for imports
snakefile_dir = workflow.basedir
scripts_dir = os.path.join(snakefile_dir, "scripts")
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from clean_data import clean_alignment

config_path = os.path.join(snakefile_dir, "config.yaml")

configfile: config_path

input_data=os.path.realpath(config["input_data"])
remove_site_patterns = config.get("remove_duplicate_site_patterns", False)


def get_subdirs(data_dir):
    return [
        f.path.split("/")[-1] for f in os.scandir(data_dir) if f.is_dir() and ".snakemake" not in f.path
    ]

dup_sites_suffix = ""
if remove_site_patterns in [True, "True", "true"]:
    dup_sites_suffix = "_no_dup_sites"

rule all:
    input:
        expand(input_data+"/{subdir}/checkpoint" + dup_sites_suffix + ".done", subdir=get_subdirs(input_data))


rule clean_data:
    input:
        fasta_file=input_data+"/{subdir}/{subdir}.fasta"
    output:
        input_fasta=input_data+"/{subdir}/input" + dup_sites_suffix + ".fasta",
        algn_length=input_data+"/{subdir}/cleaned_alignment_length" + dup_sites_suffix + ".txt"
    params:
        remove_site_patterns=remove_site_patterns,
    run:
        clean_alignment(
            input.fasta_file,
            output.input_fasta,
            output.algn_length,
            remove_site_patterns=params.remove_site_patterns
        )


checkpoint check_alignment_length:
    input:
        algn_length=input_data+"/{subdir}/cleaned_alignment_length" + dup_sites_suffix + ".txt"
    output:
        touch(input_data+"/{subdir}/checkpoint" + dup_sites_suffix + ".done"),
    params:
        fasta=input_data+"/{subdir}/checkpoint.flag",
    shell:
        """
        python scripts/check_size_fasta.py "{input.algn_length}" "{params.fasta}"
        """