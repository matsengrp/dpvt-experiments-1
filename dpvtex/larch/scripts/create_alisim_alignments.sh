#!/bin/bash

# This script simulates alignments under JC using the iqtree package alisim
# For each combination of num_sequence and alignment_length_list specified below
# num_alignments will be simulated and saved in a directory in
#  ../../data/simulated_alignments (base_directory in for loops)
# Additionally, config files are produced by calling generate_configs.py
# These can then be used for running the larch pipeline to generate dpvt
# training and testing datasets.
#
# To run the pipeline after generating alignments and configs:
#   snakemake --snakefile run_all_on_simulated.snakefile \
#       --configfile configs/<dataset_name>_prepare.yaml --cores 8
#
# Tree extraction parameters and other settings can be modified in the
# generated YAML config files.

# Parameters - Test configuration

num_alignments_list=(50 100)
num_sequences_list=(15)
alignment_length_list=(20)
edge_distributions=("constant" "random_subtree")
target_nonmp_proportions=(0.1) # Target non-MP edge proportions for perturbation

max_attempts=20
# How much larger to make the initial alignment to account for cleaning
scaling_factor=2

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
config_dir="${script_dir}/../configs"
larch_path="../../../larch/build/bin/larch-usher"

# Check if simulated_alignments directory exists, create it if not
simulated_alignments_dir="${script_dir}/../../../data/simulated_alignments"
if [ ! -d "$simulated_alignments_dir" ]; then
    echo "Creating simulated_alignments directory at: $simulated_alignments_dir"
    mkdir -p "$simulated_alignments_dir"
else
    echo "Found simulated_alignments directory at: $simulated_alignments_dir"
fi

for num_alignments in "${num_alignments_list[@]}"; do
    echo "Number of alignments:" $num_alignments
    for target_num_sequences in "${num_sequences_list[@]}"; do
        echo "Target number of sequences: $target_num_sequences"
        for target_alignment_length in "${alignment_length_list[@]}"; do
            echo "Target alignment length: $target_alignment_length"
            base_directory="$simulated_alignments_dir/simulated_${target_num_sequences}_seq_${target_alignment_length}_sites_${num_alignments}_algnmnts"
            # Create base directory for alignments
            if [ -d "$base_directory" ]; then
                echo "$base_directory exists already."
            else
                echo "Generate alignments for $base_directory."
                mkdir -p "$base_directory"

                for i in $(seq 1 $num_alignments); do
                    # Create directory for each alignment
                    alignment_dir="$base_directory/alignment_$i"
                    mkdir -p "$alignment_dir"
                    # Track whether we've successfully created this alignment
                    success=false
                    attempt=1
                    # Reset scaling factor for each new alignment
                    local_scaling_factor=$scaling_factor

                    while [ "$success" = false ] && [ $attempt -le $max_attempts ]; do
                        echo "Alignment $i, attempt $attempt of $max_attempts"

                        # Calculate scaled-up parameters to account for cleaning
                        initial_num_sequences=$((target_num_sequences + 5))
                        initial_alignment_length=$((target_alignment_length * local_scaling_factor))

                        # Create the alignment and save it as a FASTA file
                        iqtree --alisim $alignment_dir/raw_alignment_$i -t RANDOM{yh/$initial_num_sequences} -m "JC" --length $initial_alignment_length --out-format fasta -redo
                        mv $alignment_dir/raw_alignment_$i.fa $alignment_dir/raw_alignment_$i.fasta

                        # Clean the alignment and trim to exact target dimensions
                        ${script_dir}/clean_alignment.sh $alignment_dir/raw_alignment_$i.fasta $alignment_dir/alignment_$i.fasta $alignment_dir/alignment_stats_$i.txt False $target_alignment_length $target_num_sequences

                        # Check if cleaned alignment meets criteria
                        IFS=',' read cleaned_length cleaned_seqs <$alignment_dir/alignment_stats_$i.txt

                        if [ $cleaned_length -ge $target_alignment_length ] && [ $cleaned_seqs -ge $target_num_sequences ]; then
                            echo "Success! Cleaned alignment has $cleaned_seqs sequences and $cleaned_length sites."
                            success=true
                        else
                            echo "Failed: Cleaned alignment has only $cleaned_seqs sequences and $cleaned_length sites. Retrying..."
                            # Increase local scaling factor with each failed attempt
                            local_scaling_factor=$((local_scaling_factor + 1))
                            attempt=$((attempt + 1))
                        fi
                    done

                    if [ "$success" = false ]; then
                        echo "Failed to create alignment $i after $max_attempts attempts. Skipping."
                        rm -rf $alignment_dir # Clean up the failed attempt
                    else
                        echo "Created $alignment_dir/alignment_$i.fasta"
                    fi
                done
            fi
            # Generate config file with all edge distribution methods
            echo "Generating config with edge distributions: ${edge_distributions[*]}"
            echo "Target non-MP proportions: ${target_nonmp_proportions[*]}"
            dataset_name="simulated_${target_num_sequences}_seq_${target_alignment_length}_sites_${num_alignments}_algnmnts"
            # Build -e flags for all edge distributions
            edge_flags=""
            for edge_dist in "${edge_distributions[@]}"; do
                edge_flags="$edge_flags -e $edge_dist"
            done
            # Build -p flags for all target non-MP proportions
            proportion_flags=""
            for proportion in "${target_nonmp_proportions[@]}"; do
                proportion_flags="$proportion_flags -p $proportion"
            done
            python ${script_dir}/generate_configs.py \
                -i "${base_directory}" \
                -d "${dataset_name}" \
                -l "${larch_path}" \
                -o "${config_dir}" \
                $edge_flags \
                $proportion_flags \
                --no-split
        done
    done
done
