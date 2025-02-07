#!/bin/bash

# This script simulates alignments under JC using the iqtree package alisim
# For each combination of num_sequence and alignment_length_list specified below
# num_alignments will be simulated and saved in a directory in
#  ../../data/simulated_alignments (base_directory in for loops)
# Additionally, config files are produced by calling generate_configs.py
# These can then be used for running the larch pipeline to generate dpvt
# training and testing datasets.

# Parameters
num_alignments=500
num_sequences_list=(10 20)
alignment_length_list=(20 50 100)

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for num_sequences in "${num_sequences_list[@]}"; do
    echo "Number of sequences:" $num_sequences
    for alignment_length in "${alignment_length_list[@]}"; do
        echo "Alignment lengths:" $alignment_length
        base_directory="$(cd "${script_dir}/../../data/simulated_alignments" && pwd)/alisim_alignment_${num_sequences}_seq_${alignment_length}_sites_${num_alignments}_algnmnts"


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

                # Create the alignment and save it as a FASTA file
                iqtree --alisim $alignment_dir/alignment_$i -t RANDOM{yh/$num_sequences} --length $alignment_length --out-format fasta -redo

                mv $alignment_dir/alignment_$i.fa $alignment_dir/alignment_$i.fasta

                echo "Created $alignment_dir/alignment_$i.fasta"
            done

            echo "Generate config file..."
            python ${script_dir}/generate_configs.py $num_sequences $alignment_length $num_alignments
        fi
    done
done
