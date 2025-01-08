#!/bin/bash

# Parameters
num_alignments=200

for num_sequences in 5 10 15; do
    for alignment_length in 20 50 100; do
        base_directory="../../data/simulated_alignments/alisim_alignment_${num_sequences}_seq_${alignment_length}_sites_${num_alignments}_algnmnts"

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
            python generate_configs.py $num_sequences $alignment_length
        fi
    done
done