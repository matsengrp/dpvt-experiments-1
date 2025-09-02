#!/bin/bash

# This script simulates alignments under JC using the iqtree package alisim
# For each combination of num_sequence and alignment_length_list specified below
# num_alignments will be simulated and saved in a directory in
#  ../../data/simulated_alignments (base_directory in for loops)
# Additionally, config files are produced by calling generate_configs.py
# These can then be used for running the larch pipeline to generate dpvt
# training and testing datasets.

# Parameters - Test configuration
num_alignments_list=(200 500)
num_sequences_list=(15 25)
alignment_length_list=(200)
# edge_distributions=("constant" "uniform" "treesearch" "random_subtree")  # All edge distribution methods
edge_distributions=("constant" "uniform" "treesearch" "random_subtree")
no_dup_sites="False" # Whether to remove duplicate site patterns in the alignments

max_attempts=20
# How much larger to make the initial alignment to account for cleaning
scaling_factor=2

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if simulated_alignments directory exists, create it if not
simulated_alignments_dir="$(cd "${script_dir}/../../../shared_data" && pwd)/simulated_alignments"
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
            base_directory="$simulated_alignments_dir/alisim_alignment_${target_num_sequences}_seq_${target_alignment_length}_sites_${num_alignments}_algnmnts"
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
                        python ${script_dir}/clean_data.py $alignment_dir/raw_alignment_$i.fasta $alignment_dir/alignment_$i.fasta $alignment_dir/alignment_stats_$i.txt $target_alignment_length $target_num_sequences

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
            # We generate config files for all edge distribution methods
            echo "Generate config files for all edge distribution methods..."
            for edge_dist in "${edge_distributions[@]}"; do
                echo "  Generating config for edge distribution: $edge_dist"
                python ${script_dir}/generate_sim_configs.py $target_num_sequences $target_alignment_length $num_alignments --edge_distribution $edge_dist --remove_duplicate_site_patterns $no_dup_sites
            done
        done
    done
done
