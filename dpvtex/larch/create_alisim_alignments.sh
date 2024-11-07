#!/bin/bash

# Parameters
num_alignments=500
num_sequences=10
alignment_length=10
base_directory="../data/alisim_alignment_${num_sequences}_seq_${alignment_length}_sites_${num_alignments}_algnmnts"

# Create base directory for alignments
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
