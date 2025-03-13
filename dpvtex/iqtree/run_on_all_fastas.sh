#!/bin/bash

DIRECTORY="./input_alignments"

# Process each FASTA file
for file in "$DIRECTORY"/*.fasta; do
    echo "Processing: $file"
    ./run_iqtree_and_extract_dpvt_data.sh "$file"
    echo "--------------------------------------"
done

echo "All FASTA files processed"