#!/bin/bash

R_SCRIPT=log_mp_tree_search.R

for FASTA_FILE in simulated_alignments/*/*.fasta; do
    echo "Processing $FASTA_FILE"
    OUTPUT_FILE="${FASTA_FILE/.fasta/_log.trees}"
    Rscript "$R_SCRIPT" -f "$FASTA_FILE" -o "$OUTPUT_FILE"
done

echo "All fasta files processed."