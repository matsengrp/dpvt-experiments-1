#!/bin/bash

# Tp run this script, use the following command:
# ./run_tree_searches.sh <path_to_alignment_directory> <path_to_larch_build>

INPUT_DIR=$1
LARCH_BUILD=$2

for FASTA_FILE in $INPUT_DIR/*/*fasta; do
    # Skip the input.fasta file`
    FILENAME=$(basename "$FASTA_FILE")
    if [ "$FILENAME" = "input.fasta" ]; then
        continue
    fi
    echo "Processing $FASTA_FILE"

    TREE_LOG="${FASTA_FILE/.fasta/_log.trees}"
    # Run MP tree search and log trees
    if [ ! -f "$TREE_LOG" ]; then
        echo "Running log_mp_tree_search"
        Rscript log_mp_tree_search.R -f "$FASTA_FILE" -o "$TREE_LOG"
    else
        echo "log_mp_tree_search already run, skipping"
    fi

    CURRENT_DIR=$(pwd)

    LARCH_BUILD_ABS="$(readlink -f "$LARCH_BUILD")"

    # Need to save fasta alignment as input.fasta for larch-usher
    FASTA_FILE_ABS=$(readlink -f "$FASTA_FILE")
    OUTPUT_DIR=$(dirname "$FASTA_FILE_ABS")
    cp "$FASTA_FILE" "$OUTPUT_DIR/input.fasta"

    # # Run larch-usher to generate an hdag
    # echo "Running larch-usher to generate hdag"
    if [ ! -f "$OUTPUT_DIR"/output.pb ]; then
        # Pre-processing for larch-usher
        echo "Pre-processing for larch-usher"
        snakemake --snakefile ../larch/setup_larch_inputs/convert_fasta_to_larch_input.snakefile --configfile ../larch/setup_larch_inputs/config.yaml -d "$OUTPUT_DIR" -c1
    fi

    OUTPUT_DIR_ABS=$(readlink -f "$OUTPUT_DIR")

    if [ ! -f "$OUTPUT_DIR"/larch-output.pb ]; then
        echo "Run larch-usher"
        cd $LARCH_BUILD_ABS
        ./larch-usher -i $OUTPUT_DIR_ABS/output.pb -r $OUTPUT_DIR_ABS/output.txt -v $OUTPUT_DIR_ABS/output.vcf -o $OUTPUT_DIR_ABS/larch-output.pb -l $OUTPUT_DIR_ABS/log -S
        cd $CURRENT_DIR
        echo "Finish running larch-usher"
    fi

    BASENAME=$(basename "$FASTA_FILE" .fasta)
    echo "$OUTPUT_DIR"/"$BASENAME"_test.p
    if [ ! -f "$OUTPUT_DIR"/"$BASENAME"_test.p ]; then
        echo "Compute labels"
        python create_testing_data.py "$OUTPUT_DIR/larch-output.pb" "$OUTPUT_DIR"/"$BASENAME"_test.p "$OUTPUT_DIR/$(basename $TREE_LOG)" "$FASTA_FILE"
        echo "Done computing labels"
    fi

done

echo "All fasta files processed."