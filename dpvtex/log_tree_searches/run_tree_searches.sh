#!/bin/bash

for FASTA_FILE in simulated_alignments/*/*.fasta; do
    echo "Processing $FASTA_FILE"
    TREE_LOG="${FASTA_FILE/.fasta/_log.trees}"
    Rscript log_mp_tree_search.R -f "$FASTA_FILE" -o "$TREE_LOG"

    # # Generate the config file in the output directory
    # CONFIG_FILE="${OUTPUT_DIR}/preprocessing_config.yaml"

    # # Get current directory for absolute paths
    CURRENT_DIR=$(pwd)

    # # Set default values for config parameters with absolute paths
    # INPUT_DATA_ABS="$(readlink -f "${CURRENT_DIR}/../data/${DATASET_NAME}/")"
    # LARCH_BUILD_ABS="$(readlink -f "${CURRENT_DIR}/../../../larch/build")"
    # OUTPUT_DATA_ABS="$(readlink -f "${CURRENT_DIR}/../data")"
    # MAKE_WORSE_SPR=True
    # NUM_CORES=5

    # # Generate the config file
    # cat > "$CONFIG_FILE" << EOF
    # input_data: "${INPUT_DATA_ABS}"
    # larch_build: "${LARCH_BUILD_ABS}"
    # output_data: "${OUTPUT_DATA_ABS}"
    # dataset_name: "${DATASET_NAME}"
    # make_worse_spr: ${MAKE_WORSE_SPR}
    # num_cores: ${NUM_CORES}
    # EOF

    # echo "Config file generated at: $CONFIG_FILE"

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

    echo "Compute labels"
    BASENAME=$(basename "$FASTA_FILE" .fasta)
    python create_testing_data.py "$OUTPUT_DIR/larch-output.pb" "$OUTPUT_DIR"/"$BASENAME"_test.p "$OUTPUT_DIR/$TREE_LOG" "$FASTA_FILE"
    echo "Done computing labels"

done

echo "All fasta files processed."