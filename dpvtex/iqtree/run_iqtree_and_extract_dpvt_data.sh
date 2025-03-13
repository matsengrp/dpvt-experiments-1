#!/bin/bash

# Check if alignment file is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <alignment_file> [nbest]"
    echo "  alignment_file: Path to your alignment file (required)"
    echo "  nbest: Number of best trees to save (optional, default: 100)"
    exit 1
fi

# Get alignment file from command line argument
ALIGNMENT="$1"
NBEST=1

# Check if alignment file exists
if [ ! -f "$ALIGNMENT" ]; then
    echo "Error: Alignment file '$ALIGNMENT' not found!"
    exit 1
fi

# Create directory name based on input filename
BASENAME=$(basename "$ALIGNMENT" | sed 's/\.[^.]*$//')
OUTPUT_DIR="./output/${BASENAME}_iqtree"

# Create output directory
mkdir -p "$OUTPUT_DIR"
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Error: Could not create output directory '$OUTPUT_DIR'!"
    exit 1
fi

echo "Running IQ-TREE on alignment: $ALIGNMENT"
echo "Saving $NBEST locally optimal trees"
echo "Output will be saved in: $OUTPUT_DIR"

# Set output prefix to be inside the directory
PREFIX="${OUTPUT_DIR}/${BASENAME}"

echo "Filename:"
echo "$OUTPUT_DIR/$BASENAME.iqtree"

if [ ! -f "$OUTPUT_DIR/$BASENAME.iqtree" ]; then
    # Run IQ-TREE with settings to save local optima
    iqtree -s "$ALIGNMENT" \
        -pre "$PREFIX" \
        -nt AUTO \
        -nbest "$NBEST" \
        -ninit 1 \
        -m TEST \
        -wt
fi

echo "Analysis complete!"
echo "Locally optimal trees are saved in: ${PREFIX}.treels"



# Generate the config file in the output directory
CONFIG_FILE="${OUTPUT_DIR}/preprocessing_config.yaml"

# Get current directory for absolute paths
CURRENT_DIR=$(pwd)

# Set default values for config parameters with absolute paths
INPUT_DATA_ABS="$(readlink -f "${CURRENT_DIR}/../data/${DATASET_NAME}/")"
LARCH_BUILD_ABS="$(readlink -f "${CURRENT_DIR}/../../../larch/build")"
OUTPUT_DATA_ABS="$(readlink -f "${CURRENT_DIR}/../data")"
MAKE_WORSE_SPR=True
NUM_CORES=5

# Generate the config file
cat > "$CONFIG_FILE" << EOF
input_data: "${INPUT_DATA_ABS}"
larch_build: "${LARCH_BUILD_ABS}"
output_data: "${OUTPUT_DATA_ABS}"
dataset_name: "${DATASET_NAME}"
make_worse_spr: ${MAKE_WORSE_SPR}
num_cores: ${NUM_CORES}
EOF

echo "Config file generated at: $CONFIG_FILE"

# Need to save fasta alignment as input.fasta for larch-usher
cp "$ALIGNMENT" "$OUTPUT_DIR/input.fasta"

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

# if [ ! -f "$OUTPUT_DIR"/"$BASENAME"_test.p ]; then
echo "Compute labels"
python create_testing_data.py "$OUTPUT_DIR/larch-output.pb" "$OUTPUT_DIR"/"$BASENAME"_test.p "$OUTPUT_DIR/$BASENAME.treels"
echo "Done computing labels"
# fi
