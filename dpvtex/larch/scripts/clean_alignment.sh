#!/bin/bash
# Bash wrapper for calling clean_data.clean_alignment function
# This is needed when creating alignments via alisim with the create_alisim_alignments.sh script
# Usage: ./clean_alignment.sh <input_file> <output_file> <stats_file> [remove_site_patterns] [target_length] [target_seqs]

if [ $# -lt 3 ]; then
    echo "Usage: $0 <input_file> <output_file> <stats_file> [remove_site_patterns] [target_length] [target_seqs]"
    exit 1
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

input_file="$1"
output_file="$2"
stats_file="$3"
remove_site_patterns="${4:-False}"
target_length="${5:-None}"
target_seqs="${6:-None}"

python -c "
import sys
import os
sys.path.insert(0, '${script_dir}')
from clean_data import clean_alignment
from dpvtex.larch.scripts.pipeline_logger import get_logger

target_length = ${target_length}
target_seqs = ${target_seqs}
remove_site_patterns = '${remove_site_patterns}'.lower() == 'true'

# Create a logger (get_logger will create one or use existing)
data_dir = os.path.dirname('${input_file}')
logger = get_logger(data_dir, dataset_name='clean_alignment_wrapper')

clean_alignment(
    '${input_file}',
    '${output_file}',
    '${stats_file}',
    remove_site_patterns=remove_site_patterns,
    target_length=target_length,
    target_seqs=target_seqs,
    logger=logger
)
"
