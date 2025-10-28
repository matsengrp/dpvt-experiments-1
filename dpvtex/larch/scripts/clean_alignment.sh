#!/bin/bash
# Bash wrapper for calling clean_data.clean_alignment function
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
sys.path.insert(0, '${script_dir}')
from clean_data import clean_alignment

target_length = ${target_length}
target_seqs = ${target_seqs}
remove_site_patterns = '${remove_site_patterns}'.lower() == 'true'

clean_alignment(
    '${input_file}',
    '${output_file}',
    '${stats_file}',
    remove_site_patterns=remove_site_patterns,
    target_length=target_length,
    target_seqs=target_seqs
)
"
