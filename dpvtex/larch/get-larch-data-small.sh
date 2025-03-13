#!/bin/bash

larch_data_dir="larch-data"
data_dir="../data"
larch_data_small_dir="larch-data-small"

# Create larch-data-small directory if it doesn't exist
mkdir -p "$larch_data_small_dir"

# Loop through directories in larch-data
for directory in "$larch_data_dir"/*/; do
    dir_name=$(basename "$directory")
    if [ -d "$data_dir/$dir_name" ]; then
        cp -r "$directory" "$larch_data_small_dir/$dir_name"
    fi
done

echo "Symlinks created successfully."
    