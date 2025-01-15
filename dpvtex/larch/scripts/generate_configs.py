import json
import os
import argparse


def generate_config_files(num_sequences, num_sites, num_algnmnts):
    """
    Generate config files for running the larch pipeline to generate
    training and testing data for dpvt.
    These are config files for data simulated by alisim using the
    script create_alisim_alignments.sh in `dpvtex/larch/sripts`
    """
    # Create output directory if it doesn't exist
    os.makedirs("../configs", exist_ok=True)
    
    # Create the dataset name
    dataset_name = f"alisim_alignment_{num_sequences}_seq_{num_sites}_sites_{num_algnmnts}_algnmnts"
    
    # Create the config dictionary
    config = {
        "input_data": f"../data/simulated_alignments/{dataset_name}",
        "larch_build": "../../../larch/build",
        "output_data": "../data",
        "dataset_name": dataset_name,
        "num_cores": 2,
        "make_worse_spr": True
    }
    
    # Create filename for the config
    filename = f"config_{num_sequences}seq_{num_sites}sites_{num_algnmnts}alignments.json"
    filepath = os.path.join("configs", filename)
    
    # Write the JSON file
    if os.path.exists(filepath):
        print(f"Config file {filename} exists already.")
    else:
        with open(filepath, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Generated {filename}")


if __name__ == "__main__":
    # Parse command line arguments (number of sequences and number of sites)
    parser = argparse.ArgumentParser(description="Provide number of sequences, number of sites, and number of alignments.")
    parser.add_argument('num_sequences', type=int, help='Number of sequences.')
    parser.add_argument('num_sites', type=int, help='Number of sites')
    parser.add_argument('num_algnmnts', type=int, help='Number of alignments in dataset')
    args = parser.parse_args()

    generate_config_files(args.num_sequences, args.num_sites, args.num_algnmnts)
