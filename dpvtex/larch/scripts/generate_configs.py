import json
import os
import argparse


def generate_config_files(num_sequences, num_sites, num_algnmnts):
    """
    Generate config files for running the larch pipeline to generate
    training and testing data for dpvt.
    These are config files for data simulated by alisim using the
    script create_alisim_alignments.sh in `dpvtex/larch/sripts`
    We create one config to use SPRs to make the alignments worse
    and one for random subtrees.
    """
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create output directory if it doesn't exist
    config_dir = os.path.join(script_dir, "..", "configs")
    os.makedirs(config_dir, exist_ok=True)
    
    # Create the dataset name
    dataset_name = f"alisim_alignment_{num_sequences}_seq_{num_sites}_sites_{num_algnmnts}_algnmnts"
    
    for use_spr in [True, False]:
        # Create the config dictionary
        config = {
            "input_data": os.path.join(script_dir, "..", "..", "data", "simulated_alignments", dataset_name),
            "larch_build": os.path.join(script_dir, "..", "..", "..", "..", "larch", "build"),
            "output_data": os.path.join(script_dir, "..", "..", "data"),
            "dataset_name": dataset_name,
            "num_cores": 2,
            "make_worse_spr": use_spr
        }
        
        # Create filename for the config
        filename = f"config_{num_sequences}seq_{num_sites}sites_{num_algnmnts}alignments.json"
        if use_spr:
            filename = filename.replace(".json", "_spr.json")
        filepath = os.path.join(config_dir, filename)
        
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
