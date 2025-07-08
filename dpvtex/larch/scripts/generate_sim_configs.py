import json
import os
import argparse


def generate_sim_config_files(num_sequences, num_sites, num_algnmnts, edge_distribution="constant"):
    """
    Generate config files for running the larch pipeline to generate
    training and testing data for dpvt.
    These are config files for data simulated by alisim using the
    script create_alisim_alignments.sh in `dpvtex/larch/scripts`
    """
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create output directory if it doesn't exist
    config_dir = os.path.join(script_dir, "..", "configs")
    os.makedirs(config_dir, exist_ok=True)
    
    # Create the dataset name
    dataset_name = f"alisim_alignment_{num_sequences}_seq_{num_sites}_sites_{num_algnmnts}_algnmnts"
    
    # Create the config dictionary
    config = {
        "input_data": os.path.join(script_dir, "..", "..", "..", "shared_data", "simulated_alignments", dataset_name),
        "larch_build": os.path.join(script_dir, "..", "..", "..", "..", "larch", "build"),
        "output_data": os.path.join(script_dir, "..", "..", "..", "shared_data"),
        "dataset_name": dataset_name,
        "num_cores": 2,
        "edge_distribution": edge_distribution
    }
    
    # Create filename for the config
    filename = f"config_{num_sequences}seq_{num_sites}sites_{num_algnmnts}alignments_{edge_distribution}.json"
    filepath = os.path.join(config_dir, filename)
        
        # Write the JSON file
        if os.path.exists(filepath):
            print(f"Config file {filename} exists already.")
        else:
            with open(filepath, "w") as f:
                json.dump(config, f, indent=2)
            print(f"Generated {filename}")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate config files for larch pipeline.")
    parser.add_argument('num_sequences', type=int, help='Number of sequences.')
    parser.add_argument('num_sites', type=int, help='Number of sites')
    parser.add_argument('num_algnmnts', type=int, help='Number of alignments in dataset')
    parser.add_argument('--edge_distribution', type=str, 
                        default='constant',
                        choices=['constant', 'uniform', 'treesearch', 'random_subtree'],
                        help='Edge distribution method (default: constant)')
    args = parser.parse_args()

    generate_sim_config_files(args.num_sequences, args.num_sites, args.num_algnmnts, args.edge_distribution)


if __name__ == "__main__":
    main()