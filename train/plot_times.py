import os
import pickle
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# Default values (will be overridden when called from Snakemake or standalone)
nicknames_path = "my_data_nicknames.json"
nicknames_dict = {}
train_benchmark_paths = []
test_benchmark_paths = []

TRAIN_DATA_NICKNAMES = {}
TEST_DATA_NICKNAMES = {}
MODELS = {}
BENCHMARK_PREFIX = ""


def _init_defaults():
    """Initialize default values for standalone script execution."""
    global TRAIN_DATA_NICKNAMES, TEST_DATA_NICKNAMES, MODELS, BENCHMARK_PREFIX
    global nicknames_dict, train_benchmark_paths, test_benchmark_paths

    TRAIN_DATA_NICKNAMES = {
        "simulated_25_seq_100_sites_500_algnmnts_few_spr_filtered_0.8_spr": "sim 25",
        "simulated_50_seq_100_sites_500_algnmnts_few_spr_filtered_0.8_spr": "sim 50",
        "orthomam_train_0.5_1000_samples_spr": "OrthoMaM",
    }

    TEST_DATA_NICKNAMES = {
        "orthomam_test_0.5_spr": "OrthoMaM Test",
        "influenzaC_fluC_M_spr": "flu C M",
        "influenzaC_fluC_NS_spr": "flu C NS",
        "influenzaC_fluC_PB2_spr": "flu C PB2",
        "rotavirusA_H_H2_spr": "rotavirus A H H2",
        "simulated_25_seq_100_sites_200_algnmnts_few_spr_filtered_0.8_spr": "sim 25",
        "simulated_50_seq_100_sites_200_algnmnts_few_spr_filtered_0.8_spr": "sim 50",
        "pandit_full_0.8_spr": "PANDIT",
    }

    MODELS = {
        "TraverseAvgPooling": "Average pooling",
        "TraverseMaxPooling": "Maximum pooling",
        "TraverseNN": "Transformer encoder"
    }
    BENCHMARK_PREFIX = "_output/run.final/benchmark_logs/"

    with open(nicknames_path, 'r') as file:
        nicknames_dict = json.load(file)

    train_benchmark_paths = get_train_benchmark_paths()
    test_benchmark_paths = get_test_benchmark_paths()


def get_benchmark_tsv_paths(model, train_nickname, test_nickname=None, benchmark_type="train_model"):
    """
    Get the path to a benchmark TSV file.

    Args:
        model: Model name (e.g., "TraverseAvgPooling")
        train_nickname: Training data nickname
        test_nickname: Test data nickname (only needed for benchmark_type="test_model")
        benchmark_type: One of "train_model", "test_model", or "optimize_hyperparameters"

    Returns:
        Path to the benchmark TSV file
    """
    if benchmark_type == "test_model":
        if test_nickname is None:
            raise ValueError("test_nickname is required for benchmark_type='test_model'")
        filename = f"{model}-{train_nickname}-ON-{test_nickname}-Param0.tsv"
    else:
        filename = f"{model}-{train_nickname}-Param0.tsv"

    return f"{BENCHMARK_PREFIX}{benchmark_type}/{filename}"


def get_train_benchmark_paths():
    """Get all training benchmark TSV paths for all models and training datasets."""
    paths = []
    for model in MODELS:
        for train_nickname in TRAIN_DATA_NICKNAMES:
            path = get_benchmark_tsv_paths(model, train_nickname, benchmark_type="train_model")
            paths.append(path)
    return paths


def get_test_benchmark_paths():
    """Get all test benchmark TSV paths for all models, training datasets, and test datasets."""
    paths = []
    for model in MODELS:
        for train_nickname in TRAIN_DATA_NICKNAMES:
            for test_nickname in TEST_DATA_NICKNAMES:
                path = get_benchmark_tsv_paths(model, train_nickname, test_nickname, benchmark_type="test_model")
                paths.append(path)
    return paths



def parse_time_to_seconds(time_str):
    """
    Convert time string in h:m:s format to total seconds.

    Args:
        time_str: Time string in format "h:m:s" or "hh:mm:ss"

    Returns:
        Total seconds as float
    """
    parts = time_str.split(':')
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])
    return hours * 3600 + minutes * 60 + seconds


def load_dataset_sizes():
    """
    Load the number of trees in each test dataset from pickle files.

    Returns:
        Dictionary mapping dataset nickname to number of trees
    """
    dataset_sizes = {}
    data_dir = nicknames_dict.get('data_dir', '.')

    for nickname in TEST_DATA_NICKNAMES.keys():
        if nickname not in nicknames_dict:
            print(f"Warning: {nickname} not found in nicknames_dict")
            continue

        file_path = f"{data_dir}/{nicknames_dict[nickname]}"
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            dataset_sizes[nickname] = len(data)
            print(f"Loaded {nickname}: {len(data)} trees")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    return dataset_sizes


def load_dataset_avg_tree_sizes(dataset_nicknames_dict):
    """
    Load the average number of leaves and sites per tree in datasets from pickle files.

    Args:
        dataset_nicknames_dict: Dictionary mapping nicknames to display labels

    Returns:
        Tuple of two dictionaries: (avg_leaves_dict, avg_sites_dict)
        - avg_leaves_dict: Maps dataset nickname to average number of leaves
        - avg_sites_dict: Maps dataset nickname to average number of sites
    """
    avg_tree_sizes = {}
    avg_site_sizes = {}
    data_dir = nicknames_dict.get('data_dir', '.')

    for nickname in dataset_nicknames_dict.keys():
        if nickname not in nicknames_dict:
            print(f"Warning: {nickname} not found in nicknames_dict")
            continue

        file_path = f"{data_dir}/{nicknames_dict[nickname]}"
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)

            # Calculate average number of leaves across all trees
            # add 1 for root leaf
            total_leaves = sum(len(tree)+1 for tree in data.keys())
            avg_leaves = total_leaves / len(data) if len(data) > 0 else 0

            # Calculate average number of sites across all trees
            # Get sequence length from leaves of each tree
            total_sites = 0
            num_trees_with_sites = 0
            for tree in data.keys():
                leaves = tree.get_leaves()
                if len(leaves) > 0:
                    # Get sequence length from first leaf
                    num_sites = len(leaves[0].sequence)
                    total_sites += num_sites
                    num_trees_with_sites += 1

            avg_sites = total_sites / num_trees_with_sites if num_trees_with_sites > 0 else 0

            avg_tree_sizes[nickname] = avg_leaves
            avg_site_sizes[nickname] = avg_sites
            print(f"{nickname}: avg {avg_leaves:.1f} leaves, {avg_sites:.1f} sites per tree")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    return avg_tree_sizes, avg_site_sizes


def load_benchmark_times(paths, is_test=False, dataset_sizes=None):
    """
    Load timing data from benchmark TSV files.

    Args:
        paths: List of paths to benchmark TSV files
        is_test: Whether these are test benchmarks (for label mapping)
        dataset_sizes: Dictionary mapping dataset nickname to number of trees (optional, for test benchmarks)

    Returns:
        DataFrame with columns: model, train_data, test_data (if applicable), time_seconds, time_str, train_label, test_label, num_trees (if dataset_sizes provided)
    """
    data = []

    for path in paths:
        # Check if file exists
        if not Path(path).exists():
            print(f"Warning: File not found: {path}")
            continue

        # Read TSV file
        df = pd.read_csv(path, sep='\t')

        # Extract time from second column (h:m:s)
        time_str = df.iloc[0, 1]  # First row, second column
        time_seconds = parse_time_to_seconds(time_str)

        # Parse filename to extract metadata
        filename = Path(path).stem  # Remove .tsv extension
        parts = filename.split('-')

        model = parts[0]

        # Check if this is a test benchmark (contains "ON")
        if "ON" in filename:
            train_data = parts[1]
            test_data = parts[3]  # After "ON"

            # Get display labels
            train_label = TRAIN_DATA_NICKNAMES.get(train_data, train_data)
            test_label = TEST_DATA_NICKNAMES.get(test_data, test_data)
            model_label = MODELS.get(model, model)

            entry = {
                'model': model,
                'model_label': model_label,
                'train_data': train_data,
                'test_data': test_data,
                'train_label': train_label,
                'test_label': test_label,
                'time_seconds': time_seconds,
                'time_str': time_str
            }

            # Add number of trees if dataset_sizes provided
            if dataset_sizes and test_data in dataset_sizes:
                entry['num_trees'] = dataset_sizes[test_data]

            data.append(entry)
        else:
            train_data = parts[1]
            train_label = TRAIN_DATA_NICKNAMES.get(train_data, train_data)
            model_label = MODELS.get(model, model)

            data.append({
                'model': model,
                'model_label': model_label,
                'train_data': train_data,
                'train_label': train_label,
                'time_seconds': time_seconds,
                'time_str': time_str
            })

    return pd.DataFrame(data)


def plot_training_times(output_path='training_times.pdf'):
    """
    Create a plot of training times for all models and training datasets.

    Args:
        output_path: Path to save the plot
    """
    # Load average tree sizes and sites for training datasets
    print("Loading average tree sizes for training datasets...")
    avg_tree_sizes, avg_site_sizes = load_dataset_avg_tree_sizes(TRAIN_DATA_NICKNAMES)

    # Load data
    df = load_benchmark_times(train_benchmark_paths, is_test=False)

    if df.empty:
        print("No training data found to plot")
        return

    # Convert seconds to minutes for better readability
    df['time_minutes'] = df['time_seconds'] / 60

    # Add average tree size and sites to dataframe
    df['avg_tree_size'] = df['train_data'].map(avg_tree_sizes)
    df['avg_site_size'] = df['train_data'].map(avg_site_sizes)

    # Create labels with tree size and site info
    df['train_label_with_size'] = df.apply(
        lambda row: f"{row['train_label']}\n(n={int(row['avg_tree_size'])}, N={int(row['avg_site_size'])})",
        axis=1
    )

    # Sort by average tree size
    train_label_order = df.groupby('train_label_with_size')['avg_tree_size'].first().sort_values().index.tolist()

    # Create plot
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='train_label_with_size', y='time_minutes', hue='model_label',
                order=train_label_order, palette='Dark2')

    plt.xlabel('Training dataset (avg number of leaves, avg number of sites)', fontsize=16, labelpad=15)
    plt.ylabel('Time (minutes)', fontsize=16)
    plt.title('Training times by model and dataset', fontsize=16)
    plt.xticks(ha='center', fontsize=16)
    plt.legend(title='Model', fontsize=16, title_fontsize=16)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Training times plot saved to {output_path}")
    plt.close()


def plot_testing_times(output_dir='testing_times_per_tree'):
    """
    Create separate plots of testing times per tree for each training dataset.
    Creates one file per training dataset.

    Args:
        output_dir: Directory name (without extension) for output files
    """
    # Load dataset sizes (number of trees in each test dataset)
    print("Loading dataset sizes from pickle files...")
    dataset_sizes = load_dataset_sizes()

    # Load average tree sizes and sites for sorting
    print("Loading average tree sizes for test datasets...")
    avg_tree_sizes, avg_site_sizes = load_dataset_avg_tree_sizes(TEST_DATA_NICKNAMES)

    # Load benchmark timing data
    df = load_benchmark_times(test_benchmark_paths, is_test=True, dataset_sizes=dataset_sizes)

    if df.empty:
        print("No testing data found to plot")
        return

    # Check if we have num_trees column
    if 'num_trees' not in df.columns:
        print("Warning: num_trees not found in dataframe, cannot compute per-tree times")
        return

    # Calculate time per tree in seconds
    df['time_per_tree_seconds'] = df['time_seconds'] / df['num_trees']

    # Add average tree size and sites to dataframe for sorting
    df['avg_tree_size'] = df['test_data'].map(avg_tree_sizes)
    df['avg_site_size'] = df['test_data'].map(avg_site_sizes)

    # Create labels with tree size and site info
    df['test_label_with_size'] = df.apply(
        lambda row: f"{row['test_label']}\n(n={int(row['avg_tree_size'])},\n N={int(row['avg_site_size'])})",
        axis=1
    )

    # Sort test labels by average tree size
    test_label_order = df.groupby('test_label_with_size')['avg_tree_size'].first().sort_values().index.tolist()

    # Create separate plots for each training dataset
    train_labels = df['train_label'].unique()

    for train_label in train_labels:
        df_subset = df[df['train_label'] == train_label]

        # Create individual plot
        fig, ax = plt.subplots(figsize=(12, 6))

        sns.barplot(
            data=df_subset,
            x='test_label_with_size',
            y='time_per_tree_seconds',
            hue='model_label',
            order=test_label_order,
            palette='Dark2',
            ax=ax
        )

        ax.set_xlabel('Test dataset (avg number of leaves n, avg number of site N)', fontsize=16, labelpad=15)
        ax.set_ylabel('Time per tree (seconds)', fontsize=16)
        ax.set_title(f'Testing time per tree - training on {train_label}', fontsize=16)
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)

        # Move legend to top left inside the plot
        ax.legend(title='Model', fontsize=14, title_fontsize=16,
                  loc='upper left')

        # # Rotate x-axis labels
        # for label in ax.get_xticklabels():
        #     label.set_rotation(45)
        #     label.set_ha('right')

        plt.tight_layout()

        # Create filename from train_label (remove spaces and special chars)
        safe_filename = train_label.replace(' ', '_').replace('=', '').replace('\n', '_')
        output_path = f"{output_dir}_{safe_filename}.pdf"

        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Testing times per tree plot saved to {output_path}")
        plt.close()


def plot_testing_times_scatter(output_dir='testing_times_scatter'):
    """
    Create scatter plots of testing times per tree with leaves on y-axis,
    runtime on x-axis, and marker size representing number of sites.
    Creates one file per training dataset.

    Args:
        output_dir: Directory name (without extension) for output files
    """
    # Load dataset sizes (number of trees in each test dataset)
    print("Loading dataset sizes from pickle files...")
    dataset_sizes = load_dataset_sizes()

    # Load average tree sizes and sites
    print("Loading average tree sizes for test datasets...")
    avg_tree_sizes, avg_site_sizes = load_dataset_avg_tree_sizes(TEST_DATA_NICKNAMES)

    # Load benchmark timing data
    df = load_benchmark_times(test_benchmark_paths, is_test=True, dataset_sizes=dataset_sizes)

    if df.empty:
        print("No testing data found to plot")
        return

    # Check if we have num_trees column
    if 'num_trees' not in df.columns:
        print("Warning: num_trees not found in dataframe, cannot compute per-tree times")
        return

    # Calculate time per tree in seconds
    df['time_per_tree_seconds'] = df['time_seconds'] / df['num_trees']

    # Add average tree size and sites to dataframe
    df['avg_tree_size'] = df['test_data'].map(avg_tree_sizes)
    df['avg_site_size'] = df['test_data'].map(avg_site_sizes)

    # Scale marker sizes (normalize sites to reasonable marker sizes)
    # Using a scaling factor so markers are visible but not too large
    min_sites = df['avg_site_size'].min()
    max_sites = df['avg_site_size'].max()
    df['marker_size'] = ((df['avg_site_size'] - min_sites) / (max_sites - min_sites) * 300) + 50

    # Create separate plots for each training dataset
    train_labels = df['train_label'].unique()

    # Define colors for each model (using Dark2 palette)
    all_models = df['model'].unique()
    colors = sns.color_palette('Dark2', n_colors=len(all_models))
    model_colors = dict(zip(all_models, colors))

    for train_label in train_labels:
        df_subset = df[df['train_label'] == train_label]

        # Get only models that have data for this training dataset
        models_in_subset = df_subset['model'].unique()

        # Create individual plot
        fig, ax = plt.subplots(figsize=(14, 10))

        # Plot each model with different colors and offset labels
        model_list = list(models_in_subset)
        for idx, model in enumerate(model_list):
            df_model = df_subset[df_subset['model'] == model]
            model_label = MODELS.get(model, model)
            ax.scatter(
                df_model['avg_tree_size'],
                df_model['time_per_tree_seconds'],
                s=df_model['marker_size'],
                c=[model_colors[model]],
                alpha=0.6,
                edgecolors='black',
                linewidth=1.5,
                label=model_label
            )

            # Add labels only for models that are not TraverseMaxPooling
            if 'MaxPooling' not in model:
                # Add labels for each point with vertical offset based on model index
                # Alternate vertical positions to avoid overlap
                y_offset = -15 if idx % 2 == 0 else 15
                for _, row in df_model.iterrows():
                    ax.annotate(
                        row['test_label'],
                        (row['avg_tree_size'], row['time_per_tree_seconds']),
                        xytext=(5, y_offset),
                        textcoords='offset points',
                        fontsize=12,
                        alpha=0.8
                    )

        ax.set_xlabel('Average Number of Leaves', fontsize=16, labelpad=20)
        ax.set_ylabel('Time per Tree (seconds)', fontsize=16)
        ax.set_title(f'Testing Time per Tree vs Tree Size - Training: {train_label}', fontsize=16)
        ax.tick_params(axis='both', labelsize=16)
        ax.grid(True, alpha=0.3)

        # Create legend for models
        model_legend = ax.legend(title='Model', fontsize=14, title_fontsize=14,
                                  bbox_to_anchor=(1.02, 1), loc='upper left')

        # Add a second legend for marker sizes
        # Create dummy scatter plots for size legend
        size_legend_elements = []
        size_values = [min_sites, (min_sites + max_sites) / 2, max_sites]
        size_labels = [f'{int(s)} sites' for s in size_values]

        for size_val, size_label in zip(size_values, size_labels):
            marker_s = ((size_val - min_sites) / (max_sites - min_sites) * 300) + 50
            size_legend_elements.append(
                plt.scatter([], [], s=marker_s, c='gray', alpha=0.6,
                           edgecolors='black', linewidth=1.5, label=size_label)
            )

        # Add size legend below the model legend
        size_legend = ax.legend(
            handles=size_legend_elements,
            title='Number of Sites',
            fontsize=12,
            title_fontsize=12,
            bbox_to_anchor=(1.02, 0.5),
            loc='upper left'
        )
        ax.add_artist(model_legend)  # Keep both legends

        # Adjust layout to make room for legends
        plt.subplots_adjust(right=0.75)

        # Create filename from train_label
        safe_filename = train_label.replace(' ', '_').replace('=', '').replace('\n', '_')
        output_path = f"{output_dir}_{safe_filename}.pdf"

        # Get both legends as extra artists to include in bbox
        plt.savefig(output_path, dpi=300, bbox_extra_artists=[model_legend, size_legend], bbox_inches='tight')
        print(f"Testing times scatter plot saved to {output_path}")
        plt.close()


def generate_benchmark_plots(
    models,
    train_data_names,
    test_data_names,
    benchmark_dir,
    output_dir,
    nicknames_path_arg,
):
    """
    Generate training and testing time plots using provided configuration.

    This function can be called from Snakemake with parameters from config.yaml.

    Args:
        models: List of model names (e.g., ["TraverseAvgPooling", "TraverseNN"])
        train_data_names: List of training data nicknames
        test_data_names: List of test data nicknames
        benchmark_dir: Path to benchmark_logs directory
        output_dir: Directory to save output plots
        nicknames_path_arg: Path to data_nicknames.json
    """
    # Override global variables with provided config
    global MODELS, TRAIN_DATA_NICKNAMES, TEST_DATA_NICKNAMES, BENCHMARK_PREFIX
    global nicknames_dict, train_benchmark_paths, test_benchmark_paths

    # Model display names
    model_display = {
        "TraverseAvgPooling": "Average pooling",
        "TraverseMaxPooling": "Maximum pooling",
        "TraverseNN": "Transformer encoder"
    }
    MODELS = {name: model_display.get(name, name) for name in models}

    # Use nickname as both key and label
    TRAIN_DATA_NICKNAMES = {name: name for name in train_data_names}
    TEST_DATA_NICKNAMES = {name: name for name in test_data_names}

    BENCHMARK_PREFIX = benchmark_dir
    if not BENCHMARK_PREFIX.endswith('/'):
        BENCHMARK_PREFIX += '/'

    # Load nicknames
    with open(nicknames_path_arg, 'r') as f:
        nicknames_dict = json.load(f)

    # Regenerate benchmark paths
    train_benchmark_paths = get_train_benchmark_paths()
    test_benchmark_paths = get_test_benchmark_paths()

    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)

    # Generate plots
    training_plot_path = os.path.join(output_dir, "training_times.pdf")
    plot_training_times(output_path=training_plot_path)

    testing_plot_prefix = os.path.join(output_dir, "testing_times_per_tree")
    plot_testing_times(output_dir=testing_plot_prefix)

    print(f"All plots saved to {output_dir}")


if __name__ == '__main__':
    # Initialize defaults and generate plots (for standalone use)
    _init_defaults()
    plot_training_times()
    plot_testing_times()
    # plot_testing_times_scatter()
