import pickle


def aggregate_treesearch_data(input_filenames, output_filename):
    """
    Aggregate tree search data from multiple files (i.e. multiple tree searches) into a single file for each alignment.
    Args:
        input_filenames (list): List of file names containing tree search data for the same alignment.
        output_filename (str): Output file name to save the aggregated data.
        
    """

    all_data = {}
    for filename in input_filenames:
        base_name = filename.split("_rep")[0]
        with open(filename, "rb") as f:
            treesearch_data = pickle.load(f)
        all_data |= treesearch_data
    with open(output_filename, "wb") as f:
        pickle.dump(all_data, f)