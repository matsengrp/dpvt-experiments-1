import pickle


def aggregate_treesearch_data(filenames):
    """
    Aggregate tree search data from multiple files (i.e. multiple tree searches) into a single file for each alignment.
    Args:
        filenames (list): List of file names containing tree search data.
    """

    data = {}
    print("Aggregating tree search data from files:")
    print(filenames)
    for filename in filenames:
        base_name = filename.split("_rep")[0]
        print(base_name)
        with open(filename, "rb") as f:
            treesearch_data = pickle.load(f)
        if not base_name in data:
            data[base_name] = {}
        data[base_name] |= treesearch_data
    for base_name in data:
        base_name += "_tree_search.p"
        print(base_name)
        with open(base_name, "wb") as f:
            pickle.dump(data[base_name], f)
