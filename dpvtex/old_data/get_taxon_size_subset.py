import pickle
from ete3 import Tree


filename = '../dpvtex/data/larch_harrington-small_2024-06-10.p'
new_filename = 'harrington-small_0_to_50_taxa_subset.p'

with open(filename, "rb") as f:
    data = pickle.load(f)
min_taxa = 0
max_taxa = 50
data_subset = {}
for tree in data:
    if len(tree) > 0 and len(tree) < 50:
        data_subset[tree] = data[tree]

with open(new_filename, "wb") as f:
    pickle.dump(data_subset, f)
