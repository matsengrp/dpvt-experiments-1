from itertools import combinations
from ete3 import Tree
from dpvtex.perfect_phylogenies.random_perfect_phylogeny import RandomPerfectPhylogeny


def setup(no_perms=False):
    nwk = "((0,1),((2,3),4));"
    tree = Tree(nwk)

    rpp = RandomPerfectPhylogeny(
        tree,
        edges="internal",
        no_permutations=no_perms,
        size_probs=None,
        random_seed=0,
    )
    # The tree with labelled node indices is:
    #       /-4
    #    /1|
    #   |   \-5
    # -0|
    #   |      /-6
    #   |   /3|
    #    \2|   \-7
    #      |
    #       \-8
    return rpp


def test_node_counts_and_indices():
    rpp = setup(False)
    tree = rpp.tree

    assert rpp.all_node_count == 9
    assert rpp.node_count == 3
    assert all(rpp.all_nodes[node.node_index] == node for node in tree.traverse())

    leaf_start = rpp.first_leaf_index
    assert leaf_start == min(n.node_index for n in tree.get_leaves())
    max_internal_index = max(n.node_index for n in tree.traverse() if not n.is_leaf())
    assert leaf_start == 1 + max_internal_index

    assert {1, 2, 3} == set(rpp.node_indices)

    return None


def test_bad_patterns():
    rpp = setup(False)

    # Bad patterns of the shape:
    #    /-y
    # -x|
    #    \-z
    bad_patterns = [{1, 5, 4}, {2, 3, 8}, {3, 6, 7}]
    # Bad patterns of the shape:
    #       /-x
    # -root|
    #       \-y
    bad_patterns.extend(
        ({1, 2}, {1, 2, 3}, {1, 2, 4}, {1, 2, 5}, {1, 2, 6}, {1, 2, 7}, {1, 2, 8})
    )
    # Bad patterns of the shape:
    #                   /-y
    #       /-some node|
    # -root|            \-z
    #       \-x
    bad_patterns.extend(({1, 3, 8}, {2, 4, 5}))
    assert not any(map(rpp.no_bad_patterns, bad_patterns))

    good_patterns = [
        s
        for k in range(1, 4)
        for combo in combinations(range(1, 9), k)
        if (s := set(combo)) not in bad_patterns
    ]
    assert all(map(rpp.no_bad_patterns, good_patterns))

    return None


def test_mut_counts():
    rpp = setup(False)

    m1, m2, m3 = {1, 3}, {2, 3}, {3, 4}
    rpp.set_internal_state()
    for m in (m1, m2, m3):
        rpp.mut_selections.append(m)
        rpp.increment_mutation_counts(m)

    assert 1 == rpp.mut_counts[rpp.node_index_to_count_index[1]]
    assert 1 == rpp.mut_counts[rpp.node_index_to_count_index[2]]
    assert 3 == rpp.mut_counts[rpp.node_index_to_count_index[3]]

    return None


def test_superfluous():
    rpp = setup(False)

    m = {3, 4}
    rpp.set_internal_state()
    rpp.mut_selections.append(m)
    rpp.increment_mutation_counts(m)
    assert rpp.any_previous_mut_set_superfluous({1, 3})
    assert rpp.any_previous_mut_set_superfluous({2, 3})

    rpp.set_internal_state()
    for m in [{1, 4}, {2, 6, 7}]:
        rpp.mut_selections.append(m)
        rpp.increment_mutation_counts(m)

    assert rpp.any_previous_mut_set_superfluous({1, 3})
    assert not rpp.any_previous_mut_set_superfluous({3})
    assert not rpp.any_previous_mut_set_superfluous({3, 4})
    assert not rpp.any_previous_mut_set_superfluous({3, 5, 6})

    rpp.set_internal_state()
    rpp.make_random_perfect_phylogeny()
    assert all(rpp.mut_counts[rpp.node_index_to_count_index[i]] > 0 for i in (1, 2, 3))

    return None


def test_labelling():
    rpp = setup(True)

    m1, m2 = {1, 3}, {2, 3}
    rpp.mut_selections = [m1, m2]
    rpp.set_sequence_data()
    # By turning off the permutations, we can determine sequences from the mutation
    # sets. Specifically, m1 gives site 0 by assigning A to the root, G to node 1, and C
    # to node 3; m2 gives site 1 by assigning A to the root, G to node 2, and C to node
    # 3. For other nodes and sites, the node uses the value of its parent.

    subs = {
        0: [],
        1: ["A0G"],
        2: ["A1G"],
        3: ["A0C", "G1C"],
        4: [],
        5: [],
        6: [],
        7: [],
        8: [],
    }
    seqs = {
        0: ["A", "A"],
        1: ["G", "A"],
        2: ["A", "G"],
        3: ["C", "C"],
        4: ["G", "A"],
        5: ["G", "A"],
        6: ["C", "C"],
        7: ["C", "C"],
        8: ["A", "G"],
    }
    assert seqs == rpp.node_sequences
    assert subs == rpp.node_substitutions

    return None
