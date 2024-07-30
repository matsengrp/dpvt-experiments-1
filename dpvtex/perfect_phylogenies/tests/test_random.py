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
    #
    # The root node is 0. The internal nodes are 1, 2, and 3. The leaf nodes are 4, 5,
    # 6, 7, and 8. These integers are the node indices, which are not associated with
    # taxon ids from the newick string.
    return rpp


def test_node_counts_and_indices():
    rpp = setup(False)
    tree = rpp.tree

    # There are 9 nodes total and 3 internal nodes.
    assert rpp.all_node_count == 9
    assert rpp.node_count == 3

    # The node index for a node in the tree is the index of the node in the
    # all_nodes attribute.
    assert all(rpp.all_nodes[node.node_index] == node for node in tree.traverse())

    # The least node index of leaf nodes is the first_leaf_index attribute.
    leaf_start = rpp.first_leaf_index
    assert leaf_start == min(n.node_index for n in tree.get_leaves())

    # Internal nodes are assigned indices before leaf nodes, so the minimum leaf node
    # index is 1 more than the maximum internal node index.
    max_internal_index = max(n.node_index for n in tree.traverse() if not n.is_leaf())
    assert leaf_start == 1 + max_internal_index

    # The internal node are indexed by 1, 2, and 3 and these node indices are stored in
    # the node_indices attribute as the nodes that must have mutations for a perfect
    # phylogeny.
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
        # Manually add each mutation node set to the instance.
        rpp.mut_selections.append(m)
        # Manually increment the mutation counts for nodes in the mutation set.
        rpp.increment_mutation_counts(m)

    # Node 1 has 1 mutation, node 2 has 1 mutation, and node 3 has 3 mutations.
    assert 1 == rpp.mut_counts[rpp.node_index_to_count_index[1]]
    assert 1 == rpp.mut_counts[rpp.node_index_to_count_index[2]]
    assert 3 == rpp.mut_counts[rpp.node_index_to_count_index[3]]

    # Since node 4 is a leaf and rpp is set to require mutations at all internal nodes,
    # there is not an entry for node 4.
    assert 4 not in rpp.node_index_to_count_index.keys()

    return None


def test_superfluous():
    rpp = setup(False)

    m = {3, 4}
    rpp.set_internal_state()
    rpp.mut_selections.append(m)
    rpp.increment_mutation_counts(m)

    # The node indices {1, 3} make the node indices {3, 4} superfluous, because the
    # internal node 3 is covered by {1, 3} and 4 is not an internal node.
    assert rpp.any_previous_mut_set_superfluous({1, 3})
    # The node indices {2, 3} make the node indices {3, 4} superfluous, because the
    # internal node 3 is covered by {1, 3} and 4 is not an internal node.
    assert rpp.any_previous_mut_set_superfluous({2, 3})

    # Set the instance to have mutations at nodes 1 and 4 for a site, and at nodes 2, 6,
    # and 7 for another site.
    rpp.set_internal_state()
    for m in [{1, 4}, {2, 6, 7}]:
        # Manually add the sets.
        rpp.mut_selections.append(m)
        # Manually increment the mutation counts.
        rpp.increment_mutation_counts(m)

    # The node indices {1, 3} make the node indices {1, 4} superfluous, because the
    # internal node 1 is covered by {1, 3} and 4 is not an internal node.
    assert rpp.any_previous_mut_set_superfluous({1, 3})
    # The node index 3 does not make the node indices {1, 4} superfluous, because the
    # internal node 1 is not covered by {3} nor {2, 6, 7}. The node index 3 does not
    # make the node indices {2, 6, 4} superfluous, because the internal node 2 is not
    # covered by {3} nor {1, 4}.
    assert not rpp.any_previous_mut_set_superfluous({3})
    # The node indices {3, 4} do not make the node indices {1, 4} superfluous, because
    # the internal node 1 is not covered by {3, 4} nor {2, 6, 7}. The node indices
    # {3, 4} do not make the node indices {2, 6, 4} superfluous, because the internal
    # node 2 is not covered by {3, 4} nor {1, 4}.
    assert not rpp.any_previous_mut_set_superfluous({3, 4})
    # The node indices {3, 5, 6} do not make the node indices {1, 4} superfluous, because
    # the internal node 1 is not covered by {3, 5, 6} nor {2, 6, 7}. The node indices
    # {3, 5, 6} do not make the node indices {2, 6, 4} superfluous, because the internal
    # node 2 is not covered by {3, 4, 5, 6} nor {1, 4}.
    assert not rpp.any_previous_mut_set_superfluous({3, 5, 6})

    return None


def test_random_perfect_phylogeny():
    rpp = setup(False)
    _ = rpp.make_random_perfect_phylogeny()

    # The internal nodes all have mutations.
    assert all(rpp.mut_counts[rpp.node_index_to_count_index[i]] > 0 for i in (1, 2, 3))

    # No mutation set is a bad pattern (guaranteeing the connected subgraph condition of
    # perfect phylogenies).
    assert all(map(rpp.no_bad_patterns, rpp.mut_selections))

    # No mutation set makes a previous set superfluous. It is not possible for rpp to
    # append a mutation set that is itself superfluous, because a mutation set is chosen
    # to contain an internal node that is not covered by the previous sets.
    while len(rpp.mut_selections) > 0:
        last_mut_set = rpp.mut_selections.pop()
        assert not rpp.any_previous_mut_set_superfluous(last_mut_set)

        previous_indices = set()
        previous_indices.update(*rpp.mut_selections)

        assert any(
            node_index in rpp.node_indices and node_index not in previous_indices
            for node_index in last_mut_set
        )

    return None


def test_labelling():
    rpp = setup(True)

    m0, m1 = {1, 3}, {2, 3}
    rpp.mut_selections = [m0, m1]
    rpp.set_sequence_data()
    # By turning off the permutations, we can determine sequences from the mutation
    # sets. Specifically, m1 gives site 0 by assigning A to the root, G to node 1, and C
    # to node 3; m2 gives site 1 by assigning A to the root, G to node 2, and C to node
    # 3. For other combinations of nodes and sites, the node uses the value of its
    # parent.

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
