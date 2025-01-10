from dpvtex.perfect_phylogenies.perturb_phylogeny import spr_move, make_worse_spr
from ete3 import Tree
from historydag.parsimony import parsimony_score


def test_spr_move():
    tree1 = Tree("((((1,2),3),4),5);")
    tree2 = Tree("((((1,2),4),3),5);")
    tree3 = Tree("(((1,2),(3,4)),5);")
    node1 = tree1.search_nodes(name="1")[0].up
    node2 = tree1.search_nodes(name="4")[0]
    new_tree2 = spr_move(tree1, node1, node2)

    node1 = tree1.search_nodes(name="4")[0]
    node2 = tree1.search_nodes(name="3")[0]
    new_tree3 = spr_move(tree1, node2, node1)

    assert (
        new_tree2.robinson_foulds(tree2)[0] == 0
        and new_tree3.robinson_foulds(tree3)[0] == 0
    )


def test_make_worse():
    tree = Tree("((((AAG,ACT),AGT),CCG),CGG);")
    for node in tree:
        node.add_feature("sequence", node.name)
    worse_tree = make_worse_spr(tree, 2, 100)
    assert parsimony_score(worse_tree) > parsimony_score(tree)
