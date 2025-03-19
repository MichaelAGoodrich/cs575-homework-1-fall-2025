import networkx as nx # type:ignore
from dendrogram_handler import DendrogramHandler # type:ignore
from dendrogram_handler import Group

G: nx.Graph = nx.path_graph(5)
# The communities for this five-node line graph are
# [({0, 1}, {2, 3, 4}), 
# ({0, 1}, {2}, {3, 4}), 
# ({0}, {1}, {2}, {3, 4}), 
# ({0}, {1}, {2}, {3}, {4})]
# And if we add the original network, we get
# ({0, 1, 2, 3, 4})
#
# The tree hierarchy of the groups from each community is
#      {0,1,2,3,4}
#        /     \
#       /     {2,3,4}
#      /        / \
#   {0,1}      /   \
#    / \      /   {3,4}
#   /   \    /     /  \
# {0}  {1}  {2}  {3}  {4}
#
# The binary tree is 
#       0
#      /  \
#     /    2
#    1     | \
#   / \    |   4
#  /   \   |   | \
# 5     6  3   7  8

def test_group_labels_line_graph() -> None:
    # given
    groups = {0: {0, 1, 2, 3, 4}, 
                1: {0, 1},
                2: {2, 3, 4},
                3: {2}, 
                4: {3, 4}, 
                5: {0},
                6: {1},
                7: {3}, 
                8: {4}}
    answer = {0: (1, 2),
              1: (5, 6),
              2: (3, 4),
              3: (),
              4: (7, 8),
              5: (),
              6: (),
              7: (),
              8: ()}
    
    # when 
    H = DendrogramHandler(G)
    binary_tree: dict[int, list[int]] = H._get_binary_tree(groups)

    # then
    assert binary_tree == answer