# Construct known graph and Girvan-Newman communities
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


def test_group_labels_line_graph() -> None:
    # given
    all_partitions = [({0, 1}, {2, 3, 4}), 
                   ({0, 1}, {2}, {3, 4}), 
                   ({0}, {1}, {2}, {3, 4}), 
                   ({0}, {1}, {2}, {3}, {4})]
    
    groups = {0: {0, 1, 2, 3, 4}, 
                1: {0, 1},
                2: {2, 3, 4},
                3: {2}, 
                4: {3, 4}, 
                5: {0},
                6: {1},
                7: {3}, 
                8: {4}}
    
    # when 
    H = DendrogramHandler(G)
    group_labels_dict: dict[int, Group] = H._assign_groups_to_tree_nodes(all_partitions)
    # then
    assert groups == group_labels_dict

def test_group_labels_simple_graph() -> None:
    # given
    all_partitions = [({0}, {1})]
    answer = {0: {0, 1}, 
              1: {0},
              2: {1}
              }
                  
    # when 
    H = DendrogramHandler(G)
    group_labels_dict: dict[int, Group] = H._assign_groups_to_tree_nodes(all_partitions)
    
    # then
    assert group_labels_dict == answer

