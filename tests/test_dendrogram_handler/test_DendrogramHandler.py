# Construct known graph and Girvan-Newman communities
import networkx as nx
from dendrogram_handler import DendrogramHandler

G = nx.path_graph(5)
commmunities = nx.community.girvan_newman(G)
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
    communities = [({0, 1}, {2, 3, 4}), 
                   ({0, 1}, {2}, {3, 4}), 
                   ({0}, {1}, {2}, {3, 4}), 
                   ({0}, {1}, {2}, {3}, {4})]
    answer = {0: {0, 1, 2, 3, 4}, 
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
    group_labels_dict: dict[int, set[int]] = H._label_groups(communities)
    
    # then
    assert group_labels_dict == answer

def test_group_labels_simple_graph() -> None:
    # given
    communities = [({0}, {1})]
    answer = {0: {0, 1}, 
              1: {0},
              2: {1}
              }
                  
    # when 
    H = DendrogramHandler(G)
    group_labels_dict: dict[int, set[int]] = H._label_groups(communities)
    
    # then
    assert group_labels_dict == answer

