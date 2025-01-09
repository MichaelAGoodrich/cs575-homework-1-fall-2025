from network_utilities import adjacency_list_to_graph
import networkx as nx
import numpy as np

def test_homework_problem_5() -> None:
    # What I expect
    desired_number_nodes: int = 6
    desired_number_edges = 10
    desired_max_degree = 5
    desired_median_degree = 3

    # when
    ### FIX THIS ADJACENCY LIST
    adjacency_list: dict[int, set[int]] = {1: {2},
                                           2: {3},
                                           3: {4},
                                           4: {5},
                                           5: {1},
                                           6: {5},
                                           7: {6}}
    G = adjacency_list_to_graph(adjacency_list)
    degree_list: list[int] = [y for (_,y) in G.degree]

    # then
    assert nx.is_connected(G)
    assert len(G.nodes()) == desired_number_nodes
    assert len(G.edges) == desired_number_edges
    assert max(degree_list) == desired_max_degree
    assert np.median(degree_list) == desired_median_degree