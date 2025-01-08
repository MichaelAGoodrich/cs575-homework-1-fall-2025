from network_utilities import _get_median_distances
from network_utilities import adjacency_list_to_graph
import networkx as nx
import numpy as np

def test_get_median_distance_simple() -> None:
    # What I expect
    expected_distance_dictionary: dict[int,int] = {1:1, 2:1, 3:1}

    # Create graph
    adjacency_list: dict[int, set[int]] = {3:{1}, 1:{2,3}, 2:{1}}
    G = adjacency_list_to_graph(adjacency_list)

    # when
    floating_distance_dictionary = _get_median_distances(G)
    actual_distance_dictionary = {key: int(np.floor(floating_distance_dictionary[key])) 
                                  for key in floating_distance_dictionary.keys()}

    # then
    assert expected_distance_dictionary == actual_distance_dictionary

def test_get_median_distance_star() -> None:
    # What I expect
    expected_distance_dictionary: dict[int,int] = {1:1, 
                                                   2:2, 
                                                   3:2,
                                                   4:2,
                                                   5:2,
                                                   6:2}

    # Create graph
    adjacency_list: dict[int, set[int]] = {1:{2,3,4,5,6}, 
                                           2:{1},
                                           3:{1},
                                           4:{1},
                                           5:{1},
                                           6:{1}}
    G = adjacency_list_to_graph(adjacency_list)

    # when
    floating_distance_dictionary = _get_median_distances(G)
    actual_distance_dictionary = {key: int(np.floor(floating_distance_dictionary[key])) 
                                  for key in floating_distance_dictionary.keys()}

    # then
    assert expected_distance_dictionary == actual_distance_dictionary