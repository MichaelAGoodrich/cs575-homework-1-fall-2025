from network_utilities import _get_degree_count_dictionary #type: ignore
from network_utilities import adjacency_list_to_graph
import networkx as nx

def test_degree_counter_simple() -> None:
    # What I expect
    expected_degree_counter: dict[int,int] = {1:2, 2:1}

    # Create graph
    adjacency_list: dict[int, set[int]] = {3:{1}, 1:{2,3}, 2:{1}}
    G = adjacency_list_to_graph(adjacency_list)

    # when
    actual_degree_counter = dict(_get_degree_count_dictionary(G))

    # then
    assert expected_degree_counter == actual_degree_counter

def test_degree_counter_star() -> None:
    # What I expect
    expected_degree_counter: dict[int,int] = {1:5, 5:1}

    # Create graph
    adjacency_list: dict[int, set[int]] = {1:{2,3,4,5,6}, 
                                           2:{1},
                                           3:{1},
                                           4:{1},
                                           5:{1},
                                           6:{1}}
    G = adjacency_list_to_graph(adjacency_list)

    # when
    actual_degree_counter = dict(_get_degree_count_dictionary(G))

    # then
    assert expected_degree_counter == actual_degree_counter

def test_degree_counter_circulant() -> None:
    # What I expect
    expected_degree_counter: dict[int,int] = {4:8}

    # Create graph
    
    G = nx.circulant_graph(8,[1,2])

    # when
    actual_degree_counter = dict(_get_degree_count_dictionary(G))

    # then
    assert expected_degree_counter == actual_degree_counter