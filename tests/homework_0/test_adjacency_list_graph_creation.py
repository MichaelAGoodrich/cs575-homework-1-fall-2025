import pytest
from network_utilities import IllegalGraphRepresentation
from network_utilities import adjacency_list_to_graph
from network_utilities import adjacency_list_to_digraph

def test_empty_adjacency_list_graph_creation() -> None:
    ########################################
    ## Empty adjacency list negative test ##
    ########################################
    # What I expect
    expected_error_message_empty_list: str = "Adjacency list had no vertices"
    
    # Instantiate adjacency list
    empty_adjacency_list: dict[int, set[int]] = dict()
    
    # when
    with pytest.raises(IllegalGraphRepresentation ) as exception:
        _ = adjacency_list_to_graph(empty_adjacency_list)

    # then
    assert expected_error_message_empty_list == str(exception.value)

def test_incorrect_adjacency_list_graph_creation() -> None:
    #################################
    ## Missing edges negative test ##
    #################################
    # What I expect
    expected_error_message_missing_edges: str = "Adjacency list for undirected graph does not have all required edges"

    # Instantiate adjacency list
    missing_edges_adjacency_list: dict[int, set[int]] = {1:{2}, 2:set()}
    
    # when
    with pytest.raises(IllegalGraphRepresentation ) as exception:
        _ = adjacency_list_to_graph(missing_edges_adjacency_list)

    # then
    assert expected_error_message_missing_edges == str(exception.value)

def test_adjacency_list_three_vertex_graph_creation() -> None:
    ################################
    ## Three vertex positive test ##
    ################################
    # What I expect
    expected_vertex_list: list[int] = [1,2,3]
    expected_edge_set: set[tuple[int,int]] = {(1,2), (1,3)}
    
    # Instantiate adjacency list
    adjacency_list: dict[int, set[int]] = {3:{1}, 1:{2,3}, 2:{1}}

    # when
    G = adjacency_list_to_graph(adjacency_list)
    actual_edge_set = set(tuple(sorted(edge)) for edge in G.edges())

    # then
    assert expected_vertex_list == sorted(list(G.nodes()))
    assert actual_edge_set == expected_edge_set

def test_empty_adjacency_list_directed_graph_creation() -> None:
    ########################################
    ## Empty adjacency list negative test ##
    ########################################
    # What I expect
    expected_error_message_empty_list: str = "Adjacency list had no vertices"
    
    # Instantiate adjacency list
    empty_adjacency_list: dict[int, set[int]] = dict()
    
    # when
    with pytest.raises(IllegalGraphRepresentation ) as exception:
        _ = adjacency_list_to_digraph(empty_adjacency_list)

    # then
    assert expected_error_message_empty_list == str(exception.value)

def test_adjacency_list_three_vertex_directed_graph_creation() -> None:
    ################################
    ## Three vertex positive test ##
    ################################
    # What I expect
    expected_vertex_list: list[int] = [1,2,3]
    expected_edge_set: set[tuple[int,int]] = {(1,2), (1,3), (3,1), (2,1)}
    
    # Instantiate adjacency list
    adjacency_list: dict[int, set[int]] = {3:{1}, 1:{2,3}, 2:{1}}

    # when
    G = adjacency_list_to_digraph(adjacency_list)
    actual_edge_set = set(tuple(edge) for edge in G.edges())

    # then
    assert expected_vertex_list == sorted(list(G.nodes()))
    assert actual_edge_set == expected_edge_set
