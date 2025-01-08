import pytest
import numpy as np
from network_utilities import IllegalGraphRepresentation
from network_utilities import adjacency_matrix_to_graph
from network_utilities import adjacency_matrix_to_digraph

def test_empty_adjacency_matrix_graph_creation() -> None:
    ##########################################
    ## Empty adjacency matrix negative test ##
    ##########################################
    # What I expect
    expected_error_message: str = "Adjacency matrix had no vertices"
    
    # Instantiate adjacency matrix
    empty_adjacency_matrix = np.array([])
    
    # when
    with pytest.raises(IllegalGraphRepresentation ) as exception:
        _ = adjacency_matrix_to_graph(empty_adjacency_matrix)

    # then
    assert expected_error_message == str(exception.value)

def test_asymetric_adjacency_list_matrix_creation() -> None:
    #################################
    ## Missing edges negative test ##
    #################################
    # What I expect
    expected_error_message: str = "Adjacency matrix is not symmetric"

    # Instantiate adjacency list
    missing_edges_adjacency_matrix = np.array([[0,1],[0,0]])
    
    # when
    with pytest.raises(IllegalGraphRepresentation ) as exception:
        _ = adjacency_matrix_to_graph(missing_edges_adjacency_matrix)

    # then
    assert expected_error_message == str(exception.value)

def test_adjacency_matrix_three_vertex_graph_creation() -> None:
    ################################
    ## Three vertex positive test ##
    ################################
    # What I expect
    expected_vertex_list: list[int] = [0,1,2]
    expected_edge_set: set[tuple[int,int]] = {(0,1), (0,2)}
    
    # Instantiate adjacency list
    adjacency_matrix = np.array([[0,1,1],[1,0,0],[1,0,0]])

    # when
    G = adjacency_matrix_to_graph(adjacency_matrix)
    actual_edge_set = set(tuple(sorted(edge)) for edge in G.edges())

    # then
    assert expected_vertex_list == sorted(list(G.nodes()))
    assert actual_edge_set == expected_edge_set

def test_empty_adjacency_matrix_directed_graph_creation() -> None:
    ########################################
    ## Empty adjacency list negative test ##
    ########################################
    # What I expect
    expected_error_message_empty_list: str = "Adjacency matrix had no vertices"
    
    empty_adjacency_matrix = np.array([])
    
    # when
    with pytest.raises(IllegalGraphRepresentation ) as exception:
        _ = adjacency_matrix_to_digraph(empty_adjacency_matrix)
    
    # then
    assert expected_error_message_empty_list == str(exception.value)

def test_adjacency_matrix_three_vertex_directed_graph_creation() -> None:
    ################################
    ## Three vertex positive test ##
    ################################
    # What I expect
    expected_vertex_list: list[int] = [0,1,2]
    expected_edge_set: set[tuple[int,int]] = {(0,1), (0,2), (1,2), (2,1)}
    
    # Instantiate adjacency list
    adjacency_matrix = np.array([[0,1,1],[0,0,1],[0,1,0]])

    # when
    G = adjacency_matrix_to_digraph(adjacency_matrix)
    actual_edge_set = set(tuple(edge) for edge in G.edges())

    # then
    assert expected_vertex_list == sorted(list(G.nodes()))
    assert actual_edge_set == expected_edge_set
