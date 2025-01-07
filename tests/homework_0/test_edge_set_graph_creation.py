import pytest
from network_utilities import IllegalGraphRepresentation
from network_utilities import vertex_edge_sets_to_graph
from network_utilities import vertex_edge_sets_to_digraph

def test_empty_vertex_set_graph_creation() -> None:
    ##########################################
    ## Empty adjacency matrix negative test ##
    ##########################################
    # What I expect
    expected_error_message: str = "Vertex set had no vertices"
    
    # Instantiate sets
    V: set[int] = set()
    E: set[set[int]] = set()
    
    # when
    with pytest.raises(IllegalGraphRepresentation ) as exception:
        _ = vertex_edge_sets_to_graph(V,E)

    # then
    assert expected_error_message == str(exception.value)

def test_vertex_set_three_vertex_graph_creation() -> None:
    ################################
    ## Three vertex positive test ##
    ################################
    # What I expect
    expected_vertex_list: list[int] = [1,2,3]
    expected_edge_set: set[tuple[int,int]] = {(1,2), (1,3)}
    
    # Instantiate sets
    V: set[int] = {1,2,3}
    E: set[tuple[int,int]] = {(1,2),(1,3)}

    # when
    G = vertex_edge_sets_to_graph(V,E)
    actual_edge_set = set(tuple(sorted(edge)) for edge in G.edges())

    # then
    assert expected_vertex_list == sorted(list(G.nodes()))
    assert actual_edge_set == expected_edge_set

def test_empty_vertex_set_directed_graph_creation() -> None:
    ##########################################
    ## Empty adjacency matrix negative test ##
    ##########################################
    # What I expect
    expected_error_message: str = "Vertex set had no vertices"
    
    # Instantiate sets
    V: set[int] = set()
    E: set[set[int]] = set()
    
    # when
    with pytest.raises(IllegalGraphRepresentation ) as exception:
        _ = vertex_edge_sets_to_digraph(V,E)

    # then
    assert expected_error_message == str(exception.value)

def test_vertex_set_three_vertex_digraph_creation() -> None:
    ################################
    ## Three vertex positive test ##
    ################################
    # What I expect
    expected_vertex_list: list[int] = [1,2,3]
    expected_edge_set: set[tuple[int,int]] = {(1,2), (2,1), (1,3)}
    
    # Instantiate sets
    V: set[int] = {1,2,3}
    E: set[tuple[int,int]] = {(1,2), (2,1), (1,3)}

    # when
    G = vertex_edge_sets_to_digraph(V,E)
    actual_edge_set = set(tuple(edge) for edge in G.edges())

    # then
    assert expected_vertex_list == sorted(list(G.nodes()))
    assert actual_edge_set == expected_edge_set