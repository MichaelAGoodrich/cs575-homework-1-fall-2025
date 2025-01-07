from network_utilities import _get_degree_count_dictionary #type: ignore
from network_utilities import adjacency_list_to_graph

def test_degree_counter_simple() -> None:
    # What I expect
    expected_degree_counter: dict[int,int] = {1:2, 2:1, 3:1}

    # Create graph
    adjacency_list: dict[int, set[int]] = {3:{1}, 1:{2,3}, 2:{1}}
    G = adjacency_list_to_graph(adjacency_list)

    # when
    actual_degree_counter = _get_degree_count_dictionary(G)

    # then
    assert expected_degree_counter == actual_degree_counter