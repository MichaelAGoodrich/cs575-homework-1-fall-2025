import networkx as nx # type ignore
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from numpy.typing import NDArray


####################
## Error Handling ##
####################
class IllegalGraphRepresentation(Exception):
    """Class for catching graph representation errors.

    An error occurs if there are no vertices in the graph.

    Attributes:
        none 
    """
    def __init__(
        self,
        message: str = "Graph representation had no vertices",
    ) -> None:
        super().__init__(message)

####################
## Graph Plotting ##
####################
def show_graph(G: nx.Graph) -> None:
    node_positions: dict[int, tuple[float,float]] = nx.nx_pydot.graphviz_layout(G,prog='neato')
    title: str = 'My graph'
    plt.figure()
    ax: Axes = plt.gca()
    ax.set_title(title)
    nx.draw(G, 
        node_positions, 
        node_color = ['y' for node in G.nodes], 
        with_labels = True, 
        node_size = 300, 
        alpha=0.8)
    plt.show()

def show_digraph(G: nx.DiGraph) -> None:
    node_positions: dict[int, tuple[float,float]] = nx.nx_pydot.graphviz_layout(G,prog='neato')
    title = 'My directed graph'
    plt.figure()
    plt.clf()
    ax: Axes = plt.gca()
    ax.set_title(title)
    nx.draw_networkx_nodes(G, 
        node_positions, 
        node_color = ['y' for node in G.nodes], 
        node_size = 300, 
        alpha=0.8)
    nx.draw_networkx_labels(G, node_positions, font_size=15)
    nx.draw_networkx_edges(
        G,
        node_positions,
        connectionstyle='arc3, rad=0.2',
        arrows=True,
        arrowsize = 20,
        width = 1
    )
    plt.show()

####################
## Graph Creation ##
####################
def vertex_edge_sets_to_graph(V: set[int], E: set[tuple[int]]) -> nx.Graph:
    if len(V) == 0:
        raise IllegalGraphRepresentation("Vertex set had no vertices")
    # Create networkx graph
    G = nx.Graph()
    # Add vertices and edges
    G.add_nodes_from(V)
    G.add_edges_from(E)
    return G

def vertex_edge_sets_to_digraph(V: set[int], E: set[tuple[int]]) -> nx.DiGraph:
    if len(V) == 0:
        raise IllegalGraphRepresentation("Vertex set had no vertices")
    # Create networkx graph
    G = nx.DiGraph()
    # Add vertices and edges
    G.add_nodes_from(V)
    G.add_edges_from(E)
    return G

def adjacency_list_to_graph(adjacency_list: dict[int, set[int]]) -> nx.Graph:
    """Create an undirected networkx graph from an adjacency list
    
    Adjacency list must have at least one vertex
    """
    ## Check whether adjacency list is empty ##
    if len(adjacency_list.keys()) == 0:
        raise IllegalGraphRepresentation("Adjacency list had no vertices")
    
    ## Check whether k in V(j) --> j in V(k)
    for vertex1 in adjacency_list.keys():
        for vertex2 in adjacency_list[vertex1]:
            if vertex1 not in adjacency_list[vertex2]:
                raise IllegalGraphRepresentation("Adjacency list for undirected graph does not have all required edges") 
    
    ## Create empty graph ##
    G = nx.Graph()

    ## Add edges from each vertex in adjacency list to all incident vertices
    for vertex1 in adjacency_list.keys():
        for vertex2 in adjacency_list[vertex1]:
            G.add_edge(vertex1, vertex2)
    return G

def adjacency_list_to_digraph(adjacency_list: dict[int, set[int]]) -> nx.DiGraph:
    """Create a directed networkx graph from an adjacency list
    
    Adjacency list must have at least one vertex
    """
    ## Check whether adjacency list is empty ##
    if len(adjacency_list.keys()) == 0:
        raise IllegalGraphRepresentation("Adjacency list had no vertices")
    
    ## Create empty direcgted graph ##
    G = nx.DiGraph()

    ## Add edges from each vertex in adjacency list to all incident vertices
    for vertex1 in adjacency_list.keys():
        for vertex2 in adjacency_list[vertex1]:
            G.add_edge(vertex1, vertex2)
    return G

def adjacency_matrix_to_graph(adjacency_matrix: NDArray) -> nx.Graph:
    ## Check whether adjacency list is empty ##
    if len(adjacency_matrix) == 0:
        raise IllegalGraphRepresentation("Adjacency matrix had no vertices")
    
    ## Check whether k in V(j) --> j in V(k)
    if not np.array_equal(adjacency_matrix, adjacency_matrix.T):
        raise IllegalGraphRepresentation("Adjacency matrix is not symmetric") 
    
    return nx.from_numpy_array(adjacency_matrix)

def adjacency_matrix_to_digraph(adjacency_matrix: NDArray) -> nx.Graph:
    ## Check whether adjacency list is empty ##
    if len(adjacency_matrix) == 0:
        raise IllegalGraphRepresentation("Adjacency matrix had no vertices")
    
    return nx.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph)


