#############################
## Visualization Utilities ##
#############################

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import networkx as nx
import numpy as np
from numpy.typing import NDArray
from typing import Hashable, Tuple, Set, List
from dendrogram_handler_v2 import DendrogramHandler
from scipy.cluster.hierarchy import dendrogram # type: ignore
import random
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix, diags
from copy import deepcopy
from typing import Callable as function


def draw_edge_by_type(G: nx.Graph, 
                      pos: dict[Hashable, Tuple[float, float]], 
                      edge: Tuple[Hashable, Hashable], 
                      partition: Tuple[Set[Hashable], ...]
                      ) -> None:
    """
        Draw edges between nodes in different partitions using dashed lines.
        Draw edges between nodes within the same partition using solid lines.
    """
    edge_style = 'dashed'
    for part in partition:
        if edge[0] in part and edge[1] in part:
            edge_style = 'solid'
            break
    nx.draw_networkx_edges(G, pos, edgelist=[edge], style = edge_style)

def count_edges_cut(G: nx.Graph,
                    partition: Tuple[Set[Hashable], ...]
                    ) -> int:
    """ 
        Count the number of edges cut if the nodes in graph G are split into 
        the groups in the partition
    """
    cut_size:int = 0
    for i in range(len(partition) - 1):
        for j in range(i+1, len(partition)):
            for u in partition[i]:
                for v in G.neighbors(u):
                    if v in partition[j]:
                        cut_size += 1
    return cut_size

def show_graph(G: nx.Graph,
                    pos: dict[Hashable, Tuple[float, float]] | None = None,
                    title: str = ""
                    ) -> None:
    """ 
        Show the networkx graph 
    """
    
    if pos is None: 
        #pos = nx.spring_layout(G, seed = 0)
        pos = nx.nx_pydot.pydot_layout(G, prog = "neato")
    nx.draw(G, pos, node_color = 'lightblue', alpha=0.8, with_labels=True)
    plt.title(title)
    plt.axis('off')

def show_partitions(G: nx.Graph,
                    partition: Tuple[Set[Hashable], ...], 
                    pos: dict[Hashable, Tuple[float, float]] | None = None,
                    title: str = ""
                    ) -> None:
    """ 
        Show the networkx graph with colors and edges indicating properties
        of the partition

        Edges:
        • Dashed lines indicate edges between nodes in different partitions
        • Solid lines indicate edges between nodes in the same partition

        Nodes:
        • All nodes in the same partition get mapped to the same color
        • When there are more partitions than ther are in the color pallette, repeat colors
    """
    #color_list = ['c','m','y','g','r']
    color_list: list[str] = ['y', 'lightblue', 'violet', 'salmon', 
                         'aquamarine', 'magenta', 'lightgray', 'linen']
    plt.clf()
    ax: Axes = plt.gca()
    if pos is None: 
        #pos = nx.spring_layout(G, seed = 0)
        pos = nx.nx_pydot.pydot_layout(G, prog = "neato")
    for i in range(len(partition)):
        nx.draw_networkx_nodes(partition[i],pos,node_color=color_list[i%len(color_list)], alpha = 0.8)
    for edge in G.edges:
        draw_edge_by_type(G, pos, edge, partition)
    nx.draw_networkx_labels(G,pos)
    if len(G.edges) == 0:
        mod = 0
    else:
        mod = nx.algorithms.community.quality.modularity(G,partition)
    if title[-1] == ":" or title[-1] == "\n":
        title = title + " groups=" + str(len(partition))
    else:
        title = title + ", groups=" + str(len(partition))
    title = title + ", edges cut=" + str(count_edges_cut(G, partition))
    title = title + ", mod = " + str(np.round(mod,2))

    ax.set_title(title)
    ax.set_axis_off()

def show_dendrogram(G: nx.Graph,
                    title: str = "Dendrogram") -> None:
    plt.figure()
    myHandler: DendrogramHandler = DendrogramHandler(G)
    Z = myHandler.link_matrix       # Python style guides suggest direct access of public class variables
    ZLabels = myHandler.link_matrix_labels
    #plt.figure(figureNumber);plt.clf()
    dendrogram(Z, labels=ZLabels)
    plt.title(title)
    plt.xlabel("Node")
    plt.ylabel("Number of nodes in cluster")
    del myHandler


###############################################
## Newman Modularity Hill Climbing Utilities ##
###############################################
def split_into_random_shores(G: nx.Graph
                             ) -> Tuple[Set[Hashable], Set[Hashable]]:
    """ 
        The Newman algorithm for random and greedy hill-climbing 
        starts with nodes assigned randomly two two shores.
    """
    shore_size: int = np.ceil(len(G.nodes()))/2
    shore1: set[Hashable] = set(G.nodes)
    shore2: set[Hashable] = set()
    
    while len(shore2) < shore_size:
        node: Hashable = random.choice(list(shore1))
        shore2.add(node)
        shore1.remove(node)
    return (shore1, shore2)

def swap_shores(partition: Tuple[Set[Hashable], Set[Hashable]], 
                node: Hashable
                ) -> Tuple[Set[Hashable], Set[Hashable]]:
    """ 
        Swapping shores means moving a node from one
        partition to another.
    """
    shore1: Set[Hashable] = deepcopy(partition[0])
    shore2: Set[Hashable] = deepcopy(partition[1])
    if node in partition[0]:
        shore1.remove(node)
        shore2.add(node)
    else:
        shore2.remove(node)
        shore1.add(node)
    return (shore1, shore2)

def find_best_node_to_swap(G: nx.Graph,
                           partition: Tuple[Set, Set],
                           already_swapped: Set
                           ) -> Hashable | None:
    best_mod: float = -np.inf
    # Node that produces the highest modularity increase if it swaps shores
    best_node_to_swap: Hashable | None = None  
    # Track nodes that have already been swapped
    for node in set(G.nodes()) - already_swapped:
        possible_partition = swap_shores(partition, node)
        mod = nx.algorithms.community.quality.modularity(G,possible_partition)
        if mod > best_mod:
            best_mod = mod
            best_node_to_swap = node
    return best_node_to_swap

def Newman_hill_climbing(G: nx.Graph
                         ) -> Tuple[Set[Hashable], Set[Hashable]]:
    """ 
        Implement Newman's hill climbing algorithm for estimating
        the partition that maximizes modularity.

        Returns:
            The best partition found
    """
    # Initialize
    partition: Tuple[Set[Hashable], Set[Hashable]] = split_into_random_shores(G)
    already_swapped: set[Hashable] = set()
    best_partition: Tuple[Set[Hashable], Set[Hashable]] = deepcopy(partition)
    best_modularity: float = nx.community.modularity(G, partition)
    
    best_node_to_swap: Hashable| None = find_best_node_to_swap(G, partition, already_swapped)
    while best_node_to_swap is not None:
        partition = swap_shores(partition, best_node_to_swap)
        already_swapped.add(best_node_to_swap)
        
        if nx.community.modularity(G, partition) >= best_modularity:
            best_modularity = nx.community.modularity(G, partition)
            best_partition = deepcopy(partition)
        else:
            return best_partition  # Stop when modularity starts going down

        best_node_to_swap = find_best_node_to_swap(G, partition, already_swapped)

    return best_partition

#######################################
## Spectral Modularity Cut Utilities ##
#######################################
def get_leading_eigenvector(G: nx.Graph
                            ) -> Tuple[float, NDArray[np.float32]]:
    
    B: NDArray[np.float32] = nx.modularity_matrix(G, nodelist=sorted(G.nodes()))
    eigenvalues, eigenvectors = np.linalg.eig(B)
    largest_eigenvalue_index = np.argmax(eigenvalues)
    largest_eigenvalue = eigenvalues[largest_eigenvalue_index]
    leading_eigenvector = eigenvectors[:, largest_eigenvalue_index]
    return largest_eigenvalue, leading_eigenvector

def get_shores_from_eigenvector(G: nx.Graph,
                                x: NDArray[np.float32]) -> Tuple[Set[Hashable], Set[Hashable]]:
    shore1: Set[Hashable] = set()
    shore2: Set[Hashable] = set()
    nodes = sorted(list(G.nodes()))
    for i in range(len(nodes)):
        if x[i] >= 0: 
            shore1.add(nodes[i])
        else: 
            shore2.add(nodes[i])
    return (shore1, shore2)

def modularity_spectral_split(G: nx.Graph) -> Tuple[Set[Hashable], Set[Hashable]]:
    _, v = get_leading_eigenvector(G)
    return get_shores_from_eigenvector(G,v)

###########################################
## Kernighan-Lin Hill-Climbing Graph Cut ##
###########################################
def initialize_partition(G: nx.Graph,
                         seed: int | None = None
                         ) -> Tuple[set[Hashable], set[Hashable]]:
    """
        Input: networkx undirected graph
        Output: two sets with the graph nodes split in half
    """
    # Check types
    if type(G) is not nx.Graph:
        raise TypeError("Requires undirected graph")
    
    # Initialize partitions
    nodes: list[Hashable] = list(G.nodes())
    if seed is not None:
        random.seed(seed)
    random.shuffle(nodes)
    mid: int = len(nodes) // 2
    A: set[Hashable] = set(nodes[:mid])
    B: set[Hashable] = set(nodes[mid:])

    return (A,B)

def gain(G: nx.Graph,
         u: Hashable, 
         group_A: set[Hashable], 
         group_B: set[Hashable]
         ) -> int:
    """ 
        count the net gain in the number of edges cut if we swap node u
        from group A to group B. See D_a from https://en.wikipedia.org/wiki/Kernighan-Lin_algorithm
    """

    internal = sum(1 for v in G.neighbors(u) if v in group_A)
    external = sum(1 for v in G.neighbors(u) if v in group_B)
    return external - internal

def gain_from_swap(G: nx.Graph,
                   u: Hashable,
                   v: Hashable,
                   group_A: set[Hashable],
                   group_B: set[Hashable]
                   ) -> int:
    """ 
        Compute the net gain from swapping a and b using the equation
        T_{old} - T_{new} = D(a,A) + D(b,B) - 2 delta({a,b} in E)
    """
    gain_u: int = gain(G, u, group_A, group_B)
    gain_v: int = gain(G, v, group_B, group_A)
    delta: int = int(v in G.neighbors(u))

    return gain_u + gain_v - 2*delta

def kernighan_lin_bisection(G: nx.Graph, 
                            max_iter: int=100,
                            seed: int | None = None
                            ) -> Tuple[Set[Hashable], Set[Hashable]]:
    """
        Input: undirected graph
    """
    # Check types
    if type(G) is not nx.Graph:
        raise TypeError("Requires undirected graph")
    
    # Initialize
    group_A, group_B = initialize_partition(G, seed)
    
    # Compute scores for all swaps
    for _ in range(max_iter):
        gains: List[Tuple[int, Hashable, Hashable]] = []
        for u in group_A:
            for v in group_B:
                swap_score: int = gain_from_swap(G, u, v, group_A, group_B)
                gains.append((swap_score, u, v))
        
        gains.sort(reverse=True)
        
        best_gain: int = 0
        best_pair: tuple[Hashable, Hashable] | None = None
        current_gain: int = 0
        for gain_value, u, v in gains:
            current_gain += gain_value
            if current_gain > best_gain:
                best_gain = current_gain
                best_pair = (u, v)
        
        if best_pair is not None:
            u, v = best_pair
            group_A.remove(u)
            group_B.add(u)
            group_B.remove(v)
            group_A.add(v)
        else:
            break
    
    return group_A, group_B

################################
## Minimum Balanced Graph Cut ##
################################

def get_fiedler_eigenvector_sparse(L: csr_matrix) -> NDArray[np.float32]:
    """
        Computes the numerically stable Fiedler eigenvector for a given sparse 
        Laplacian matrix. Generated by chatGPT in response to some numerical 
        stability problems that arise when some of the vertices in the graph have 
        degree one.
    """
    eigenvectors: NDArray[np.float32]
    _, eigenvectors = eigsh(L, k=2, which="SM")  # Compute two smallest eigenvalues
    return eigenvectors[:, 1]  # Return the second smallest eigenvector

def get_fiedler_eigenvector(Laplacian: NDArray[np.float32]
                            ) -> NDArray[np.float32]:
    
    eigenvalues, eigenvectors = np.linalg.eig(Laplacian)
    # choose second smallest as fiedler eigenvalue
    sorted_indices = np.argsort(eigenvalues)
    # return eigenvector of second smallest index
    return(eigenvectors[:,sorted_indices[1]])

def laplacian_graph_cut(G: nx.Graph) -> Tuple[Set[Hashable], Set[Hashable]]:
    L = nx.laplacian_matrix(G, nodelist=sorted(G.nodes())).toarray()
    v = get_fiedler_eigenvector(L)
    return get_shores_from_eigenvector(G,v)

def laplacian_graph_cut_sparse(G: nx.Graph,
                               get_shores: function[[nx.Graph, NDArray[np.float32]], Tuple[Set[Hashable], Set[Hashable]]] = get_shores_from_eigenvector
                               ) -> Tuple[Set[Hashable], Set[Hashable]]:
    """
        Computes graph cut using the standard Laplacian matrix with sparse computations.
        Generated by chatGPT in response to help finding numerically stable computations
    """
    L = nx.laplacian_matrix(G, nodelist=sorted(G.nodes())).astype(float)  # Sparse matrix
    v = get_fiedler_eigenvector_sparse(L)
    return get_shores(G,v)

def normalized_laplacian_graph_cut(G: nx.Graph) -> Tuple[Set[Hashable], Set[Hashable]]:
    N = nx.normalized_laplacian_matrix(G, nodelist=sorted(G.nodes())).toarray()
    v = get_fiedler_eigenvector(N)
    return get_shores_from_eigenvector(G,v)

def normalized_laplacian_graph_cut_sparse(G: nx.Graph,
                                          get_shores: function[[nx.Graph, NDArray[np.float32]], Tuple[Set[Hashable], Set[Hashable]]] = get_shores_from_eigenvector
                                          ) -> Tuple[Set[Hashable], Set[Hashable]]:
    """
        Computes graph cut using the normalized Laplacian matrix with sparse computations.
        Generated by chatGPT in response to help finding numerically stable computations
    """
    N = nx.normalized_laplacian_matrix(G, nodelist=sorted(G.nodes())).astype(float)  # Sparse matrix
    v = get_fiedler_eigenvector_sparse(N)
    return get_shores(G,v)

def randomwalk_laplacian_graph_cut(G: nx.Graph) -> Tuple[Set[Hashable], Set[Hashable]]:
    L = nx.laplacian_matrix(G, nodelist=sorted(G.nodes())).toarray()
    D = compute_degree_matrix(G)
    v = get_fiedler_eigenvector(L@np.linalg.inv(D))
    return get_shores_from_eigenvector(G,v)

def randomwalk_laplacian_graph_cut_sparse(G: nx.Graph,
                                          get_shores: function[[nx.Graph, NDArray[np.float32]], Tuple[Set[Hashable], Set[Hashable]]] = get_shores_from_eigenvector
                                          ) -> Tuple[Set[Hashable], Set[Hashable]]:
    """
        Computes graph cut using the stable random walk Laplacian (fully sparse).
        Generated by chatGPT in response to help finding numerically stable computations
    """

    # Compute Standard Laplacian (Sparse)
    L = nx.laplacian_matrix(G, nodelist=sorted(G.nodes())).astype(float)

    # Compute Degree Matrix (Sparse D⁻¹)
    degrees = np.array([G.degree(node) for node in sorted(G.nodes())], dtype=float)
    degrees[degrees == 0] = 1  # Avoid division by zero
    D_inv = diags(1.0 / degrees)  # Sparse D^(-1)

    # Compute Random Walk Laplacian: L_rw = D⁻¹ L
    L_rw = L @ D_inv   # Sparse matrix multiplication

    # Compute Fiedler Eigenvector
    v = get_fiedler_eigenvector_sparse(L_rw)

    return get_shores(G,v)

def compute_degree_matrix(G: nx.Graph) -> NDArray[np.float32]:
    """Computes the degree matrix D of a graph G."""
    degrees = np.array([G.degree(node) for node in sorted(G.nodes())], dtype=float)  # Extract node degrees
    D = np.diag(degrees)  # Create a diagonal matrix with degrees
    return D

def get_shores_from_eigenvector_median(G: nx.Graph,
                                       x: NDArray[np.float32]
                                       ) -> Tuple[Set[Hashable], Set[Hashable]]:
    """
        Partitions nodes into two sets based on the median of the Fiedler eigenvector.
        Generated by chatGPT in response to prompts about how to improve partitioning
    """
    nodes = sorted(G.nodes())  # Ensure consistent ordering
    median_value = np.median(x)  # Compute the median of the eigenvector values

    shore1 = {nodes[i] for i in range(len(nodes)) if x[i] >= median_value}
    shore2 = set(G.nodes()) - shore1

    return shore1, shore2
    