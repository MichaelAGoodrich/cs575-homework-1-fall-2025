""" Create a graph using the algorithm following equations 1 and 2 
from https://arxiv.org/pdf/cond-mat/0210146

Newman, Mark EJ, and Michelle Girvan. 
"Mixing patterns and community structure in networks." 
Statistical mechanics of complex networks. 
Springer, Berlin, Heidelberg, 2003. 66-87.

Implementation by Mike Goodrich
Brigham Young University
February 2022

Updated February 2024
Updated March 2025
"""

import networkx as nx
import numpy as np
from numpy.typing import NDArray
import random
from typing import List, Tuple
from copy import deepcopy


########################
## Set Default Values ##
########################
# Default mixing matrix has four classes
DEFAULT_M: NDArray[np.float32] = np.array([[0.4, 0.02, 0.01, 0.03],
                [0.02, 0.4, 0.03, 0.02],
                [0.01, 0.03, 0.4, 0.01],
                [0.03, 0.02, 0.01, 0.4]]) 
# By default, all node classes have same degree distribution
DEFAULT_LAMBDA: list[int] = [5, 4, 3, 2] 
DEFAULT_NUM_EDGES: int = 200

class Node_Type:
    def __init__(self, ID: int, degree: int, node_type: int):
        self.ID: int = ID
        self.degree: int = degree
        self.node_type: int = node_type

    def __repr__(self) -> str:
        return f"Node_Type(ID={self.ID}, degree={self.degree}, node_type={self.node_type})"

class AssortativeMixing:
    def __init__(self,
                 M: NDArray[np.float32] = DEFAULT_M, 
                 poisson_lambda:list[int] = DEFAULT_LAMBDA, 
                 num_edges:int = DEFAULT_NUM_EDGES
                 ) -> None:
        
        ## Choose how many edges of each type will be  used
        edge_count_by_type = self._draw_edges_from_mixing_matrix(M, num_edges)

        # Count the number of stubs required in each class
        stubs_per_class = self._get_number_of_stubs_per_class(edge_count_by_type, len(M))

        ## For each class
        all_nodes = self._create_nodes(stubs_per_class, poisson_lambda, seed = 42)
        
        ## Find edges to add
        all_edges = self._add_edges(all_nodes, edge_count_by_type, all_nodes)

        # Create an empty graph and add nodes and edges to it
        G: nx.Graph = nx.Graph()
        G = self._add_nodes_to_graph(G, all_nodes)
        G = self._add_edges_to_graph(G, all_edges)
        
        self.G: nx.Graph = G
            
    #############################
    ## Initialization Routines ##
    #############################
    def _draw_edges_from_mixing_matrix(self, 
                    mixing_matrix: NDArray[np.float32],
                    num_edges: int,
                    seed: int | None = None
                    ) -> dict[tuple[int,int],int]:
        """
            Choose the number of each edge type between and within classes 
            that will be in the graph using the mixing matrix

            • Input: a square mixing matrix with probabilities
                describing probability of edges within and
                across classes
            • Output: a dictionary keyed by a tuple of edge type
                with values the number of edges of that type
        """
        # Set seed if asked to
        if seed is not None:
            random.seed(seed)

        num_types: int = len(mixing_matrix)
        edge_count_by_type: dict[tuple[int, int], int] = dict()
        # Initialize dictionary of edges (i,j) 
        edge_count_by_type = {(i, j): 0 for i in range(num_types) for j in range(num_types)}
                
        # Draw edges until the desired number is reached
        count: int = 0
        edge_types: list[tuple[int, int]] = list(edge_count_by_type.keys())
        while True:
            # Shuffle edge order
            random.shuffle(edge_types)
            for edge in edge_types:
                if random.uniform(0.0, 1.0) < mixing_matrix[edge[0]][edge[1]]:
                    edge_count_by_type[edge] += 1
                    count += 1
                    if count == num_edges:
                        return edge_count_by_type

    def _get_number_of_stubs_per_class(self,
                                   edge_count_by_type: dict[tuple[int,int],int],
                                   num_classes: int
                                   ) -> dict[int, int]:
        """
            Once we know how many edges are needed for each edge type, count
            the number of stubs that are needed in each class. Do this by
            counting the number of edge end points that are in each class

            • Input: 
                - a dictionary of the number of edges indexed by edge type
                - the number of classes
            • Output: a dictionary indexed by the class number that contains
                the number of stubs required for that class
        """

        class_total_degree: dict[int, int] = dict()
        class_total_degree = {node_class: 0 for node_class in range(num_classes)}
        for edge in edge_count_by_type.keys():
            class_total_degree[edge[0]] += edge_count_by_type[edge]
            class_total_degree[edge[1]] += edge_count_by_type[edge]
        return class_total_degree

    def _get_number_nodes_in_class(self,
                                    stubs_in_class: int,
                                    average_degree_per_node: int
                                   ) -> int:
        """
            Given the number of stubs in a class, determine the number
            of nodes that will be required in that class. This is done
            by dividing the number of stubs in the class by the average
            degree per node in the class

            • Input: number of stubs per class
            • Output: number of nodes in the class
        """
        
        return int(np.ceil(stubs_in_class/average_degree_per_node))

    ##################################
    ## Routines for loop over nodes ##
    ##################################
    def _create_nodes(self, 
                      stubs_per_class: dict[int, int],
                      poisson_lambda: list[int],
                      seed: int | None = None,
                      ) -> list[Node_Type]:
        """
            Turn the stubs per class into a dictionary of nodes per type by
            sampling the nodes from a Poisson distribution. 

            Input:
                • stubs per class dictionary that specifies the number of 
                  stubs required for each class
                • lambda parameter for the Poisson distribution for each class
                • seed in case we want to control the outcome

            Output:
                • a list of allnodes
        """
        ## For each class
        starting_node_id: int = 0
        nodelist: list[Node_Type] = []
        
        for node_class in stubs_per_class.keys():
            # Get the number of nodes required for that class
            num_required_nodes = self._get_number_nodes_in_class(stubs_per_class[node_class], 
                                                                 poisson_lambda[node_class])
            
            # Create that many nodes with random degrees
            node_list = self._create_nodes_for_class(node_class, 
                                                     num_required_nodes, 
                                                     poisson_lambda[node_class], 
                                                     starting_node_id,
                                                     seed = seed)
            starting_node_id += len(node_list)
            
            # Resample node degree so that total degree = required number of stubs
            node_list = self._resample_node_degrees_in_nodelist(node_list, 
                                                                stubs_per_class[node_class],
                                                                poisson_lambda[node_class],
                                                                seed=seed)

            # Add nodes from this class to list of all nodes
            nodelist.extend(node_list)
        ## End of for loop over nodes to add
        return nodelist

    def _create_nodes_for_class(self, 
                                node_class: int,
                                num_required_nodes: int,
                                poisson_lambda: int,
                                starting_node_ID: int,
                                seed: int | None = None
                                ) -> list[Node_Type]:
        """
            Given a required number of nodes and the lambda value
            for a class, create a list of that many nodes and their
            degree, which are randomly chosen.  Nodes are labeled
            with IDs that start at the starting node ID

            • Input:
                - node class
                - required number of nodes in the class
                - the poisson parameter used to assign the degree for the node
                - the starting node ID 
            • Output: a list of nodes, which has a node ID and a node degree
                - all node degrees are greater than zero so we don't generate
                  a disconnected graph with lone nodes
        """ 
        if seed is not None:
            np.random.seed(seed)

        node_list: list[Node_Type] = []
        for node_id in range(starting_node_ID, starting_node_ID+num_required_nodes):
            node_degree: int = np.random.poisson(poisson_lambda)
            # No nodes with degree zero are allowed
            while node_degree == 0:
                node_degree = np.random.poisson(poisson_lambda)
            node_list.append(Node_Type(node_id, node_degree, node_class))
        return node_list
    
    def _resample_node_degrees_in_nodelist(self,
                                          node_list: list[Node_Type],
                                          required_number_of_stubs: int,
                                          poisson_lambda: int,
                                          seed: int | None = None
                                          ) -> list[Node_Type]:
        """
            The list of nodes that were generated in the previous step have
            arbitrary degree, and the total degree of the nodes might not
            match the number of stubs required for the class. Choose a node
            at random, resample its degree, and repeat until the total degree
            equals the required number of stubs

            Input:
                • list of nodes
                • required number of stubs in the class
                • the poisson parameter of the class
            Output:
                • a list of nodes with the degree updated, all degrees are greater than 0
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        total_degree: int = sum([node.degree for node in node_list])
        while total_degree != required_number_of_stubs:
            node: Node_Type = random.choice(node_list)
            total_degree -= node.degree
            node.degree = np.random.poisson(poisson_lambda)
            # No nodes with degree zero are allowed
            while node.degree == 0:
                node.degree = np.random.poisson(poisson_lambda)
            total_degree += node.degree

        return node_list
    
    ###############################
    ## Routines for adding edges ##
    ###############################
    def _add_edges(self,
                   all_nodes: list[Node_Type],
                   edge_count_by_type: dict[Tuple[int, int], int],
                   nodelist: list[Node_Type],
                   seed: int | None = None
                   ) -> List[Tuple[Node_Type, Node_Type]]:
        if seed is not None:
            random.seed(seed)

        # Step 1: Construct an approximate spanning tree
        while True:
            # There is a chance that the spanning tree algorithm will terminate
            # before all nodes make it into the tree just because of bad luck
            # That chance should be small, so if it happens try a new tree
            edge_count_by_type_copy = deepcopy(edge_count_by_type)
            edge_list = self._get_edges_from_spanning_tree(edge_count_by_type_copy, 
                                                    nodelist)
            if len(edge_list) == len(nodelist) - 1: #spanning trees have one fewer edge than nodes
                break
        
        # Step 2: Add remaining edges while respecting constraints
        #edge_list.extend(self.add_remaining_edges(all_nodes, edge_count_by_type_copy, edge_list))

        
        return edge_list

    def _get_edges_from_spanning_tree(self,
                                     edge_count_by_type: dict[tuple[int, int], int],
                                     node_list: list[Node_Type]
                                     ) -> list[tuple[Node_Type,Node_Type]]:

        free_stubs_in_tree, nodes_not_in_tree = self._initialize_tree(node_list)

        edge_list: list[tuple[Node_Type,Node_Type]] = []
        edge_count_by_type_copy = deepcopy(edge_count_by_type)
        while len(nodes_not_in_tree) > 0 and len(free_stubs_in_tree) > 0:
            # find nodes to join and add edge
            node1, node2 = self._pair_nodes(free_stubs_in_tree, nodes_not_in_tree, edge_count_by_type_copy)
            edge_list.append((node1, node2))

            # update counts
            free_stubs_in_tree = self._update_free_stubs_in_tree(node1, node2, free_stubs_in_tree)
            nodes_not_in_tree = self._update_nodes_not_in_tree(node2, nodes_not_in_tree)
            
            # update copy of edge counter
            edge_count_by_type_copy[(node1.node_type, node2.node_type)] -= 1

        return edge_list
    
    def _get_rootnode(welf, node_list: list[Node_Type]) -> Node_Type:
        # Choose root node randomly with the constraint that it can't have degree 1
        root_node: Node_Type = random.choice(node_list)
        #while root_node.degree < 2:
        #    root_node = random.choice(node_list)
        #root_node  = max(node_list, key=lambda node: node.degree)
        return root_node

    def _initialize_tree(self,
                        node_list: list[Node_Type]
                        ) -> tuple[list[Node_Type], set[Node_Type]]:
        free_stubs_in_tree: list[Node_Type] = []
        nodes_not_in_tree: set[Node_Type] = set(node_list)
        root_node = self._get_rootnode(node_list)
        # Add copies into tree, one copy for each stub
        free_stubs_in_tree.extend([root_node for _ in range(root_node.degree)]) 
        nodes_not_in_tree.remove(root_node)
        return free_stubs_in_tree, nodes_not_in_tree

    def _pair_nodes(self,
                   free_stubs_in_tree: list[Node_Type],
                   nodes_not_in_tree: set[Node_Type],
                   edge_count_by_type: dict[tuple[int,int], int]
                   ) -> tuple[Node_Type, Node_Type]:   
        node1: Node_Type = random.choice(free_stubs_in_tree)
        node2: Node_Type = random.choice(list(nodes_not_in_tree))
        attempts: int = 0
        while edge_count_by_type[(node1.node_type, node2.node_type)] == 0:
            node1 = random.choice(free_stubs_in_tree)
            node2 = random.choice(list(nodes_not_in_tree))
            attempts += 1
            if attempts == 1000:
                print("Failed to find a permissible edge.")
                print(f"Adding impermissible edge ({node1.ID, node2.ID})")
                break
        return node1, node2

    def _update_free_stubs_in_tree(self,
                                  node1: Node_Type,
                                  node2: Node_Type,
                                  free_stubs_in_tree: list[Node_Type]
                                  ) -> list[Node_Type]:
        # Add copies into tree, one copy for each stub
        free_stubs_in_tree.extend([node2 for _ in range(node2.degree-1)]) 
        free_stubs_in_tree.remove(node1)
        return free_stubs_in_tree

    def _update_nodes_not_in_tree(self,
                                 node: Node_Type,
                                 nodes_not_in_tree: set[Node_Type]
                                 ) -> set[Node_Type]:
        nodes_not_in_tree.remove(node)
        return nodes_not_in_tree
    
        ###############################
    ## Add Remaining Edges        ##
    ###############################
    def add_remaining_edges(self, all_nodes: list[Node_Type], edge_count_by_type: dict[tuple[int, int], int], current_edges: list[tuple[Node_Type, Node_Type]]):
        """
        Adds additional edges while ensuring edge count constraints and node degrees are met.
        """
        free_stubs = self.compute_free_stubs(all_nodes, current_edges)
        additional_edges = []

        while len(free_stubs) > 1:
            node1, node2 = self.select_valid_edge(free_stubs, edge_count_by_type)
            additional_edges.append((node1, node2))

            free_stubs.remove(node1)
            free_stubs.remove(node2)

            edge_count_by_type[(node1.node_type, node2.node_type)] -= 1

        return additional_edges

    def compute_free_stubs(self, all_nodes: list[Node_Type], current_edges: list[tuple[Node_Type, Node_Type]]):
        """
        Computes nodes that still need edges.
        """
        stub_counts = {node: node.degree for node in all_nodes}
        for edge in current_edges:
            stub_counts[edge[0]] -= 1
            stub_counts[edge[1]] -= 1
        return [node for node, stubs in stub_counts.items() if stubs > 0]

    def select_valid_edge(self, free_stubs: list[Node_Type], edge_count_by_type: dict[tuple[int, int], int]):
        """
        Selects a valid edge between free stubs while respecting edge count constraints.
        """
        while True:
            node1, node2 = random.sample(free_stubs, 2)
            if edge_count_by_type.get((node1.node_type, node2.node_type), 0) > 0:
                return node1, node2


            
    ########################
    ## Graph Construction ##
    ########################
    def _add_nodes_to_graph(self,
                             G: nx.Graph,
                             node_list: list[Node_Type]
                             ) -> nx.Graph:
        """ 
            Take an empty graph and add the nodes from the nodelist to it.
            Label each node with its class and its degree

            • Input:
                - empty graph nx.Graph
                - node list

            • Output: graph with nodes added
        """
        for node in node_list:
            G.add_node(node.ID, node_class = node.node_type)
        
        return G
    
    def _add_edges_to_graph(self,
                             G: nx.Graph,
                             edge_list: list[Tuple[Node_Type,Node_Type]]
                             ) -> nx.Graph:
        """ 
            Take graph with node edges and add the edges from the.
            Label each node with its class and its degree

            • Input:
                - graph nx.Graph with no edges
                - edge list

            • Output: graph with edges added
        """
        for node1, node2 in edge_list:
            G.add_edge(node1.ID, node2.ID)
        
        return G