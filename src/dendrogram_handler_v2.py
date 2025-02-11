## Dendrogram
## 
## Mike Goodrich
## Brigham Young University
## ## February 2022 -- first implementation
## December 2024 -- reimplemented with type hints
## 
## Code adapted from
## https://stackoverflow.com/questions/59821151/plot-the-dendrogram-of-communities-found-by-networkx-girvan-newman-algorithm 
## Thank you Giora Simchoni

import networkx as nx # type: ignore
from itertools import chain, combinations
import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Union, Hashable, Set, Any # Used for type hints

Group = Set[Hashable]

class DendrogramHandler:
    def __init__(self,G: nx.Graph):
        # Find all partitions using Girvan Newman greedy algorithm on edge betweenness
        all_partitions: list[Tuple[Group, ...]] = self._get_girvan_newman_communities(G)
        # Create a tree node for each group that can be formed by the algorithm
        treenode_group_dict: dict[int, Group] = self._assign_groups_to_tree_nodes(all_partitions)
        # Organize the tree nodes into a binary tree depending on how the partitions are nested
        binary_tree: dict[int, list[int]] = self._get_binary_tree(treenode_group_dict)
        # Create labels for each tree node id depending on what group it represents
        node_labels: dict[int, str] = self._get_node_labels(treenode_group_dict)
        # Get the link matrix
        #self.link_matrix, leaves = self._getLinkMatrix(binary_tree, all_partitions, node_labels)
        # I don't know what this does
        #self.link_matrix_labels: list[int | str] = [node_labels[node_id] for node_id in leaves]
    
    def _get_girvan_newman_communities(self, 
                                       G: nx.Graph
                                       ) -> list[Tuple[Group, ...]]:
        """ Get communities using edge betweenness algorithm from Girvan and Newman
        
        First, some terminology.
          • A group is a list of agents that belong together according to some criterion
          • A partition is a set of groups that (a) don't overlap and (b) cover the set of all nodes
          • This function returns list of all possible partitions
        
        The networkx implementation of the girvan_newman returns a generator, 
        which I turn into a list. This list contains partitions, which are
        represented as tuples. Thus, each tuple is a partition, and the elements
        of the partitions are groups. 
        
        The groups are represented as sets of graph nodes, where all
        the nodes in a single set belong to the same group. 
        
        The Girvan-Newman algorithm can return partitions with two groups, three groups, 
        groups communities, etc., depending on when you terminate the algorithm. The
        function below returns all possible partitions, so you'll have 
        one partition tuple with two sets (two groups), another tuple with three
        sets (three groups), and so on until you have a tuple with
        a set of groups with only one agent in each group --> one singleton for each node in the graph.
        """
        return list(nx.algorithms.community.centrality.girvan_newman(G))
    
    def _assign_groups_to_tree_nodes(self, 
                      all_partitions: list[Tuple[Group, ...]]
                      ) -> dict[int, Group]:
        ###################################
        ## Step 1: Initialize Dictionary ##
        ###################################
        # The node_id = 0 group is just the list of all nodes, which we obtain by unioning
        # the two groups in the first partition.
        node_id: int = 0
        group_labels: dict[int, Group] = {node_id: all_partitions[0][0].union(all_partitions[0][1])}

        ###################################################
        ## Step 2: Assign all other groups to tree nodes ##
        ###################################################
        # Create a dictionary where each unique group is assigned to a tree node
        # The tree nodes are numbered in the order in which they are found, so
        # node_id = 1 and node_id = 2 are the two groups that belong to the
        # first community, node_id = 3 and 4 are the groups appear when
        # one of the previous groups is split, and so on
        
        for partition in all_partitions:                # Iterate through each tuple of groups
            for group in list(partition):               # Iterate through each group in the tuple
                if group not in group_labels.values():  # If I haven't seen that subset before
                    node_id += 1
                    group_labels[node_id] = group      # Add that group to the dictionary
        return group_labels

    def _get_node_labels(self,
                        treenode_group_dict: dict[int, Group]
                        ) -> dict[int, str]:
        node_labels: dict[int, str] = dict()
        for node in treenode_group_dict.keys():
            if len(treenode_group_dict[node]) == 1:
                node_labels[node] = str(list(treenode_group_dict[node])[0])
            else:
                node_labels[node] = ""
        return node_labels

    def _get_binary_tree(self,
                         treenode_group_dict: dict[int, Group]
                         ) -> dict[int, list[int]]:
        """ Create binary tree that shows which tree nodes of which other tree nodes.
        Each tree node represents one of the subgroups found in the communities.
        The root node is set of all agents.
        The leaf nodes are sets of individual agents.
        Return a binary tree of tree nodes
        """

        # Initialize the dictionary containing the {node: empty children list} in the tree
        binary_tree: dict[int, list[int]] = {tree_node: [] for tree_node in treenode_group_dict.keys()} 
        
        # For each possible way of partitioning a group into two subgroups
        for node_1, node_2 in combinations(treenode_group_dict.keys(), 2): 
             # For each tree node and existing group
            for parent_node, group in treenode_group_dict.items():
                # If the subgroups form a partition, that is the two enumerated groups don't intersect AND
                # If the group is made up from the two enumerated subgroups
                if len(treenode_group_dict[node_1].intersection(treenode_group_dict[node_2])) == 0 and \
                        group == treenode_group_dict[node_1].union(treenode_group_dict[node_2]):
                    # The left branch of the tree is the first subgroup in the partition
                    binary_tree[parent_node].append(node_1) 
                    # The right branch of the tree is the second subgroup in the partiion
                    binary_tree[parent_node].append(node_2) 
        
        return binary_tree

    def _get_linkage_matrix(self, binary_tree: dict[int, list[int]]) -> NDArray[np.float64]:
        """
        Create a linkage matrix from the binary tree representation.
        
        Parameters:
        - binary_tree: dict mapping a parent node to its two child nodes.

        Returns:
        - linkage_matrix: A NumPy array that can be used in scipy's dendrogram function.
        """
        
        G: nx.Graph = nx.DiGraph(binary_tree)
        nodes: list[int] = list(G.nodes)
        leaves: set[int] = {n for n in nodes if G.out_degree(n) == 0}
        num_leaves = len(leaves)

        linkage_matrix = []
        node_to_index = {}  # Map internal nodes to linkage matrix indices
        current_index = num_leaves  # Start numbering internal nodes after the leaf indices
        
        for parent, children in binary_tree.items():
            if not isinstance(parent, int):  # Debugging check
                raise TypeError(f"Expected parent to be an int, got {type(parent)}: {parent}")

            if len(children) != 2:
                raise ValueError(f"Each parent should have exactly two children, got: {children}")

            child1, child2 = children  # Unpack child nodes

            # Get indices for children (if it's an internal node, use the mapping)
            index1 = node_to_index.get(child1, child1)  # If not in mapping, it's a leaf
            index2 = node_to_index.get(child2, child2)

            # Ensure valid indices before accessing linkage_matrix
            count1 = 1 if index1 < num_leaves else linkage_matrix[index1 - num_leaves][3]
            count2 = 1 if index2 < num_leaves else linkage_matrix[index2 - num_leaves][3]
            count = count1 + count2

            # Compute distance using depth level or another metric
            distance = max(index1, index2)  # You might replace this with another metric

            # Append to linkage matrix
            linkage_matrix.append([index1, index2, distance, count])

            # Map parent to new index
            node_to_index[parent] = current_index
            current_index += 1

        return np.array(linkage_matrix, dtype=np.float64)


    def _get_subset_rank_dict(self, 
                              all_partitions: list[Tuple[Group, ...]]
                              ) -> dict[Tuple[Any, ...], int]:
        # also needing a subset to rank dict to later know within all k-length merges which came first
        subset_rank_dict: dict[Tuple[Any, ...], int] = dict()
        rank: int = 0
        for e in all_partitions[::-1]:
            for p in list(e):
                if tuple(p) not in subset_rank_dict:
                    subset_rank_dict[tuple(sorted(p))] = rank
                    rank += 1
        subset_rank_dict[tuple(sorted(chain.from_iterable(all_partitions[-1])))] = rank
        return subset_rank_dict 
    
    def _get_leaves_of_subtree(self, 
                               inner_nodes: list[int], 
                               leaves: set[int],
                               node_id_to_children: dict[int, list[int]]
                               ) -> dict[int, list[int]]:

        # Compute the size of each subtree
        subtree: dict[int, list[int]] = dict( (n, [n]) for n in leaves ) #TODO: Change the type of the key so you don't have to use union
        for u in inner_nodes:
            children = set()
            node_list = list(node_id_to_children[u])
            while len(node_list) > 0:  # This is a depth-first search traversal of the subtree rooted at u. Nodelist acts as the stack
                v = node_list.pop(0)
                children.add( v )
                node_list += node_id_to_children[v] # += appends one list to another
            subtree[u] = sorted(children & leaves) # & means intersection, so find all the descendants of node and select the leaves
        return subtree
    
    def _getLinkMatrix(self,
                       binary_tree: dict[int, list[int]],
                       all_partitions: list[Tuple[Group, ...]],
                       node_labels: dict[int, str]
                       ) -> Tuple[list[list[float]], set[int]]:
        # finally using @mdml's magic, slightly modified:
        G: nx.Graph            = nx.DiGraph(binary_tree)
        nodes: list[int]       = G.nodes()
        leaves: set[int]       = set( n for n in nodes if G.out_degree(n) == 0 )
        inner_nodes: list[int] = [ n for n in nodes if G.out_degree(n) > 0 ]

        # Get the leaves that descend from each subtree 
        subtree: dict[Any, list[Any]] = self._get_leaves_of_subtree(inner_nodes, leaves, binary_tree)
        #subtree: dict[Union[int, Tuple[int, ...]], list[int]] = dict((i,j) for i,j in returned_subtree.items())
        #subtree: dict[Any, list[Any]] = dict((i,j) for i,j in returned_subtree.items())
        
        # Order inner nodes ascending by subtree size, root is last
        inner_nodes.sort(key=lambda n: len(subtree[n])) # <-- order inner nodes ascending by subtree size, root is last
        # TODO: sort by first sets split instead

        # Construct the linkage matrix
        sorted_leaves: list[int] = sorted(leaves)
        index  = dict( (tuple([n]), i) for i, n in enumerate(sorted_leaves) )
        Z: list[list[float]]= []
        k: int = len(sorted_leaves)
        for i, n in enumerate(inner_nodes):
            children = binary_tree[n]
            x: Union[int, Tuple[int, ...]] = children[0]
            for y in children[1:]:
                z = tuple(sorted(subtree[x] + subtree[y]))
                i, j = index[tuple(sorted(subtree[x]))], index[tuple(sorted(subtree[y]))]
                Z.append([i, j, self._get_merge_height(subtree[n], all_partitions, node_labels), len(z)]) # <-- float is required by the dendrogram function
                index[z] = k
                subtree[z] = list(z)
                x = z
                k += 1
        return Z, leaves
    
    def _get_merge_height(self,
                          sub: list[int],
                          all_partitions: list[Tuple[Group, ...]],
                          node_labels: dict[int, str]
                          ) -> float:
        subset_rank_dict = self._get_subset_rank_dict(all_partitions)
        # Giora Simchoni's function to get a merge height so that it is unique (probably not that efficient)
        sub_tuple = tuple(sorted([node_labels[i] for i in sub]))
        #sub_tuple = tuple([int(v) for v in sub_tuple_str])
        n = len(sub_tuple)
        other_same_len_merges = {k: v for k, v in subset_rank_dict.items() if len(k) == n}
        min_rank, max_rank = min(other_same_len_merges.values()), max(other_same_len_merges.values())
        range = (max_rank-min_rank) if max_rank > min_rank else 1
        return float(len(sub)) + 0.8 * float((subset_rank_dict[sub_tuple] - min_rank) / range)
