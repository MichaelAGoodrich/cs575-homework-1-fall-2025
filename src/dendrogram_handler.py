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
from typing import Tuple, Union # Used for type hints

class DendrogramHandler:
    def __init__(self,G: nx.Graph):
        communities = self._get_girvan_newman_communities(G)
        node_id_to_children, self.node_labels = self._get_treenode_id_to_children_dict(communities)
        self.link_matrix, leaves = self._getLinkMatrix(node_id_to_children, communities)
        self.link_matrix_labels: list[int | str] = [self.node_labels[node_id] for node_id in leaves]
    
    """ Private methods """
    def _get_girvan_newman_communities(self, G: nx.Graph) -> list[Tuple[set[int], ...]]:
        """ Get communities using edge betweenness algorithm from Girvan and Newman
        
        First, some terminology.
          • A group is a list of agents that belong together according to some criterion
          • A community is a collection of groups. The union of all the agents in the groups covers all agents
        
        The networkx implementation of the girvan_newman returns a generator, 
        which I turn into a list. This list contains communities, which are
        represented as tuples. Thus, each tuple is a community, and the elements
        of the community are groups. 
        
        The groups are represented as sets, where all
        the nodes in a single set belong to the same group. 
        
        The Girvan-Newman algorithm can return two communities, three communities, 
        four communitie ... depending on when you terminate the algorithm. The
        function below returns all possible communities, so you'll have 
        one community tuple with two sets (two groups), another tuple with three
        sets (three groups), and so on until you have a tuple with
        a set of groups with only one agent in each group --> one singleton for each node in the graph.
        
        Networkx allows node types to be any hashable data type, but this code
        is only implemented in a way that allows integers. Thus, the members
        of each community are integers with type hint set[int]
        """
        communities = list(nx.algorithms.community.centrality.girvan_newman(G))
        return communities
    
    def _label_groups(self, communities: list[Tuple[set[int], ...]]) -> dict[int, set[int]]:
        ###################################
        ## Step 1: Initialize Dictionary ##
        ###################################
        # The node_id = 0 community is just the list of all nodes, which we obtain by unioning
        # the two sets in the first community.
        node_id: int = 0
        graphnode_labels_dict: dict[int, set[int]] = {node_id: communities[0][0].union(communities[0][1])}

        #######################################
        ## Step 2: Label Each Possible Group ##
        #######################################
        # Create a dictionary where each unique group is given a unique label
        # The groups are labeled in the order in which they are found, so
        # node_id = 1 and node_id = 2 are the two groups that belong to the
        # first community, node_id = 3 and 4 are the groups appear when
        # one of the previous groups is split, and so on
        # Label each group in the order in which they split from a larger group
        for comm in communities:                                 # Iterate through each tuple of groups
            for subset in list(comm):                            # Iterate through each group in the tuple
                if subset not in graphnode_labels_dict.values(): # If I haven't seen that subset before
                    node_id += 1                                 # Make up a label
                    graphnode_labels_dict[node_id] = subset      # Add that subset to the dictionary
        return graphnode_labels_dict

    def _get_treenode_id_to_children_dict(self, 
                                          communities: list[Tuple[set[int], ...]]) \
                                            -> Tuple[dict[int, list[int]], dict[int, int|str]]:
        """ Create binary tree that shows which tree nodes of which other tree nodes.
        Each tree node represents one of the subgroups found in the communities.
        The root node is set of all agents.
        The leaf nodes are sets of individual agents.
        Return a binary tree and the labels for each node
        """

        group_labels_dict: dict[int, set[int]] = self._label_groups(communities)

        ##############################################
        ## Step 1: Organize groups into binary tree ##
        ##############################################
        # Initialize the dictionary containing the {node: children} in the tree
        node_id_to_children: dict[int, list[int]] = {group_id: [] for group_id in group_labels_dict.keys()} 
        # Each node has a group
        # The children of each node are the subgroups of the group that partition the group
        for node_id1, node_id2 in combinations(group_labels_dict.keys(), 2): 
            # For each possible way of partitioning a group into two subgroups
            for node_id_parent, group in group_labels_dict.items():
                # For each tree node and existing group
                if len(group_labels_dict[node_id1].intersection(group_labels_dict[node_id2])) == 0 and \
                        group == group_labels_dict[node_id1].union(group_labels_dict[node_id2]):
                    # If the subgroups form a partition, that is the two enumerated groups don't intersect AND
                                # If the group is made up from the two enumerated subgroups
                    # The left branch of the tree is the first subgroup in the partition
                    node_id_to_children[node_id_parent].append(node_id1) 
                    # The right branch of the tree is the second subgroup in the partiion
                    node_id_to_children[node_id_parent].append(node_id2) 
        
        ########################################################
        ## Step 2: Associate Each Tree node with a Node Label ##
        ########################################################
        # Make the labels of the interior tree nodes blank 
        # Set the labels of the leaf nodes to the agent (i.e., graph node) number
        node_labels: dict[int, int | str] = dict()
        for node_id, group in group_labels_dict.items():
            if len(group) == 1:
                node_labels[node_id] = list(group)[0]
            else:
                node_labels[node_id] = ''
        return (node_id_to_children, node_labels)

    def _get_subset_rank_dict(self, communities: list[Tuple[set[int], ...]]) -> dict[Tuple[int, ...], int]:
        # also needing a subset to rank dict to later know within all k-length merges which came first
        subset_rank_dict: dict[Tuple[int, ...], int] = dict()
        rank: int = 0
        for e in communities[::-1]:
            for p in list(e):
                if tuple(p) not in subset_rank_dict:
                    subset_rank_dict[tuple(sorted(p))] = rank
                    rank += 1
        subset_rank_dict[tuple(sorted(chain.from_iterable(communities[-1])))] = rank
        return subset_rank_dict 
    
    def _get_leaves_of_subtree(self, 
                               inner_nodes: list[int], 
                               leaves: set[int],
                               node_id_to_children: dict[int, list[int]]) \
                                  -> dict[int, list[int]]:

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
                       node_id_to_children: dict[int, list[int]],
                       communities: list[Tuple[set[int], ...]]) \
                          -> Tuple[list[list[float]], set[int]]:
        # finally using @mdml's magic, slightly modified:
        G: nx.Graph            = nx.DiGraph(node_id_to_children)
        nodes: list[int]       = G.nodes()
        leaves: set[int]       = set( n for n in nodes if G.out_degree(n) == 0 )
        inner_nodes: list[int] = [ n for n in nodes if G.out_degree(n) > 0 ]

        # Get the leaves that descend from each subtree 
        returned_subtree: dict[int, list[int]] = self._get_leaves_of_subtree(inner_nodes, leaves, node_id_to_children)
        subtree: dict[Union[int, Tuple[int, ...]], list[int]] = dict((i,j) for i,j in returned_subtree.items())
        
        # Order inner nodes ascending by subtree size, root is last
        inner_nodes.sort(key=lambda n: len(subtree[n])) # <-- order inner nodes ascending by subtree size, root is last

        # Construct the linkage matrix
        sorted_leaves: list[int] = sorted(leaves)
        index  = dict( (tuple([n]), i) for i, n in enumerate(sorted_leaves) )
        Z: list[list[float]]= []
        k: int = len(sorted_leaves)
        for i, n in enumerate(inner_nodes):
            children = node_id_to_children[n]
            x: Union[int, Tuple[int, ...]] = children[0]
            for y in children[1:]:
                z = tuple(sorted(subtree[x] + subtree[y]))
                i, j = index[tuple(sorted(subtree[x]))], index[tuple(sorted(subtree[y]))]
                Z.append([i, j, self._get_merge_height(subtree[n], communities), len(z)]) # <-- float is required by the dendrogram function
                index[z] = k
                subtree[z] = list(z)
                x = z
                k += 1
        return Z, leaves
    
    def _get_merge_height(self,
                          sub: list[int],
                          communities: list[Tuple[set[int], ...]]
                          ) \
                            -> float:
        subset_rank_dict = self._get_subset_rank_dict(communities)
        # Giora Simchoni's function to get a merge height so that it is unique (probably not that efficient)
        sub_tuple = tuple(sorted([self.node_labels[i] for i in sub]))
        n = len(sub_tuple)
        other_same_len_merges = {k: v for k, v in subset_rank_dict.items() if len(k) == n}
        min_rank, max_rank = min(other_same_len_merges.values()), max(other_same_len_merges.values())
        range = (max_rank-min_rank) if max_rank > min_rank else 1
        return float(len(sub)) + 0.8 * float((subset_rank_dict[sub_tuple] - min_rank) / range)
