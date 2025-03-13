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
"""

import networkx as nx
import numpy as np
from numpy.typing import NDArray

# Default mixing matrix has four classes
DEFAULT_M: NDArray[np.float32]= np.array([[0.4, 0.02, 0.01, 0.03],
                [0.02, 0.4, 0.03, 0.02],
                [0.01, 0.03, 0.4, 0.01],
                [0.03, 0.02, 0.01, 0.4]]) 
# By default, all node classes have same degree distribution
DEFAULT_LAMBDA: list[int] = [5, 5, 5, 5] 
DEFAULT_NUM_EDGES: int = 200

class MixedNetworkFormation:
    def __init__(self,
                 M: NDArray[np.float32] = DEFAULT_M, 
                 poisson_lambda:list[int] = DEFAULT_LAMBDA, 
                 num_edges:int = DEFAULT_NUM_EDGES):
        ### Initialize Algorithm with four types using Equation 3 ###
        self.M:  NDArray[np.float32] = M

        # Number of node types equals dimension of M matrix
        self.type_list = [i for i in range(len(M))]
        self.color_template = ['y', 'b', 'm', 'c', 'k'] 

        # Create empty graph
        self.G = nx.Graph()
        
        # Run the algorithm
        self.__AlgorithmStep1__(poisson_lambda)
        self.__AlgorithmStep2__(num_edges)
        self.__AlgorithmStep3__()
        self.__AlgorithmStep4__()

        ### Set graph properties ### 
        self.color_map = self._getColorMap()        

    """ Public methods """
    def getGraph(self) -> nx.Graph: 
        return self.G
    def getGroundTruthColors(self) -> list[str]: 
        return self.color_map   
    def getColorTemplate(self) -> list[str]: 
        return self.color_template
    def getAverageDegree(self) -> float: 
        return float(np.average([self.G.degree(n) for n in self.G.nodes]))
    """ Private methods """
    ### Graph Helpers 
    def _getColorMap(self) -> list[str]:
        color_map = []
        num_colors = len(self.color_template)
        for node in self.G.nodes: 
            color_map.append(self.color_template[self.G.nodes[node]['type']%num_colors])
        return color_map

    ### Algorithm Step 1 ###
    def __AlgorithmStep1__(self, poisson_lambda: list[int]) -> None:
        self.PoissonLambda: list[int] = poisson_lambda
   
    ### Algorithm Step 2 ###
    def __AlgorithmStep2__(self, num_edges: int) -> None:
        self.num_edges = num_edges # This is how many edges I want
        self.edgeNumbersDict = self._drawEdgesFromMixingMatrix()
        self.endsByTypeDict = self._countEndsOfEdgesByType()
        self.expectedNumberOfNodes = self._computeExpectedNumberOfNodes()
    
    def _drawEdgesFromMixingMatrix(self) -> dict[tuple[int, int], int]:
        edgeNumbersDict: dict[tuple[int,int], int] = dict()
        # initialize dictionary
        for type1 in self.type_list:
            for type2 in self.type_list:
                if type2>=type1:
                    edgeNumbersDict[(type1,type2)] = 0
        count = 0
        while count < self.num_edges:
            for type1 in self.type_list:
                for type2 in self.type_list:
                    if np.random.uniform(low=0.0,high=1.0) < self.M[type1][type2]:
                        if type2 >= type1: 
                            edgeNumbersDict[(type1,type2)] += 1
                        else: 
                            edgeNumbersDict[(type2,type1)] += 1
                        count+=1
        #(edgeNumbersDict)
        #print("There are ", sum([edgeNumbersDict[key] for key in edgeNumbersDict.keys()])," edges in the dictionary")
        return edgeNumbersDict
    
    def _countEndsOfEdgesByType(self) -> dict[int, int]:
        endsByTypeDict = {k: 0 for k in self.type_list} # initialize dictionary
        for type1 in self.type_list:
            for type2 in self.type_list:
                if type2<type1: continue
                if type1 == type2: endsByTypeDict[type1] += self.edgeNumbersDict[(type1,type2)]*2
                else: 
                    endsByTypeDict[type1] += self.edgeNumbersDict[(type1,type2)]
                    endsByTypeDict[type2] += self.edgeNumbersDict[(type1,type2)]
        #print("node ends by type ", endsByTypeDict)
        return endsByTypeDict
    
    def _computeExpectedNumberOfNodes(self) -> dict[int, int]:
        numNodeDict = dict()
        for type in self.type_list:
            n = np.round(self.endsByTypeDict[type]/self.PoissonLambda[type])
            #print(int(n))
            numNodeDict[type] = int(n)
        return numNodeDict

    ### Algorithm Step 3 ###
    def __AlgorithmStep3__(self) -> None:
        self.nodeListByTypeAndDegree = self._drawNodesFromTypes()

    def _drawNodesFromTypes(self) -> dict[int, list[int]]:
        nodesByDegreeDict: dict[int, list[int]] = {k: [] for k in self.type_list} # initialize to dictionary of empty lists
        for type in self.type_list:
            nodeList = self._drawNodesByType(type)
            nodesByDegreeDict[type] = nodeList
        #print("nodesByDegreeDict = ", nodesByDegreeDict) 
        return nodesByDegreeDict  
    
    def _drawNodesByType(self,type: int) -> list[int]:
        nodeList: list[int] = list()
        #print("Trying to get ", self.endsByTypeDict[type]," total degrees")
        while len(nodeList) != self.expectedNumberOfNodes[type]:
            if len(nodeList) < self.expectedNumberOfNodes[type]:
                node = np.random.poisson(lam=self.PoissonLambda)
                if node==0: continue
                nodeList.append(node)
            #print("Number of nodes in node set ", type, " is ", sum(nodeList))
            if len(nodeList) > self.expectedNumberOfNodes[type]:
                nodeList.pop(0)
                #print("Number of nodes in node set ", type, " is ", sum(nodeList))
        #print("Length of nodeList for type ",type, " = ",len(nodeList))
        return nodeList

    ### Algorithm Step 4 ###
    def __AlgorithmStep4__(self) -> None:
        self._addNodesToGraph()
        self._addEdgesToGraph() # Requires nodes to be added to graph
    def _addNodesToGraph(self) -> None:
        # Make the node list into a a format with node_id by type
        nodeList = []
        nodeID = 0
        for type in self.type_list:
            for nodeDegree in self.nodeListByTypeAndDegree[type]:
                nodeList.append((nodeID, {"type":type, "degree":nodeDegree}))
                nodeID += 1
        #print("node list is ", nodeList)
        self.G.add_nodes_from(nodeList)

    def _addEdgesToGraph(self) -> None:
        #print("Adding edges to graph")
        for edge_type in self.edgeNumbersDict.keys():
            type1 = edge_type[0]; type2 = edge_type[1]
            while self.edgeNumbersDict[edge_type] > 0 :
                free_agents1 = self._getFreeAgents(type1)
                free_agents2 = self._getFreeAgents(type2)
                if free_agents1 == [] or free_agents2 == []: break
                node1 = free_agents1[np.random.randint(0,len(free_agents1))]
                neighbors_of_node = [n for n in self.G[node1]]
                neighbors_of_node.append(node1) # If same type, don't allow self loops
                possible_neighbors = list(set(free_agents2) - set(neighbors_of_node))
                if possible_neighbors == []: break
                index2 = np.random.randint(0,high=len(possible_neighbors))
                node2 = possible_neighbors[index2]
                self.G.add_edge(node1,node2)
                self.edgeNumbersDict[edge_type] -= 1
                self._decrementRemainingDegree(node1)
                self._decrementRemainingDegree(node2)
        #print("Done adding edges to graph")
        return 
    def _getFreeAgents(self,type: int) -> list[int]:
        # return agents of specified type that have free stubs
        #print("Graph node info is ", self.G.nodes.data())
        nodes: list[int] = []
        for node in self.G.nodes.data():
            node_info = list(node)
            node_index = node_info[0]
            node_degree = node_info[1]['degree']
            node_type = node_info[1]['type']
            if node_type == type and node_degree > 0: 
                nodes.append(node_index)
        return nodes
    def _decrementRemainingDegree(self,node_index: int) -> None:
        # subtract one from the nodeList
        self.G.nodes[node_index]['degree'] -= 1
        #print("node ", node_index, " now has remaining degree ", self.G.nodes[node_index]['degree'], "\n\n")
        return