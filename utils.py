import math
import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


class Loader:
    """A class for loading node coordinates from a file and creating a NetworkX graph.

    Attributes:
        nodes (np.ndarray): An array of node coordinates.
        nxGraph (None or nx.Graph): The NetworkX graph object.

    Methods:
        readFile(filepath: str): Read and process the node coordinates from a file.
        createNxGraph() -> nx.Graph: Create a NetworkX graph from the loaded node coordinates.
        distEuclidean(): Calculate the Euclidean distances between nodes.
    """

    def __init__(self):
        self.nodes = None
        self.nxGraph = None

    def readFile(self, filepath: str) -> None:
        """Read and process the node coordinates from a file.

        Args:
            filepath (str): The path to the input file.
        """
        try:
            with open(filepath, 'r') as f:
                nodes = []

                for line in f.readlines():
                    splitted = line.strip().split()
                    label = splitted[0]

                    if label.isdigit():
                        node = (float(splitted[1]), float(splitted[2]))
                        nodes.append(node)

                self.nodes = np.array(nodes)

        except FileNotFoundError:
            print("Error: The specified file was not found.")

    def createNxGraph(self) -> nx.graph:
        """Create a NetworkX graph from the loaded node coordinates.

        Raises:
            TypeError: If the nodes attribute is None.

        Returns:
            nx.Graph: The constructed NetworkX graph.
        """
        try:
            if self.nodes is None:
                raise TypeError

            G = nx.complete_graph(len(self.nodes))

            for i, node in enumerate(G.nodes()):
                G.nodes[node]["pos"] = self.nodes[i]

            distances = self.distEuclidean()

            for i in range(len(self.nodes)):
                for j in range(i+1, len(self.nodes)):
                    G[i][j]['weight'] = distances[i][j]

            return G
        except TypeError:
            print('Error: Attribute nodes can not be None. Read file before.')

    def distEuclidean(self) -> np.ndarray:
        """Calculate the Euclidean distances between nodes.

        Raises:
            TypeError: If the nodes attribute is None.

        Returns:
            np.ndarray: A matrix of Euclidean distances between nodes.
        """
        try:
            if self.nodes is None:
                raise TypeError

            lenNodes = len(self.nodes)
            distances = np.zeros((lenNodes, lenNodes))

            for i in range(lenNodes):
                for j in range(i+1, lenNodes):
                    x1, y1 = self.nodes[i]
                    x2, y2 = self.nodes[j]
                    distances[i][j] = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

            return distances
        except TypeError:
            print('Error: Attribute nodes can not be None. Read file before.')


class Utilities:
    """A class providing utility functions.

    Methods:
        draw(G: nx.Graph): Draw a NetworkX graph.
    """

    def __init__(self) -> None:
        pass

    def draw(self, G: nx.graph) -> None:
        """Draw a NetworkX graph.

        Args:
            G (nx.Graph): The graph to be drawn.
        """
        _, ax = plt.subplots(figsize=(10, 7))
        nx.draw(G, nx.get_node_attributes(G, 'pos'), ax, node_size=40)
        
    def getTspFiles(self, directory: str) -> list:
        """Get a list of .tsp files in the specified directory.
        
        Args:
            directory (str): The path of the directory to search for .tsp files.
            
        Returns:
            list: A list of filenames ending with '.tsp' in the directory.
        """
        tsp_files = []
        
        for filename in os.listdir(directory):
            if filename.endswith('.tsp'):
                tsp_files.append(filename)
                
        return tsp_files