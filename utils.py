import math

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


class Loader:
    
    def __init__(self):
        self.nodes = None
        self.nxGraph = None    
        
    def readFile(self, filepath: str):
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
        try:
            if self.nodes is None:
                raise ValueError
            
            G = nx.complete_graph(len(self.nodes))
            
            for i, node in enumerate(G.nodes()):
                G.nodes[node]["pos"] = self.nodes[i] 
            
            distances = self.distEuclidean()
            
            for i in range(len(self.nodes)):
                for j in range(i+1, len(self.nodes)):
                    G[i][j]['weight'] = distances[i][j]
                
            return G         
        except ValueError:        
            print('Error: Attribute nodes not be None. Read file before.')
            
    def distEuclidean(self):
        try:
            if self.nodes is None:
                raise ValueError
            
            lenNodes = len(self.nodes)
            distances = np.zeros((lenNodes, lenNodes))
            
            for i in range(lenNodes):
                for j in range(i+1, lenNodes):
                        x1, y1 = self.nodes[i]
                        x2, y2 = self.nodes[j]
                        distances[i][j] = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                        
            return distances
        except ValueError:
            print('Error: Attribute nodes not be None. Read file before.')
            
class Utilities:
    
    def __init__(self) -> None:
        pass
        
    def draw(self, G: nx.graph): 
        _, ax = plt.subplots(figsize=(10,7))
        nx.draw(G, nx.get_node_attributes(G, 'pos'), ax, node_size=40)
        
