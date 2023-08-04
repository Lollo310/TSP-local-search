from typing import Any
import numpy as np
import networkx as nx

class Loader:
    
    def __init__(self):
        self.nodes = None
        self.nxGraph = None
        
        
    def readFile(self, filepath: str):
        try:
            f = open(filepath, 'r')
            nodes = []
            for line in f.readlines():
                splitted = line.strip().split()
                label = splitted[0]
                if label.isdigit():
                    node = (float(splitted[1]), float(splitted[2]))
                    nodes.append(node)
            self.nodes = np.array(nodes)
            f.close()
        except FileNotFoundError:
            print("Error: The specified file was not found.")
            
    def createNxGraph(self) -> nx.Graph:
        try:
            if self.nodes is None:
                raise ValueError
            
            G = nx.complete_graph(len(self.nodes))
            
            for i, node in enumerate(G.nodes()):
                G.nodes[node]["pos"] = self.nodes[i] 
                
            return G         
        except ValueError:        
            print('Error: Attribute nodes not be None. Read file before.')
            
