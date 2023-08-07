import networkx as nx
import sys

class TSP:
    
    def __init__(self) -> None:
        self.hamiltonian_cycle = None
        self.hamiltonian_cost = 0
        
    def NN(self, G: nx.graph):
        weights = nx.get_edge_attributes(G, 'weight')
        visited_nodes = {0}
        self.hamiltonian_cycle = []
        node = 0
        
        while nx.number_of_nodes(G) != len(visited_nodes): 
            last_node = None
            minimum = float('inf')
        
            for i in range(nx.number_of_nodes(G)): 
                if i not in visited_nodes and i != node:
                    x = min(node, i)
                    y = max(node, i)
                    weight = weights[(x,y)]
                    if weight < minimum:
                        minimum = weight
                        last_node = i
            
            self.hamiltonian_cycle.append((node, last_node))
            self.hamiltonian_cost += minimum
            visited_nodes.add(last_node)
            node = last_node
        
        self.hamiltonian_cycle.append((0, node))          
        
          
