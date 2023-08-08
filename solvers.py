import networkx as nx
import time

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
        
    def toNetworkX(self, G: nx.graph) -> nx.graph:
        try:
            if self.hamiltonian_cycle is None:
                raise ValueError
            
            H = nx.Graph(self.hamiltonian_cycle)
            nx.set_node_attributes(H, nx.get_node_attributes(G, 'pos'), 'pos')          
            nx.set_edge_attributes(H, nx.get_edge_attributes(G, 'weight'), 'weight')
            
            return H
        except ValueError:
            print('Error: Attribute hamiltonian_cycle not be None. Read file before.')
            
    def twoOpt(self, G):
        """
        non posso farlo con gli indici. devo ciclare in qualche modo sugli archi del ciclo 
        hamiltoniano e invertire il path. cambiando il nodo destinazione dell'arco i e del 
        nodo sorgente dell'arco j, stessa cosa per i+1 e j+1.
        """
        weights = nx.get_edge_attributes(G,'wieght')
        locally_optimal = False
        start_time = time.time()
        
        while not locally_optimal:
            locally_optimal = True
            
            for i in nx.number_of_nodes(G) - 2:
                
                for j in range(i+2, nx.number_of_nodes(G) - 1 if i == 0 else nx.number_of_nodes(G)):
            
          
