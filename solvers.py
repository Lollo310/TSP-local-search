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
        
        self.hamiltonian_cycle.append((node, 0))
        
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
            
    def twoOpt(self, G: nx.graph) -> float:
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
            
            for i in range(nx.number_of_nodes(G) - 2):
                
                for j in range(i+2, nx.number_of_nodes(G) - 1 if i == 0 else nx.number_of_nodes(G)):
                   gain = self.gain(G,i,j)
                   
                   if gain < 0:
                       self.swap(i,j)
                       locally_optimal = False
        end_time = time.time()
        
        return end_time - start_time
    
    def gain(self, G: nx.graph, i: int, j: int) -> float:
        weights = nx.get_edge_attributes(G,'wieght')
        
        a, b = self.hamiltonian_cycle[i]
        c, d = self.hamiltonian_cycle[j]
        
        x = min(a, b)
        y = max(a, b)
        weight_ab = weights[(x,y)]
        
        x = min(c, d)
        y = max(c, d)
        weight_cd = weights[(x,y)]
        
        x = min(a, c)
        y = max(a, c)
        weight_ac = weights[(x,y)]
        
        x = min(b, d)
        y = max(b, d)
        weight_bd = weights[(x,y)]
        
        return (weight_ac + weight_bd) - (weight_ab + weight_cd)
    
    def swap(self, i, j):
        pass
        