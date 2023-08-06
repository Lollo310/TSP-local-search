import networkx as nx

class tsp:
    
    def __init__(self) -> None:
        self.hamiltonian_cycle = None
        
    def NN(self, G) -> nx.Graph:
        for node in G.nodes():
            list(G.neighbors(node))    
