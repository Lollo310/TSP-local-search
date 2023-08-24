import time

import networkx as nx


class TSP:

    def __init__(self) -> None:
        self.hamiltonian_cycle = None
        self.hamiltonian_cost = 0

    def NN(self, G: nx.Graph) -> float:
        weights = nx.get_edge_attributes(G, 'weight')
        visited_nodes = {0}
        self.hamiltonian_cycle = []
        node = 0
        start_time = time.time()

        while nx.number_of_nodes(G) != len(visited_nodes):
            last_node = None
            minimum = float('inf')

            for i in range(nx.number_of_nodes(G)):
                if i not in visited_nodes and i != node:
                    x = min(node, i)
                    y = max(node, i)
                    weight = weights[(x, y)]

                    if weight < minimum:
                        minimum = weight
                        last_node = i

            self.hamiltonian_cycle.append((node, last_node))
            visited_nodes.add(last_node)
            node = last_node

        end_time = time.time()
        self.hamiltonian_cycle.append((node, 0))
        self.updateHamiltonianCost(G)

        return end_time - start_time
    
    def RepNN(self, G: nx.Graph) -> float:
        weights = nx.get_edge_attributes(G, 'weight')
        self.hamiltonian_cycle = []
        self.hamiltonian_cost = float('inf')
        start_time = time.time()
        
        for i in range(nx.number_of_nodes(G)):    
            visited_nodes = {i}
            node = i
            prev_cycle = self.hamiltonian_cycle
            prev_cost = self.hamiltonian_cost
            self.hamiltonian_cycle = []
            
            while nx.number_of_nodes(G) != len(visited_nodes):
                last_node = None
                minimum = float('inf')

                for j in range(nx.number_of_nodes(G)):
                    if j not in visited_nodes and j != node:
                        x = min(node, j)
                        y = max(node, j)
                        weight = weights[(x, y)]

                        if weight < minimum:
                            minimum = weight
                            last_node = j

                self.hamiltonian_cycle.append((node, last_node))
                visited_nodes.add(last_node)
                node = last_node

            self.hamiltonian_cycle.append((node, i))
            self.updateHamiltonianCost(G)
            
            if prev_cost < self.hamiltonian_cost:
                self.hamiltonian_cycle = prev_cycle
                self.hamiltonian_cost = prev_cost

        end_time = time.time()
        return end_time - start_time

    def toNetworkX(self, G: nx.Graph) -> nx.Graph:
        try:
            if self.hamiltonian_cycle is None:
                raise ValueError

            H = nx.Graph(self.hamiltonian_cycle)
            nx.set_node_attributes(H, nx.get_node_attributes(G, 'pos'), 'pos')
            nx.set_edge_attributes(
                H, nx.get_edge_attributes(G, 'weight'), 'weight')

            return H
        except ValueError:
            print('Error: Attribute hamiltonian_cycle not be None. Read file before.')

    def updateHamiltonianCost(self, G: nx.Graph) -> None:
        H = self.toNetworkX(G)
        weights = nx.get_edge_attributes(H, 'weight')
        self.hamiltonian_cost = sum(list(weights.values()))

    def twoOpt(self, G: nx.Graph) -> float:
        locally_optimal = False
        start_time = time.time()
        weights = nx.get_edge_attributes(G, 'weight')

        while not locally_optimal:
            locally_optimal = True

            for i in range(nx.number_of_nodes(G) - 2):
                if not locally_optimal:
                    break

                for j in range(i+2, nx.number_of_nodes(G) - 1 if i == 0 else nx.number_of_nodes(G)):
                    gain = self.gain(G, i, j, weights)

                    if gain < 0:
                        self.swap(i, j, gain)
                        locally_optimal = False
                        break
        end_time = time.time()
        self.updateHamiltonianCost(G)

        return end_time - start_time

    def gain(self, G: nx.Graph, i: int, j: int, weights: dict) -> float:
        a, b = self.hamiltonian_cycle[i]
        c, d = self.hamiltonian_cycle[j]

        x = min(a, b)
        y = max(a, b)
        weight_ab = weights[(x, y)]

        x = min(c, d)
        y = max(c, d)
        weight_cd = weights[(x, y)]

        x = min(a, c)
        y = max(a, c)
        weight_ac = weights[(x, y)]

        x = min(b, d)
        y = max(b, d)
        weight_bd = weights[(x, y)]

        return (weight_ac + weight_bd) - (weight_ab + weight_cd)

    def swap(self, i: int, j: int, gain: float) -> None:
        a, b = self.hamiltonian_cycle[i]
        c, d = self.hamiltonian_cycle[j]

        self.hamiltonian_cycle[i] = (a, c)
        self.hamiltonian_cycle[j] = (b, d)
        self.hamiltonian_cycle[i+1:j] = reversed(self.hamiltonian_cycle[i+1:j])

        for k in range(i+1, j):
            self.hamiltonian_cycle[k] = self.hamiltonian_cycle[k][::-1]
