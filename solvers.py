import time
from typing import Tuple

import networkx as nx
import numpy as np


class TSP:

    def __init__(self) -> None:
        self.hamiltonian_cycle = None
        self.hamiltonian_cost = 0

    def NN(self, G: nx.Graph) -> float:
        """Nearest Neighbor algorithm to find a Hamiltonian cycle in a graph.

        Args:
            G (nx.Graph): The input graph.

        Returns:
            float: The time taken to complete the algorithm.
        """
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

    def repNN(self, G: nx.Graph) -> float:
        """Repeated Nearest Neighbor algorithm to find a Hamiltonian cycle in a graph.

        Args:
            G (nx.Graph): The input graph.

        Returns:
            float: The time taken to complete the algorithm.
        """
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
        """Convert the Hamiltonian cycle representation to a NetworkX graph.

        Args:
            G (nx.Graph): The input graph.

        Raises:
            ValueError: If the hamiltonian_cycle attribute is None.

        Returns:
            nx.Graph: The NetworkX graph representing the Hamiltonian cycle.
        """
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
        """Update the cost of the Hamiltonian cycle.

        Args:
            G (nx.Graph): The input graph.
        """
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
                    gain = self.gain(i, j, weights)

                    if gain < 0:
                        self.swap(i, j)
                        locally_optimal = False
                        break
        end_time = time.time()
        self.updateHamiltonianCost(G)

        return end_time - start_time

    def gain(self, i: int, j: int, weights: dict) -> float:
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

    def swap(self, i: int, j: int) -> None:
        a, b = self.hamiltonian_cycle[i]
        c, d = self.hamiltonian_cycle[j]

        self.hamiltonian_cycle[i] = (a, c)
        self.hamiltonian_cycle[j] = (b, d)
        self.hamiltonian_cycle[i+1:j] = reversed(self.hamiltonian_cycle[i+1:j])

        for k in range(i+1, j):
            self.hamiltonian_cycle[k] = self.hamiltonian_cycle[k][::-1]

    def threeOpt(self, G: nx.Graph) -> float:
        locally_optimal = False
        weights = nx.get_edge_attributes(G, 'weight')
        start_time = time.time()

        while not locally_optimal:
            locally_optimal = True

            for i in range(nx.number_of_nodes(G) - 4):
                if not locally_optimal:
                    break

                for j in range(i+2, nx.number_of_nodes(G) - 2):
                    if not locally_optimal:
                        break

                    for k in range(j+2, nx.number_of_nodes(G) - 1 if i == 0 else nx.number_of_nodes(G)):
                        case, gain = self.gainThreeOpt(i, j, k, weights)

                        if gain < 0:
                            self.swapThreeOpt(i, j, k, case, gain)
                            locally_optimal = False
                            break

        end_time = time.time()
        self.updateHamiltonianCost(G)

        return end_time - start_time

    def gainThreeOpt(self, i: int, j: int, k: int, weights: dict) -> Tuple[int, float]:
        a, b = self.hamiltonian_cycle[i]
        c, d = self.hamiltonian_cycle[j]
        e, f = self.hamiltonian_cycle[k]

        dist = np.zeros(8)

        dist[0] = weights[(min(a, b), max(
            a, b))] + weights[(min(c, d), max(c, d))] + weights[(min(e, f), max(e, f))]
        dist[1] = weights[(min(a, c), max(
            a, c))] + weights[(min(b, d), max(b, d))] + weights[(min(e, f), max(e, f))]
        dist[2] = weights[(min(a, e), max(
            a, e))] + weights[(min(c, d), max(c, d))] + weights[(min(b, f), max(b, f))]
        dist[3] = weights[(min(a, b), max(
            a, b))] + weights[(min(c, e), max(c, e))] + weights[(min(d, f), max(d, f))]
        dist[4] = weights[(min(a, c), max(
            a, c))] + weights[(min(b, e), max(b, e))] + weights[(min(d, f), max(d, f))]
        dist[5] = weights[(min(a, e), max(
            a, e))] + weights[(min(b, d), max(b, d))] + weights[(min(c, f), max(c, f))]
        dist[6] = weights[(min(a, d), max(
            a, d))] + weights[(min(c, e), max(c, e))] + weights[(min(b, f), max(b, f))]
        dist[7] = weights[(min(a, d), max(
            a, d))] + weights[(min(b, e), max(b, e))] + weights[(min(c, f), max(c, f))]

        d0 = dist[0]
        gains = [di - d0 for di in dist]

        return np.argmin(gains), np.min(gains)

    def swapThreeOpt(self, i: int, j: int, k: int, case: int, gain: float) -> None:
        if case == 1:
            self.swap(i+1, j)
        elif case == 2:
            self.swap(i+1, k)
        elif case == 3:
            self.swap(j+1, k)
        elif case == 4:
            self.swap(i+1, j)
            self.swap(j+1, k)
        elif case == 5:
            self.swap(i+1, j)
            self.swap(i+1, k)
        elif case == 6:
            self.swap(j+1, k)
            self.swap(i+1, k)
        elif case == 7:
            self.swap(i+1, j)
            self.swap(j+1, k)
            self.swap(i+1, k)
