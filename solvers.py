import time
from typing import Tuple

import networkx as nx
import numpy as np


class TSP:
    """This class represents the Traveling Salesman Problem (TSP) solver using various heuristics.
    """

    def __init__(self) -> None:
        """
        Initialize the TSP solver with default attributes.
        """
        self.hamiltonian_cycle = None  # Stores the Hamiltonian cycle
        self.hamiltonian_cost = 0  # Stores the cost of the Hamiltonian cycle

    def NN2Opt(self, G: nx.Graph, rep=False, dlb=False) -> float:
        """This function performs a Nearest Neighbor (NN) heuristic followed by a 2-Opt heuristic on a given graph.

        Args:
            G (nx.Graph): The input graph.
            rep (bool, optional): If True, the repetitive Nearest Neighbor (repNN) heuristic is used. Defaults to False.
            dlb (bool, optional): If True, the Don't Look Bit (2-Opt with Don't Look Bit) is used. Defaults to False.

        Returns:
            float: The total time taken for both heuristics.
        """
        if rep:
            time_NN = self.repNN(G)  # Perform repNN heuristic
        else:
            time_NN = self.NN(G)  # Perform regular NN heuristic

        time_2Opt = self.twoOpt(G, dlb)  # Perform 2-Opt heuristic

        return time_NN + time_2Opt  # Return the total time taken for both heuristics

    def NN3Opt(self, G: nx.Graph, rep=False, dlb=False) -> float:
        """This function performs a Nearest Neighbor (NN) heuristic followed by a 3-Opt heuristic on a given graph.

        Args:
            G (nx.Graph): The input graph.
            rep (bool, optional): If True, the repetitive Nearest Neighbor (repNN) heuristic is used. Defaults to False.
            dlb (bool, optional): If True, the Don't Look Bit (3-Opt with Don't Look Bit) is used. Defaults to False.

        Returns:
            float: The total time taken for both heuristics.
        """
        if rep:
            time_NN = self.repNN(G)  # Perform repNN heuristic
        else:
            time_NN = self.NN(G)  # Perform regular NN heuristic

        time_3Opt = self.threeOpt(G, dlb)  # Perform 2-Opt heuristic

        return time_NN + time_3Opt  # Return the total time taken for both heuristics

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
            TypeError: If the hamiltonian_cycle attribute is None.

        Returns:
            nx.Graph: The NetworkX graph representing the Hamiltonian cycle.
        """
        try:
            if self.hamiltonian_cycle is None:
                raise TypeError

            H = nx.Graph(self.hamiltonian_cycle)
            nx.set_node_attributes(H, nx.get_node_attributes(G, 'pos'), 'pos')
            nx.set_edge_attributes(
                H, nx.get_edge_attributes(G, 'weight'), 'weight')

            return H
        except TypeError:
            print('Error: Attribute hamiltonian_cycle can not be None.')

    def updateHamiltonianCost(self, G: nx.Graph) -> None:
        """Update the cost of the Hamiltonian cycle.

        Args:
            G (nx.Graph): The input graph.
        """
        # Convert Hamiltonian cycle representation to NetworkX graph
        H = self.toNetworkX(G)
        weights = nx.get_edge_attributes(H, 'weight')  # Get edge weights

        # Update the Hamiltonian cycle cost
        self.hamiltonian_cost = sum(list(weights.values()))

    def twoOpt(self, G: nx.Graph, dlb=False) -> float:
        """This function applies the 2-opt heuristic to improve a Hamiltonian cycle in a graph.

        Args:
            G (nx.Graph): The input graph.
            dlb (bool, optional): Whether to use the Don't Look Bit (DLB). Defaults to False.

        Raises:
            TypeError: If the attribute hamiltonian_cycle is None.

        Returns:
            float: The time taken for the 2-opt heuristic to run.
        """
        try:
            if self.hamiltonian_cycle is None:
                raise TypeError

            locally_optimal = False  # Flag to track if a local optimum is reached
            weights = nx.get_edge_attributes(G, 'weight')
            start_time = time.time()

            if dlb:
                dlb_list = [False] * nx.number_of_nodes(G)

            while not locally_optimal:
                locally_optimal = True

                for i in range(nx.number_of_nodes(G) - 2):
                    if not locally_optimal:
                        break

                    if dlb:
                        a, b = self.hamiltonian_cycle[i]
                        if dlb_list[a]:
                            continue
                        node_improved = False

                    for j in range(i+2, nx.number_of_nodes(G) - 1 if i == 0 else nx.number_of_nodes(G)):
                        gain = self.gain(i, j, weights)

                        if gain < 0:
                            # Perform the swap to improve the cycle
                            self.swap(i, j)
                            locally_optimal = False
                            if dlb:
                                c, d = self.hamiltonian_cycle[j]
                                dlb_list = self.setDLB(dlb_list, [a, b, c, d])
                                node_improved = True
                            break

                    if dlb and not node_improved:
                        dlb_list[a] = True
            end_time = time.time()

            # Update the cost of the Hamiltonian cycle
            self.updateHamiltonianCost(G)

            return end_time - start_time  # Return the time taken for the 2-opt heuristic
        except TypeError:
            print('Error: Attribute hamiltonian_cycle can not be None.')

    def gain(self, i: int, j: int, weights: dict) -> float:
        """Calculate the gain achieved by swapping edges i and j in the Hamiltonian cycle.

        Args:
            i (int): Index of the first edge.
            j (int): Index of the second edge.
            weights (dict): Dictionary of edge weights.

        Returns:
            float: The calculated gain.
        """
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
        """Perform a 2-opt swap of edges i and j in the Hamiltonian cycle.

        Args:
            i (int): Index of the first edge.
            j (int): Index of the second edge.
        """
        a, b = self.hamiltonian_cycle[i]
        c, d = self.hamiltonian_cycle[j]

        self.hamiltonian_cycle[i] = (a, c)
        self.hamiltonian_cycle[j] = (b, d)
        self.hamiltonian_cycle[i+1:j] = reversed(self.hamiltonian_cycle[i+1:j])

        for k in range(i+1, j):
            self.hamiltonian_cycle[k] = self.hamiltonian_cycle[k][::-1]

    def setDLB(self, dlb_list: list, node_list: list) -> list:
        """Reset the Don't Look Bit (DLB) flags for the nodes in the given list.

        Args:
            dlb_list (list): List of DLB flags for nodes.
            node_list (list): List of nodes to reset DLB flags for.

        Returns:
            list: Updated DLB flags list after resetting the specified nodes.
        """
        for node in node_list:
            dlb_list[node] = False

        return dlb_list

    def threeOpt(self, G: nx.Graph, dlb=False) -> float:
        """Apply the 3-opt heuristic to improve the Hamiltonian cycle.

        Args:
            G (nx.Graph): The input graph.
            dlb (bool, optional): Whether to use the Don't Look Bit (DLB). Defaults to False.

        Raises:
            TypeError: If the hamiltonian_cycle attribute is None.

        Returns:
            float: The time taken to complete the 3-opt heuristic.
        """
        try:
            if self.hamiltonian_cycle is None:
                raise TypeError

            locally_optimal = False  # Flag to track if a local optimum is reached
            weights = nx.get_edge_attributes(G, 'weight')
            start_time = time.time()

            if dlb:
                dlb_list = [False] * nx.number_of_nodes(G)

            while not locally_optimal:
                locally_optimal = True

                for i in range(nx.number_of_nodes(G) - 4):
                    if not locally_optimal:
                        break

                    if dlb:
                        a, b = self.hamiltonian_cycle[i]
                        if dlb_list[a]:
                            continue
                        node_improved = False

                    for j in range(i+2, nx.number_of_nodes(G) - 2):
                        if not locally_optimal:
                            break

                        for k in range(j+2, nx.number_of_nodes(G) - 1 if i == 0 else nx.number_of_nodes(G)):
                            case, gain = self.gainThreeOpt(i, j, k, weights)

                            if gain < 0:
                                self.swapThreeOpt(i, j, k, case)
                                locally_optimal = False

                                if dlb:
                                    c, d = self.hamiltonian_cycle[j]
                                    e, f = self.hamiltonian_cycle[k]
                                    dlb_list = self.setDLB(
                                        dlb_list, [a, b, c, d, e, f])
                                    node_improved = True

                                break
                    if dlb and not node_improved:
                        dlb_list[a] = True
            end_time = time.time()

            # Update the cost of the Hamiltonian cycle
            self.updateHamiltonianCost(G)

            return end_time - start_time  # Return the time taken for the 3-opt heuristic
        except TypeError:
            print('Error: Attribute hamiltonian_cycle can not be None.')

    def gainThreeOpt(self, i: int, j: int, k: int, weights: dict) -> Tuple[int, float]:
        """Calculate the gain and case achieved by applying the 3-opt swaps in the Hamiltonian cycle.

        Args:
            i (int): Index of the first edge.
            j (int): Index of the second edge.
            k (int): Index of the third edge.
            weights (dict): Dictionary of edge weights.

        Returns:
            Tuple[int, float]: A tuple containing the index of the best case and the corresponding gain.
        """
        a, b = self.hamiltonian_cycle[i]
        c, d = self.hamiltonian_cycle[j]
        e, f = self.hamiltonian_cycle[k]

        dist = np.zeros(8)

        # no exchange
        dist[0] = weights[(min(a, b), max(
            a, b))] + weights[(min(c, d), max(c, d))] + weights[(min(e, f), max(e, f))]
        # 2-opt exchange
        dist[1] = weights[(min(a, c), max(
            a, c))] + weights[(min(b, d), max(b, d))] + weights[(min(e, f), max(e, f))]
        dist[2] = weights[(min(a, e), max(
            a, e))] + weights[(min(c, d), max(c, d))] + weights[(min(b, f), max(b, f))]
        dist[3] = weights[(min(a, b), max(
            a, b))] + weights[(min(c, e), max(c, e))] + weights[(min(d, f), max(d, f))]
        # 3-opt exchange
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

    def swapThreeOpt(self, i: int, j: int, k: int, case: int) -> None:
        """Apply the corresponding 3-opt swaps based on the selected case.

        Args:
            i (int): Index of the first edge.
            j (int): Index of the second edge.
            k (int): Index of the third edge.
            case (int): Index of the selected case.
        """
        # 2-opt exchange
        if case == 1:
            self.swap(i, j)
        elif case == 2:
            self.swap(i, k)
        elif case == 3:
            self.swap(j, k)
        # 3-opt exchange
        elif case == 4:
            self.swap(i, j)
            self.swap(j, k)
        elif case == 5:
            self.swap(i, j)
            self.swap(i, k)
        elif case == 6:
            self.swap(j, k)
            self.swap(i, k)
        elif case == 7:
            self.swap(i, j)
            self.swap(j, k)
            self.swap(i, k)
