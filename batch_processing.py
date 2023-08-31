import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

import solvers as sls
import utils


def getResults(G: nx.Graph, i: int, file: str, results: dict) -> dict:
    """Function to compute various TSP solutions and store results.

    Args:
        G (nx.Graph): The input graph representing the TSP instance.
        i (int): Index for the current TSP instance.
        file (str): The filename of the TSP instance.
        results (dict): A dictionary to store the results.

    Returns:
        dict: The updated results dictionary.
    """
    tsp = sls.TSP()  # Initialize a TSP solver instance

    results['name'][i] = file  # Store the filename in the results dictionary
    # Perform Nearest Neighbor (NN) algorithm
    results['NN_time'][i] = tsp.NN(G)  # Record the execution time
    results['NN_cost'][i] = tsp.hamiltonian_cost  # Record the resulting cost
    # Perform NN + 2Opt algorithm
    """ results['NN2Opt_time'][i] = tsp.NN2Opt(G)
    results['NN2Opt_cost'][i] = tsp.hamiltonian_cost """
    # Perform NN + 2Opt + DLB (Don't Look Bit) algorithm
    results['NN2OptDLB_time'][i] = tsp.NN2Opt(G, dlb=True)
    results['NN2OptDLB_cost'][i] = tsp.hamiltonian_cost
    # Perform NN + 3Opt algorithm
    """  results['NN3Opt_time'][i] = tsp.NN3Opt(G)
    results['NN3Opt_cost'][i] = tsp.hamiltonian_cost """
    # Perform NN + 3Opt + DLB algorithm
    results['NN3OptDLB_time'][i] = tsp.NN3Opt(G, dlb=True)
    results['NN3OptDLB_cost'][i] = tsp.hamiltonian_cost
    # Perform repetitive Nearest Neighbor (repNN) algorithm
    results['repNN_time'][i] = tsp.repNN(G)
    results['repNN_cost'][i] = tsp.hamiltonian_cost
    # Perform repNN + 2Opt algorithm
    """ results['repNN2Opt_time'][i] = tsp.NN2Opt(G, True)
    results['repNN2Opt_cost'][i] = tsp.hamiltonian_cost """
    # Perform repNN + 2Opt + DLB algorithm
    results['repNN2OptDLB_time'][i] = tsp.NN2Opt(G, True, True)
    results['repNN2OptDLB_cost'][i] = tsp.hamiltonian_cost
    # Perform repNN + 3Opt algorithm
    """ results['repNN3Opt_time'][i] = tsp.NN3Opt(G, True)
    results['repNN3Opt_cost'][i] = tsp.hamiltonian_cost """
    # Perform repNN + 3Opt + DLB algorithm
    results['repNN3OptDLB_time'][i] = tsp.NN3Opt(G, True, True)
    results['repNN3OptDLB_cost'][i] = tsp.hamiltonian_cost

    return results


def getDf(tsp_file_list):
    """Function to generate a DataFrame with TSP results.

    Args:
        tsp_file_list (list): List of TSP filenames.

    Returns:
        pandas.DataFrame: The DataFrame containing the results.
    """
    loader = utils.Loader()  # Initialize a loader instance
    n = len(tsp_file_list)  # Get the number of TSP instances

    # Create a dictionary to hold the results
    results = {
        'name': ['']*n,
        'NN_cost': np.zeros(n),
        'NN_time': np.zeros(n),
        'NN2Opt_cost': np.zeros(n),
        'NN2Opt_time': np.zeros(n),
        'NN2OptDLB_cost': np.zeros(n),
        'NN2OptDLB_time': np.zeros(n),
        'NN3Opt_cost': np.zeros(n),
        'NN3Opt_time': np.zeros(n),
        'NN3OptDLB_cost': np.zeros(n),
        'NN3OptDLB_time': np.zeros(n),
        'repNN_cost': np.zeros(n),
        'repNN_time': np.zeros(n),
        'repNN2Opt_cost': np.zeros(n),
        'repNN2Opt_time': np.zeros(n),
        'repNN2OptDLB_cost': np.zeros(n),
        'repNN2OptDLB_time': np.zeros(n),
        'repNN3Opt_cost': np.zeros(n),
        'repNN3Opt_time': np.zeros(n),
        'repNN3OptDLB_cost': np.zeros(n),
        'repNN3OptDLB_time': np.zeros(n)
    }

    # Loop through each TSP instance and compute results
    for i, file in tqdm(enumerate(tsp_file_list), total=len(tsp_file_list)):
        loader.readFile('TSP_datasets/' + file)  # Load the TSP instance
        G = loader.createNxGraph()  # Create a NetworkX graph from the instance
        results = getResults(G, i, file, results)  # Compute and store results

    return pd.DataFrame(data=results)  # Convert results to a DataFrame


if __name__ == '__main__':
    utl = utils.Utilities()  # Initialize utilities instance

    tsp_file_list = utl.getTspFiles('test')  # Get the list of TSP filenames

    df = getDf(tsp_file_list)  # Generate DataFrame with results

    df.to_csv('TSP_results.csv')  # Save the DataFrame to a CSV file
