import networkx as nx
import utils
import solvers as sls
import matplotlib.pyplot as plt
import os
import pandas as pd

def getCsv(tsp_file_list):
    for file in tsp_file_list:
        loader.readFile('TSP_datasets/' + file)
        G = loader.createNxGraph()

if __name__ == '__main__':
    utl = utils.Utilities()
    loader = utils.Loader()
    tsp = sls.TSP()
    
    tsp_file_list = utl.getTspFiles('TSP_datasets')
    
    getCsv(tsp_file_list)