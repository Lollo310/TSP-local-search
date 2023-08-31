import pandas as pd
import os
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser(description='Merge multiple CSV files into a single DataFrame and save it to a CSV file.')
    parser.add_argument('path', help='The path to the directory containing CSV files for merging.')
    parser.add_argument('-n', '--name', default='results.csv', help='The name of the output merged CSV file.')
    args = parser.parse_args()
    
    dfs_csv = []
        
    for filename in os.listdir(args.path):
        if filename.endswith('.csv'):
            dfs_csv.append(pd.read_csv(args.path + filename, index_col=0))
            
    df_rsl = pd.concat(dfs_csv, ignore_index=True)
    df_rsl.to_csv(args.name)