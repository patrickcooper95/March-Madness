import pandas as pd
import csv

"""
    Matchups.py creates a csv file of all possible matchups in the NCAA Tournament
    and assigns an ID to each using the Matchups.csv file, which contains all teams
    participating in the tournament.
    
    The result is an output csv file of three columns: ID, Team1, Team2.
    """
with open('Matchups.csv', 'rb') as f:
    reader = csv.reader(f)
    teams = list(reader)

df = pd.DataFrame()
inner_index = 0
index = 0
SAVE_PATH = '/Users/Patrick/PycharmProjects/March Madness/Matchups_Final'

for t in teams:
    for e in teams:
        df.loc[inner_index, 'Teams1'] = teams[index]
        df.loc[inner_index, 'Teams2'] = e
        df.loc[inner_index, 'ID'] = inner_index
        inner_index += 1
    index += 1
df.to_csv(path_or_buf=SAVE_PATH, sep=',')
