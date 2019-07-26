import pandas as pd
import csv
import sys

# Initialize dataframes and variables
matchup_df = pd.read_csv('/Users/Patrick/PycharmProjects/March Madness/Matchups2019.csv')
teams_df = pd.read_csv('/Users/Patrick/PycharmProjects/March Madness/2019TeamsCondensed.csv')
regression_df = pd.read_csv('/Users/Patrick/PycharmProjects/March Madness/NCAA Regression Formula.csv')
model = {}

# Test all possible tournament matchups
for index, row in matchup_df.iterrows():
    temp_df = teams_df.loc[teams_df['School'] == matchup_df.iloc[index, 1]]
    temp2_df = teams_df.loc[teams_df['School'] == matchup_df.iloc[index, 2]]

    # Calculate difference between statistics of matched teams
    try:
        fg_attempt = temp_df.iloc[0, 26] - temp2_df.iloc[0, 26]
        fg_pct = temp_df.iloc[0, 9] - temp2_df.iloc[0, 9]
        threept_pct = temp_df.iloc[0, 12] - temp2_df.iloc[0, 12]
        threept_ratio = temp_df.iloc[0, 39] - temp2_df.iloc[0, 39]
        ft_attempt = temp_df.iloc[0, 31] - temp2_df.iloc[0, 31]
        ft_to_fg = temp_df.iloc[0, 25] - temp2_df.iloc[0, 25]
        tot_rebounds = temp_df.iloc[0, 33] - temp2_df.iloc[0, 33]
        turnovers = temp_df.iloc[0, 37] - temp2_df.iloc[0, 37]
        steals = temp_df.iloc[0, 35] - temp2_df.iloc[0, 35]
        assists = temp_df.iloc[0, 34] - temp2_df.iloc[0, 34]
        blocks = temp_df.iloc[0, 36] - temp2_df.iloc[0, 36]
        personal_fouls = temp_df.iloc[0, 24] - temp2_df.iloc[0, 24]
        threept_attempt = temp_df.iloc[0, 29] - temp2_df.iloc[0, 29]
        off_rebounds = temp_df.iloc[0, 32] - temp2_df.iloc[0, 32]

    except IndexError:
        # If there's an error during comparison, identify which record caused it.
        print(matchup_df.iloc[index, 1])
        print(matchup_df.iloc[index, 2])
        sys.exit()

    # Initialize regression coefficients
    intercept = regression_df.iloc[0, 1]
    co1 = regression_df.iloc[1, 1]
    co2 = regression_df.iloc[2, 1]
    co3 = regression_df.iloc[3, 1]
    co4 = regression_df.iloc[4, 1]
    co5 = regression_df.iloc[5, 1]
    co6 = regression_df.iloc[6, 1]
    co7 = regression_df.iloc[7, 1]
    co8 = regression_df.iloc[8, 1]
    co9 = regression_df.iloc[9, 1]
    co10 = regression_df.iloc[10, 1]
    co11 = regression_df.iloc[11, 1]
    co12 = regression_df.iloc[12, 1]
    co13 = regression_df.iloc[13, 1]
    co14 = regression_df.iloc[14, 1]

    # Regression equation; using the differences in stats for each team, calculate the expected point spread
    expected_spread = (intercept + co1 * fg_attempt * fg_pct + co3 * threept_pct + co4 * threept_ratio
                       + co5 * ft_attempt + co6 * ft_to_fg + co7 * tot_rebounds + co8 * turnovers + co9 * steals
                       + co10 * assists + co11 * blocks + co12 * personal_fouls + co13 * threept_attempt
                       + co14 * off_rebounds)

    # Factor in NET Rating to normalize results
    spread = expected_spread + (temp_df.iloc[0, 40] - temp2_df.iloc[0, 40])

    # Append matchup results to the model dict
    spread_final = str(spread)
    model.update({index: spread_final})

# Export to CSV file for analysis
with open('2019 NCAA Predictions_nom_net.csv', 'w') as csv_file:
    writer = csv.writer(csv_file)
    for key, value in model.items():
        writer.writerow([key, matchup_df.iloc[key, 1], matchup_df.iloc[key, 2], value])
