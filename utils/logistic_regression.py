import logging
import random
import sqlite3 as sql

from bracketeer import build_bracket
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


LOGGER = logging.getLogger()


model = LogisticRegression(solver='liblinear', C=5.0, random_state=0)


def get_db_conn():
    return sql.connect("madness.db")


def adjust_for_net_ranking(team_a, team_b, predictions):
    conn = get_db_conn()
    cur = conn.cursor()
    LOGGER.info("Adjusting predictions based on NET rankings...")
    team_a_name = cur.execute(f"SELECT TeamName FROM team_men WHERE TeamID='{team_a}';").fetchone()[0]
    team_b_name = cur.execute(f"SELECT TeamName FROM team_men WHERE TeamID='{team_b}';").fetchone()[0]

    # Fix apostrophe issue
    team_a_name = team_a_name.replace("'s", "''s")
    team_b_name = team_b_name.replace("'s", "''s")

    # Get NET Ranking
    net_constant = 363
    net_ranking_a = cur.execute(f"SELECT net_ranking from \"2023_final_net_rankings\" WHERE team_name='{team_a_name}';").fetchone()[0]
    net_ranking_b = cur.execute(f"SELECT net_ranking from \"2023_final_net_rankings\" WHERE team_name='{team_b_name}';").fetchone()[0]

    net_ranking = net_ranking_a - net_ranking_b
    LOGGER.info(f"Net ranking for matchup: {net_ranking}")

    net_ranking_adj = net_ranking / net_constant
    if net_ranking_adj < 0:
        if abs(net_ranking_adj) > (1 - predictions[1]):
            LOGGER.info("Not applying net ranking updates for this matchup.")
        else:
            predictions[0] = predictions[0] - net_ranking_adj
            predictions[1] = predictions[1] + net_ranking_adj
    elif net_ranking_adj > 0:
        if abs(net_ranking_adj) > (predictions[1] - 0):
            LOGGER.info(f"Not applying net ranking updates for this matchup.")
        else:
            predictions[0] = predictions[0] - net_ranking_adj
            predictions[1] = predictions[1] + net_ranking_adj
    LOGGER.info(f"Predictions adjusted to: {predictions[0]}-{predictions[1]} for matchup: {team_a}-{team_b}")
    return predictions


def classify_data(training: bool = True):
    conn = sql.connect("madness.db")
    cur = conn.cursor()

    table_name = "TRAINING_SET" if training else "TEST_SET"
    cur.execute(f"SELECT NOR, NDR, NTSP, N3PTP, NFTP, NTOVP FROM {table_name};")
    raw_data = cur.fetchall()
    conn.close()
    LOGGER.info(f"Retrieved {table_name} set (PREVIEW): {raw_data[:10]}")

    LOGGER.info("Classifying data into wins (1) and losses (0)")

    training_data = []
    for row in raw_data:
        # if it's a 1, stick with win-focused data
        if table_name == "TRAINING_SET":
            if random.randint(0, 1):
                training_data.append({1: [value for value in row]})
            else:
                training_data.append({0: [value*-1 for value in row]})
        else:
            training_data.append({1: [value for value in row]})

    return training_data


def train_model(training_data: list):

    LOGGER.info("--- Fitting model to training data ---")

    observations = []
    result = []
    for obs in training_data:
        key = [ob for ob in obs][0]
        observations.append(key)
        result.append(obs[key])
    LOGGER.info(f"Data shaped for model training")

    LOGGER.info(f"Beginning logistic regression training for {len(training_data)} observations")
    model.fit(result, observations)
    LOGGER.info("Training complete!")

    LOGGER.info(f"Intercept: {model.intercept_}")
    LOGGER.info(f"Coefficients: {model.coef_}")


def test_model(test_data: list):
    """Test trained model"""
    LOGGER.info("--- Beginning test of trained model ---")
    LOGGER.info(f"{len(test_data)} observations to test")

    observations = []
    for obs in test_data:
        key = [ob for ob in obs][0]
        observations.append(obs[key])
    prob_result = model.predict_proba(observations)

    total = len(prob_result)
    correct = 0
    for i in range(len(prob_result)):
        if prob_result[i][1] > .5:
            correct += 1

    scoring_array = [1] * len(test_data)
    LOGGER.info(f"Raw Accuracy: {correct/total}")
    LOGGER.info(f"Score: {model.score(observations, scoring_array)}")
    LOGGER.info(f"Classification Report:\n{classification_report(scoring_array, model.predict(observations))}")


def build_predictions(execution_name: str):

    conn = get_db_conn()
    cur = conn.cursor()
    final_results = pd.DataFrame(columns=["team_id_x", "team_id_y", "prob_x", "prob_y"])

    cur.execute("SELECT TeamID from team_men;")
    team_ids = [team_id[0] for team_id in cur.fetchall()]
    LOGGER.info(f"Running predictions for all matchups of teams: {team_ids}")

    for team_a in team_ids:
        for team_b in team_ids:
            cur.execute(f"SELECT off_reb, def_reb, tsp, \"3ptp\", ftp, tovp FROM team_stats WHERE team_id='{team_a}';")
            team_a_stats = np.array(cur.fetchone())
            cur.execute(f"SELECT off_reb, def_reb, tsp, \"3ptp\", ftp, tovp FROM team_stats WHERE team_id='{team_b}';")
            team_b_stats = np.array(cur.fetchone())

            if not team_a_stats.any():
                LOGGER.info(f"Team stats missing for: {team_a} - SKIPPING")
                break
            if not team_b_stats.any():
                print(team_b_stats)
                LOGGER.info(f"Team stats missing for: {team_b} - SKIPPING")
                continue

            combined_result = np.subtract(team_a_stats, team_b_stats).reshape(1, -1)
            prediction = model.predict_proba(combined_result)[0]
            LOGGER.info(f"Team: {team_a} probability to beat team: {team_b}: {prediction}")

            # Still a work in progress - comment out for now
            # prediction = adjust_for_net_ranking(team_a, team_b, prediction)

            # Add result to DataFrame
            final_results = pd.concat(
                [
                    final_results,
                    pd.DataFrame(
                        {
                            "team_id_x": [team_a],
                            "team_id_y": [team_b],
                            "prob_x": [prediction[1]],
                            "prob_y": [prediction[0]]
                        }
                    )
                ],
                ignore_index=False
            )
    final_results.to_sql(f"predictions_{execution_name}", con=conn, if_exists="replace", index=False)
    LOGGER.info(f"All results written to table: predictions_{execution_name}")


def export_to_csv(execution_name: str):

    file_name = "bracket_predictions_2023.csv"
    LOGGER.info(f"Exporting matchup results to CSV: {file_name}")
    query = f"select team_id_x, team_id_y, prob_x from predictions_{execution_name};"
    LOGGER.info(f"Running query: {query} to build DataFrame")

    output_base = pd.read_sql(query, con=get_db_conn())
    output_base["ID"] = "2023_" + output_base["team_id_x"].astype(str) + "_" + output_base["team_id_y"].astype(str)
    output_base["Pred"] = output_base["prob_x"]
    header = ["ID", "Pred"]
    output_base.to_csv(file_name, columns=header, index=False)
    LOGGER.info("Matchup results exported to CSV.")


def build_bracketeer_bracket(execution_name: str):
    """Create a PNG file of the bracket."""

    LOGGER.info("Creating bracketeer bracket image!")
    base_path = "march-machine-learning-mania-2023"
    b = build_bracket(
        outputPath=f"bracket_{execution_name}.png",
        teamsPath=f"{base_path}/MTeams.csv",
        seedsPath=f"{base_path}/MNCAATourneySeeds.csv",
        submissionPath="bracket_predictions_2023.csv",
        slotsPath=f"{base_path}/MNCAATourneySlots.csv",
        year=2023
    )
    LOGGER.info(f"Bracket image created: bracket_{execution_name}.png")
