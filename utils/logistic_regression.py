import json
import logging
import random
import sqlite3 as sql

from bracketeer import build_bracket
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


LOGGER = logging.getLogger()

with open("run_config.json", "r") as config_file:
    configs = json.load(config_file)


model = LogisticRegression(solver='liblinear', C=1.0, random_state=0)
coefficient_modifiers = [
    {
        "key": "NOR",
        "position": 0,
        "modifier": 0
    },
    {
        "key": "NDR",
        "position": 1,
        "modifier": 0
    },
    {
        "key": "NTSP",
        "position": 2,
        "modifier": 0
    },
    {
        "key": "N3PTP",
        "position": 3,
        "modifier": 0
    },
    {
        "key": "NFTP",
        "position": 4,
        "modifier": 0
    },
    {
        "key": "NTOVP",
        "position": 5,
        "modifier": 0
    },
    {
        "key": "NRANK",
        "position": 6,
        "modifier": .001
    }
]


def get_db_conn() -> sql.Connection:
    return sql.connect(configs["database_file"])


def adjust_for_net_ranking(team_a: str, team_b: str, predictions: list) -> list:
    """Adjust the model's predicted values by applying NET ranking."""
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
        # Skip net ranking adjustment if the adjustment would be 0 > x > 1
        if abs(net_ranking_adj) > (1 - predictions[1]):
            LOGGER.info("Not applying net ranking updates for this matchup.")
        else:
            predictions[0] = predictions[0] - net_ranking_adj
            predictions[1] = predictions[1] + net_ranking_adj
    elif net_ranking_adj > 0:
        # Skip net ranking adjustment if the adjustment would be 0 > x > 1
        if abs(net_ranking_adj) > (predictions[1] - 0):
            LOGGER.info(f"Not applying net ranking updates for this matchup.")
        else:
            predictions[0] = predictions[0] - net_ranking_adj
            predictions[1] = predictions[1] + net_ranking_adj
    LOGGER.info(f"Predictions adjusted to: {predictions[0]}-{predictions[1]} for matchup: {team_a}-{team_b}")
    return predictions


def get_transformation_fields(context: str = "train"):

    # These are the fields we are considering as statistically significant
    master_field_list = [
    ]
    if context == "train":
        if configs.get("ranking_system"):
            return "NOR, NDR, NTSP, N3PTP, NFTP, NTOVP, NRANK"
        else:
            return "NOR, NDR, NTSP, N3PTP, NFTP, NTOVP"
    if context == "predict":
        return "off_reb, def_reb, tsp, \"3ptp\", ftp, tovp, rank"


def classify_data(training: bool = True, sport: str = "men") -> list:
    """Classify data as a win (1) or loss (0)."""
    conn = sql.connect(configs["database_file"])
    cur = conn.cursor()

    table_name = f"{sport.upper()}_TRAINING_SET" if training else f"{sport.upper()}_TEST_SET"
    if configs.get("ranking_system"):
        cur.execute(f"SELECT {get_transformation_fields()} FROM {table_name};")
    else:
        cur.execute(f"SELECT {get_transformation_fields()} FROM {table_name};")
    raw_data = cur.fetchall()
    conn.close()
    LOGGER.info(f"Retrieved {table_name} set (PREVIEW): {raw_data[:10]}")

    LOGGER.info("Classifying data into wins (1) and losses (0)")

    training_data = []
    for row in raw_data:
        # if it's a 1, stick with win-oriented data
        if table_name == f"{sport.upper()}_TRAINING_SET":
            if random.randint(0, 1):
                training_data.append({1: [value for value in row]})
            else:
                training_data.append({0: [value*-1 for value in row]})
        else:
            training_data.append({1: [value for value in row]})

    return training_data


def train_model(training_data: list):
    """Train model on specified training data."""

    LOGGER.info("--- Fitting model to training data ---")

    observations = []
    result = []
    for obs in training_data:
        key = [ob for ob in obs][0]
        observations.append(key)
        result.append(obs[key])
    LOGGER.info(f"Observations: {observations[:5]}")
    LOGGER.info(f"Results: {result[:5]}")
    LOGGER.info(f"Data shaped for model training")

    LOGGER.info(f"Beginning logistic regression training for {len(training_data)} observations")
    model.fit(result, observations)
    LOGGER.info("Training complete!")

    LOGGER.info(f"Intercept: {model.intercept_}")
    LOGGER.info(f"Coefficients: {model.coef_}")

    if configs["use_coefficient_modifiers"]:
        LOGGER.info("Applying Coefficient modifiers")
        for mod in coefficient_modifiers:
            updated_value = model.coef_[0, mod["position"]] + mod["modifier"]
            model.coef_[0, mod["position"]] = updated_value
            LOGGER.info(f"{mod['key']} new coefficient value: {updated_value}")


def test_model(test_data: list):
    """Test trained model."""

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


def build_predictions(execution_name: str, sport: str = "men"):
    """Predict matchups using trained model."""

    conn = get_db_conn()
    cur = conn.cursor()
    final_results = pd.DataFrame(columns=["team_id_x", "team_id_y", "prob_x", "prob_y"])

    cur.execute(f"SELECT TeamID from team_{sport};")
    team_ids = [team_id[0] for team_id in cur.fetchall()]
    LOGGER.info(f"Running predictions for all matchups of teams: {team_ids}")

    for team_a in team_ids:
        for team_b in team_ids:
            cur.execute(f"SELECT {get_transformation_fields(context='predict')} FROM {sport}_team_stats WHERE team_id='{team_a}';")
            team_a_stats = np.array(cur.fetchone())
            cur.execute(f"SELECT {get_transformation_fields(context='predict')} FROM {sport}_team_stats WHERE team_id='{team_b}';")
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
    final_results.to_sql(f"predictions_{sport}_{execution_name}", con=conn, if_exists="replace", index=False)
    LOGGER.info(f"All results written to table: predictions_{sport}_{execution_name}")


def export_to_csv(execution_name: str, sport: str = "women"):
    """Export the prediction results to CSV."""

    file_name = f"output/bracket_predictions_{sport}_{configs['season']}.csv"
    LOGGER.info(f"Exporting matchup results to CSV: {file_name}")
    query = f"select team_id_x, team_id_y, prob_x from predictions_{sport}_{execution_name};"
    LOGGER.info(f"Running query: {query} to build DataFrame")

    output_base = pd.read_sql(query, con=get_db_conn())
    output_base["ID"] = f"{configs['season']}_" + output_base["team_id_x"].astype(str) + "_" + output_base["team_id_y"].astype(str)
    output_base["Pred"] = output_base["prob_x"]
    header = ["ID", "Pred"]
    output_base.to_csv(file_name, columns=header, index=False)
    LOGGER.info("Matchup results exported to CSV.")


def build_bracketeer_bracket(execution_name: str, sport: str = "women"):
    """Create a PNG file of the bracket."""

    sport_abbrev = "W" if sport == "women" else "M"
    LOGGER.info("Creating bracketeer bracket image!")
    base_path = f"march-machine-learning-mania-{configs['season']}"
    b = build_bracket(
        outputPath=f"output/bracket_{sport_abbrev}_{execution_name}.png",
        teamsPath=f"{base_path}/{sport_abbrev}Teams.csv",
        seedsPath=f"{base_path}/{sport_abbrev}NCAATourneySeeds.csv",
        submissionPath=f"output/bracket_predictions_{sport}_{configs['season']}.csv",
        slotsPath=f"{base_path}/{sport_abbrev}NCAATourneySlots.csv",
        year=2024
    )
    LOGGER.info(f"Bracket image created: bracket_{sport_abbrev}_{execution_name}.png")
