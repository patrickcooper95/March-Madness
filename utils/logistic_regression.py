import logging
import random
import sqlite3 as sql

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


LOGGER = logging.getLogger()


model = LogisticRegression(solver='liblinear', C=10.0, random_state=0)


def classify_data(training: bool = True):
    conn = sql.connect("madness.db")
    cur = conn.cursor()

    table_name = "TRAINING_SET" if training else "TEST_SET"
    cur.execute(f"SELECT NOR, NDR, NTSP, NFTP, NTOVP FROM {table_name};")
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
        if prob_result[i][0] > .5:
            correct += 1

    scoring_array = [1] * len(test_data)
    LOGGER.info(f"Raw Accuracy: {correct/total}")
    LOGGER.info(f"Score: {model.score(observations, scoring_array)}")
    LOGGER.info(f"Classification Report:\n{classification_report(scoring_array, model.predict(observations))}")