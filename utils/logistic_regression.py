import logging
import random
import sqlite3 as sql

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


LOGGER = logging.getLogger()


def classify_training_data():
    conn = sql.connect("madness.db")
    cur = conn.cursor()
    cur.execute("SELECT * FROM TRAINING_SET;")
    raw_data = cur.fetchall()
    conn.close()
    LOGGER.info(f"Retrieved training set (PREVIEW): {raw_data[:10]}")

    LOGGER.info("Classifying data into wins (1) and losses (0)")

    training_data = []
    for row in raw_data:
        # if it's a one, stick with win-focused data
        print(row)
        if random.randint(0, 1):
            training_data.append({1: [value for value in row]})
        else:
            training_data.append({0: [value*-1 for value in row]})

    print(training_data[:20])

# x = np.arange(10).reshape(-1, 1)
# y = np.array([1, 1, 0, 1, 1, 1, 1, 1, 1, 1])

# model = LogisticRegression(solver='liblinear', random_state=0)
# model.fit(x, y)
