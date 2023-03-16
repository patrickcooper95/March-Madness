import sqlite3 as sql


def add_conference_multiplier(team_a: str, team_b: str, prediction: list):

    conn = sql.connect("madness.db")
    cur = conn.cursor()
    cur.execute("SELECT TeamName,  FROM team_men WHERE TeamID=")
