import logging
import sqlite3 as sql

import pandas as pd


LOGGER = logging.getLogger()


def get_db_conn():
    return sql.connect("madness.db")


def create_database():
    """Load data from CSVs to a SQLite Database."""
    file_to_table_mapping = {
        "Cities.csv": "city",
        "Conferences.csv": "conference",
        "MConferenceTourneyGames.csv": "conference_tourney_games_men",
        "MGameCities.csv": "game_city_men",
        "MMasseyOrdinals.csv": "massey_ordinal_men",
        "MNCAATourneyCompactResults.csv": "tourney_compact_results_men",
        "MNCAATourneyDetailedResults.csv": "tourney_detailed_results_men",
        "MNCAATourneySeedRoundSlots.csv": "tourney_seed_round_slots_men",
        "MNCAATourneySeeds.csv": "tourney_seeds_men",
        "MNCAATourneySlots.csv": "tourney_slots_men",
        "MRegularSeasonCompactResults.csv": "regular_season_compact_results_men",
        "MRegularSeasonDetailedResults.csv": "regular_season_detailed_results_men",
        "MSeasons.csv": "season_men",
        "MTeamConferences.csv": "team_conference_men",
        "MTeams.csv": "team_men",
        # "MTeamSpellings.csv": "team_spelling_men", -- Gives unicode decode error, probably not needed anyway
        "WGameCities.csv": "game_city_women",
        "WNCAATourneyCompactResults.csv": "tourney_compact_results_women",
        "WNCAATourneyDetailedResults.csv": "tourney_detailed_results_women",
        "WNCAATourneySeeds.csv": "tourney_seeds_women",
        "WNCAATourneySlots.csv": "tourney_slots_women",
        "WRegularSeasonCompactResults.csv": "regular_season_compact_results_women",
        "WRegularSeasonDetailedResults.csv": "regular_season_detailed_results_women",
        "WSeasons.csv": "season_women",
        "WTeamConferences.csv": "team_conference_women",
        "WTeams.csv": "team_women",
        # "WTeamSpellings.csv": "team_spelling_women" -- Gives unicode decode error, probably not needed anyway
    }

    conn = sql.connect("madness.db")
    LOGGER.info("Connected to / created: madness.db")
    for file, table in file_to_table_mapping.items():
        LOGGER.info(f"Trying {file} -> {table}")

        df = pd.read_csv(f"march-machine-learning-mania-2023/{file}")
        LOGGER.info(f"Created a DataFrame for: {file}")

        df.to_sql(table, conn, if_exists="replace", index=False)
        LOGGER.info(f"Created table: {table}")

    LOGGER.info("Data loaded")


def create_advanced_statistics():
    """Create insightful stats from existing data."""

    BASE_ADD_COLUMN = "ALTER TABLE {} ADD COLUMN {} REAL;"
    conn = get_db_conn()
    cur = conn.cursor()

    # TSA and TSP / FG% / 3PT % / FT %
    alter_tables = (
        "regular_season_detailed_results_men",
        "tourney_detailed_results_men",
        "regular_season_detailed_results_women",
        "tourney_detailed_results_women"
    )
    for table in alter_tables:

        LOGGER.info(f"Creating the following stats: TSA, TSP, FG%, 3PTFG%, FT% for: {table}")
        # Add TSA Columns
        cur.execute(BASE_ADD_COLUMN.format(table, "WTSA"))
        cur.execute(BASE_ADD_COLUMN.format(table, "LTSA"))
        cur.execute(BASE_ADD_COLUMN.format(table, "WTSP"))
        cur.execute(BASE_ADD_COLUMN.format(table, "LTSP"))
        cur.execute(BASE_ADD_COLUMN.format(table, "WFGP"))
        cur.execute(BASE_ADD_COLUMN.format(table, "LFGP"))
        cur.execute(BASE_ADD_COLUMN.format(table, "W3PTP"))
        cur.execute(BASE_ADD_COLUMN.format(table, "L3PTP"))
        cur.execute(BASE_ADD_COLUMN.format(table, "WFTP"))
        cur.execute(BASE_ADD_COLUMN.format(table, "LFTP"))

        # Insert all but TSP values
        # TODO: Determine why some columns are not populating
        cur.execute(f"""UPDATE {table}
            SET WTSA=WFGA+WFTA*0.44,
                LTSA=LFGA+LFTA*0.44,
                WFGP=WFGM/WFGA,
                LFGP=LFGM/LFGA,
                W3PTP=WFGM3/WFGA3,
                L3PTP=LFGM3/LFGA3,
                WFTP=WFTM/WFTA,
                LFTP=LFTM/LFTA
            ;   
            """
        )
        LOGGER.info(f"TSA, FG%, 3PTFG%, FT% added to table: {table}")

        # Insert TSP Values
        cur.execute(f"UPDATE {table} SET WTSP=WScore/(2*WTSA), LTSP=LScore/(2*LTSA);")
        LOGGER.info(f"TSP inserted successfully for table: {table}")

        # Commit after each table is complete
        conn.commit()
