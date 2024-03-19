import csv
import json
import logging
import os
import sqlite3 as sql
from sqlite3 import OperationalError

import pandas as pd


LOGGER = logging.getLogger()

with open("run_config.json", "r") as config_file:
    configs = json.load(config_file)

name_exception_path = "external_sources/name_exceptions.csv"


def get_db_conn() -> sql.Connection:
    return sql.connect(configs["database_file"])


def create_database(add_external_sources: bool = True):
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

    conn = sql.connect(configs["database_file"])
    LOGGER.info(f"Connected to / created: {configs['database_file']}")

    # Create database metadata table
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS _metadata;")
    cur.execute("CREATE TABLE _metadata (table_name TEXT, external_source INTEGER);")

    for file, table in file_to_table_mapping.items():
        LOGGER.info(f"Trying {file} -> {table}")

        df = pd.read_csv(f"march-machine-learning-mania-2023/{file}")
        LOGGER.info(f"Created a DataFrame for: {file}")

        df.to_sql(table, conn, if_exists="replace", index=False)
        LOGGER.info(f"Created table: {table}")

    if add_external_sources:
        LOGGER.info("Loading external data sources")
        for file in os.listdir("external_sources"):
            if file.endswith(".csv") and "name_exceptions" not in file:
                LOGGER.info(f"Attempting to load: {file}")
                df = pd.read_csv(f"external_sources/{file}")
                table_name = file.split(".")[0]
                df.to_sql(table_name, conn, if_exists="replace", index=False)

                # Record table in _metadata as external source
                cur.execute(f"INSERT INTO _metadata VALUES ('{table_name}', 1);")
                LOGGER.info(f"File: {file} loaded")

    conn.commit()
    conn.close()
    LOGGER.info("Data loaded")


def standardize_team_names():
    conn = get_db_conn()
    cur = conn.cursor()

    LOGGER.info("Standardizing team names...")
    # Men
    cur.execute("SELECT TeamName FROM team_men;")
    men_teams = cur.fetchall()
    men_teams = [team[0] for team in men_teams]

    # Only required for external data - Kaggle data should already be standardized
    external_source_tables = cur.execute("SELECT table_name FROM _metadata WHERE external_source=1;").fetchall()
    external_source_tables = [table[0] for table in external_source_tables]
    LOGGER.info(f"Found external source tables: {external_source_tables}")

    if os.path.isfile("external_sources/team_name_mapping.json"):
        LOGGER.info("Team name mapping file found - loading JSON file")
        with open("external_sources/team_name_mapping.json", "r") as json_file:
            team_name_mapping = json.load(json_file)

    for table in external_source_tables:
        LOGGER.info(f"Updating team names in table: {table}")
        # Get some easy wins with Saints and States
        cur.execute(f"UPDATE '{table}' SET team_name=replace(team_name,'St.', 'St') WHERE team_name like '%St.';")
        cur.execute(f"UPDATE '{table}' SET team_name=replace(team_name,'St.', 'St') WHERE team_name like 'St.%';")
        cur.execute(f"UPDATE '{table}' SET team_name=replace(team_name,'Saint', 'St') WHERE team_name like 'Saint%';")

        LOGGER.info("Standard updates made - proceeding to mapped name updates")

        conn.commit()

        # Handle all other cases - mapped in JSON
        ext_team_names = [team[0] for team in cur.execute(f"SELECT team_name FROM '{table}';").fetchall()]
        for proper_name, alt_names in team_name_mapping.items():
            if any([name for name in alt_names if name in ext_team_names]):
                for name in alt_names:
                    if name in ext_team_names:
                        # Escape the single quote for the query
                        if "'" in name:
                            name = name.replace("'", "''")
                        cur.execute(f"UPDATE '{table}' SET team_name='{proper_name}' WHERE team_name='{name}';")
        LOGGER.info(f"Team names standardized for table: {table}")

        conn.commit()

    if not os.path.isfile(name_exception_path):
        LOGGER.info("Running team name exceptions check")
        mismatches = []
        for table in external_source_tables:
            external_team_names = cur.execute(f"SELECT team_name FROM '{table}';").fetchall()
            for team in external_team_names:
                if team[0] not in men_teams:
                    LOGGER.info(f"{team[0]} not in men_teams")
                    mismatches.append(team[0])

        LOGGER.info(f"{len(mismatches)} name exceptions found!")
        LOGGER.info(f"Writing name exceptions file to: {name_exception_path}")
        with open(name_exception_path, "w", newline='') as exceptions_file:
            writer = csv.writer(exceptions_file)
            for name in mismatches:
                writer.writerow([name])

    conn.commit()
    conn.close()


def create_advanced_statistics():
    """Create insightful stats from existing data."""

    BASE_ADD_COLUMN = "ALTER TABLE {} ADD COLUMN {} REAL;"
    conn = get_db_conn()
    cur = conn.cursor()

    # TSA and TSP / FG% / 3PT % / FT %
    alter_tables = {
        "regular_season_detailed_results_men",
        "tourney_detailed_results_men",
        "regular_season_detailed_results_women",
        "tourney_detailed_results_women"
    }
    for table in alter_tables:

        LOGGER.info(f"Creating the following stats: TSA, TSP, FG%, 3PTFG%, FT%, TOVP for: {table}")
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
        cur.execute(BASE_ADD_COLUMN.format(table, "WTOVP"))
        cur.execute(BASE_ADD_COLUMN.format(table, "LTOVP"))

        # Insert all but TSP values
        cur.execute(f"""UPDATE {table}
            SET WTSA=WFGA+WFTA*0.44,
                LTSA=LFGA+LFTA*0.44,
                WFGP=CAST(WFGM AS REAL)/WFGA,
                LFGP=CAST(LFGM AS REAL)/LFGA,
                W3PTP=CAST(WFGM3 AS REAL)/WFGA3,
                L3PTP=CAST(LFGM3 AS REAL)/LFGA3,
                WFTP=CAST(WFTM AS REAL)/WFTA,
                LFTP=CAST(LFTM AS REAL)/LFTA,
                WTOVP=CAST((100 * WTO) AS REAL)/CAST((WFGA+0.44*WFTA+WTO) AS REAL),
                LTOVP=CAST((100 * LTO) AS REAL)/CAST((LFGA+0.44*LFTA+LTO) AS REAL)
            ;   
            """
        )
        LOGGER.info(f"TSA, FG%, 3PTFG%, FT%, TOV% added to table: {table}")

        # Insert TSP Values
        cur.execute(f"UPDATE {table} SET WTSP=WScore/(2*WTSA), LTSP=LScore/(2*LTSA);")
        LOGGER.info(f"TSP inserted successfully for table: {table}")

        # Commit after each table is complete
        conn.commit()

    patch_null_values(alter_tables)


def patch_null_values(table_set: set):
    """Update known null values with appropriate placeholders."""

    conn = get_db_conn()
    cur = conn.cursor()

    # WFTP/LFTP are the only known columns to have NULLs
    for table in table_set:
        LOGGER.info(f"Patching null values for table: {table}")
        cur.execute(f'UPDATE "{table}" SET WFTP=.5 WHERE WFTP IS NULL;')
        cur.execute(f'UPDATE "{table}" SET LFTP=.5 WHERE LFTP IS NULL;')
        cur.execute(f'UPDATE "{table}" SET W3PTP=0 WHERE W3PTP IS NULL;')
        cur.execute(f'UPDATE "{table}" SET L3PTP=0 WHERE L3PTP IS NULL;')
        conn.commit()
    LOGGER.info(f"NULL fields patched for tables: {table_set}")


def create_team_aggregate_table(sport: str = "men"):

    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute(f"DROP TABLE IF EXISTS {sport}_team_stats;")
    cur.execute(f"""CREATE TABLE {sport}_team_stats (
                    team_id TEXT PRIMARY_KEY,
                    fg_made REAL,
                    fg_att REAL,
                    fg_made_3 REAL,
                    fg_att_3 REAL,
                    ft_made REAL,
                    ft_att REAL,
                    off_reb REAL,
                    def_reb REAL,
                    ast REAL,
                    "to" REAL,
                    stl REAL,
                    blk REAL,
                    pf REAL,
                    tsa REAL,
                    tsp REAL,
                    fgp REAL,
                    "3ptp" REAL,
                    ftp REAL,
                    tovp REAL);
                    """)
    LOGGER.info("Created team aggregate stats table")
    conn.commit()


def build_team_aggregates(sport: str = "men"):

    conn = get_db_conn()
    cur = conn.cursor()

    LOGGER.info(f"Building team aggregate stats for: {sport}")
    cur.execute(f"SELECT TeamID FROM team_{sport};")
    team_ids = [id[0] for id in cur.fetchall()]

    create_team_aggregate_table()
    for team_id in team_ids:
        games_won = f"""SELECT WTeamID as team_id,
                            WFGM as fg_made,
                            WFGA as fg_att,
                            WFGM3 as fg_made_3,
                            WFGA3 as fg_att_3,
                            WFTM as ft_made,
                            WFTA as ft_att,
                            WOR as off_reb,
                            WDR as def_reb,
                            WAst as ast,
                            WTO as "to",
                            WStl as stl,
                            WBlk as blk,
                            WPF as pf,
                            WTSA as tsa,
                            WTSP as tsp,
                            WFGP as fgp,
                            W3PTP as "3ptp",
                            WFTP as ftp,
                            WTOVP as tovp 
                        FROM regular_season_detailed_results_{sport}
                        WHERE WTeamID='{team_id}' and Season='2023';"""
        games_lost = f"""SELECT LTeamID as team_id,
                            LFGM as fg_made,
                            LFGA as fg_att,
                            LFGM3 as fg_made_3,
                            LFGA3 as fg_att_3,
                            LFTM as ft_made,
                            LFTA as ft_att,
                            LOR as off_reb,
                            LDR as def_reb,
                            LAst as ast,
                            LTO as "to",
                            LStl as stl,
                            LBlk as blk,
                            LPF as pf,
                            LTSA as tsa,
                            LTSP as tsp,
                            LFGP as fgp,
                            L3PTP as "3ptp",
                            LFTP as ftp,
                            LTOVP as tovp 
                        FROM regular_season_detailed_results_{sport}
                        WHERE LTeamID='{team_id}' and Season='2023';"""

        LOGGER.info(f"Compiled game stats for team: {team_id}")

        games_df = pd.concat([pd.read_sql(games_won, conn), pd.read_sql(games_lost, conn)])
        avg_stats = games_df.mean()

        avg_stats_df = avg_stats.to_frame().T
        clean_avg_stats_df = avg_stats_df.dropna()
        clean_avg_stats_df["team_id"] = clean_avg_stats_df["team_id"].astype(int).astype(str)
        clean_avg_stats_df.to_sql(f"{sport}_team_stats", con=conn, if_exists="append", index=False)


def create_massey_ordinal_mapping(ranking_system: str = "POM"):
    """Create a table that maps a season day to a ranking."""

    dataframe_export = False
    LOGGER.info(
        f"Creating massey ordinal mapping using system: {ranking_system} "
        + f"and a dataframe_export setting of: {dataframe_export}"
    )

    conn = get_db_conn()
    cur = conn.cursor()

    # For each historical game, find the ranking
    games = cur.execute(
        """
                SELECT RowID, Season, DayNum, WTeamID, LTeamID
                FROM regular_season_detailed_results_men;
        """
    ).fetchall()

    LOGGER.info("Delete massey ordinal matchup mapping table if exists...")
    cur.execute("DROP TABLE IF EXISTS ordinal_mapping_men;")
    LOGGER.info("Dropped table: ordinal_mapping_men")

    rankings_df = pd.read_sql_query(
        """SELECT *
        FROM massey_ordinal_men;""",
        con=conn
    )
    LOGGER.info(f"Retrieved all massey ordinals as dataframe: {rankings_df.head()}")

    matchup_ordinals = []
    for game in games:
        matchup_row_id = game[0]
        season = game[1]
        day_num = game[2]
        wteam_id = game[3]
        lteam_id = game[4]

        winning_team_index = find_nearest_ranking_day(
            rankings_df[
                (rankings_df["TeamID"] == wteam_id) &
                (rankings_df["Season"] == season) &
                (rankings_df["SystemName"] == ranking_system)
            ],
            day_num
        )
        losing_team_index = find_nearest_ranking_day(
            rankings_df[
                (rankings_df["TeamID"] == lteam_id) &
                (rankings_df["Season"] == season) &
                (rankings_df["SystemName"] == ranking_system)
                ],
            day_num
        )
        LOGGER.info(f"Found {ranking_system} ranking for Winning Team: {wteam_id} and Losing Team: {lteam_id}")

        try:
            winning_team_ranking = rankings_df.iloc[[winning_team_index]]["OrdinalRank"].values[0]
            losing_team_ranking = rankings_df.iloc[[losing_team_index]]["OrdinalRank"].values[0]
        except TypeError:
            LOGGER.error(f"One of the following teams is missing ranking data for this matchup: {wteam_id} or {lteam_id}")
            LOGGER.warning(f"Matchup: {matchup_row_id} will be ignored")
            continue

        LOGGER.info(f"Adding matchup {matchup_row_id} ranking mapping")

        if dataframe_export:
            matchup_ordinals.append(
                {
                    "matchup_row_id": matchup_row_id,
                    "wteam_ranking": winning_team_ranking,
                    "lteam_ranking": losing_team_ranking,
                    "ranking_system": ranking_system,
                }
            )
        else:
            matchup_ordinals.append(
                (
                    winning_team_ranking,
                    losing_team_ranking,
                    matchup_row_id,
                )
            )

    LOGGER.info("Writing massey ordinal mapping to table: regular_season_detailed_results_men")
    check_or_add_rank_columns()

    if dataframe_export:
        # Convert list of dicts to DataFrame and then export to SQL
        pd.DataFrame(matchup_ordinals).to_sql("ordinal_mapping_men", con=conn)
    else:
        # SQLite batch insert
        for matchup in matchup_ordinals:
            cur.execute(
                f"""
                    UPDATE regular_season_detailed_results_men
                    SET WRANK={matchup[0]}, LRANK={matchup[1]} 
                    WHERE rowid={matchup[2]};
                """
            )
    conn.commit()
    conn.close()


def find_nearest_ranking_day(df, day_num):
    exact_match = df[df["RankingDayNum"] == day_num]
    if not exact_match.empty:
        return exact_match.index.values[0]
    else:
        try:
            lower_neighbor = df[df["RankingDayNum"] < day_num]["RankingDayNum"].idxmax()
        except ValueError:
            # if there is no ranking day less than day_num, set to None
            lower_neighbor = None
        try:
            upper_neighbor = df[df["RankingDayNum"] > day_num]["RankingDayNum"].idxmin()
        except ValueError:
            # if there is no ranking day greater than day_num, set to None
            upper_neighbor = None

        return lower_neighbor if lower_neighbor else upper_neighbor


def check_or_add_rank_columns():
    """Add the matchup ordinals to the regular season results table."""

    conn = get_db_conn()
    cur = conn.cursor()

    alter_table_base = """
        ALTER TABLE regular_season_detailed_results_men
        ADD COLUMN {} INTEGER;
    """
    columns = ["WRANK", "LRANK"]
    LOGGER.info(f"Adding {columns} if not exists")
    for column in columns:
        try:
            cur.execute(alter_table_base.format(column))
        except OperationalError:
            LOGGER.info(f"Column {column} already exists.")
    conn.close()


# Provide the option to run this function only if these tables are missing
if __name__ == "__main__":
    # build_team_aggregates(sport="men")
    create_massey_ordinal_mapping()
