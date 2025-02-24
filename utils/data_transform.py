import csv
import json
import logging
import os
import sqlite3 as sql

import pandas as pd


LOGGER = logging.getLogger()


def get_db_conn() -> sql.Connection:
    return sql.connect("madness.db")


def create_years_list(
        start_year: int,
        training: bool,
        offset: bool = False,
        alternate: bool = False) -> str:
    """Create a list of years specific to training/test period."""

    years = [start_year]

    if alternate:
        for i in range(10 if not offset else 9):
            years.append(start_year + 2 * (i + 1))
    else:
        if training:
            for i in range(15):
                years.append(start_year + 1 * (i + 1))
        else:
            years = [2019, 2020, 2021, 2022]

    return ", ".join([f"'{str(year)}'" for year in years])


def create_transformation_table(sport: str, date_range: str, training: bool = True):
    """Create transformation/analysis table for regression model."""

    conn = get_db_conn()
    cur = conn.cursor()

    table_name = f"{sport.upper()}_TRAINING_SET" if training else f"{sport}_TEST_SET"
    # Create table
    cur.execute(f"DROP TABLE IF EXISTS {table_name};")
    new_fields = """WFGM-LFGM as NFGM,
                        WFGA-LFGA as NFGA,
                        WFGM3-LFGM3 as NFGM3,
                        WFTM-WFTA as NFTA,
                        WOR-LOR as NOR,
                        WDR-LDR as NDR,
                        WAst-LAst as NAST,
                        WTO-LTO as NTO,
                        WStl-LStl as NSTL,
                        WBlk-LBlk as NBLK,
                        WTSP-LTSP as NTSP,
                        WFGP-LFGP as NFGP,
                        W3PTP-L3PTP as N3PTP,
                        WFTP-LFTP as NFTP,
                        WTOVP-LTOVP as NTOVP
                        """
    create_statement = f"""CREATE TABLE {table_name} AS
                            SELECT {new_fields} FROM regular_season_detailed_results_{sport}
                            WHERE Season in ({date_range})
                            UNION
                            SELECT {new_fields} FROM tourney_detailed_results_{sport}
                            WHERE Season in ({date_range});"""
    LOGGER.info(f"Running statement: {create_statement}")
    cur.execute(create_statement)
    LOGGER.info(f"Successfully created {table_name} table")
    conn.commit()
    conn.close()


def get_test_data(sport: str = "men"):
    """Create test data set for model."""

    date_range = create_years_list(2004, training=False, offset=True)
    LOGGER.info(f"Setting up test data for: {sport} and date_range: {date_range}")

    # Create table
    create_transformation_table(sport, date_range, training=False)


def get_training_data(sport: str = "men"):
    """Create training data set for model."""

    date_range = create_years_list(2003, training=True)
    LOGGER.info(f"Setting up training data for: {sport} and date_range: {date_range}")

    # Create table
    create_transformation_table(sport, date_range, training=True)
