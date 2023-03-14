import logging

from utils import data_prep


LOGGER = logging.getLogger()
logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] | %(levelname)s | %(filename)s | %(funcName)s:%(lineno)d | %(message)s',
                    datefmt='%H:%M:%S'
                    )

add_external_sources = True

# SETUP - only needs to be run once to create the database
data_prep.create_database(add_external_sources)

# Add advanced stats to DB
data_prep.create_advanced_statistics()

# Standardize team names for external data sources
data_prep.standardize_team_names()

LOGGER.info("Done.")
