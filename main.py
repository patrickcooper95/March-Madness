import logging

from utils import data_prep, data_transform, logistic_regression as lr

LOGGER = logging.getLogger()
logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] | %(levelname)s | %(filename)s | %(funcName)s:%(lineno)d | %(message)s',
                    datefmt='%H:%M:%S'
                    )

add_external_sources = True
run_data_setup = False

if run_data_setup:
    # SETUP - only needs to be run once to create the database
    data_prep.create_database(add_external_sources)

    # Add advanced stats to DB
    data_prep.create_advanced_statistics()

    # Standardize team names for external data sources
    data_prep.standardize_team_names()

# Build regression training data
data_transform.get_training_data(women=False)
data_transform.get_test_data(women=False)

# Classify training data
training_data = lr.classify_data(training=True)
# And test data
test_data = lr.classify_data(training=False)
lr.train_model(training_data)
lr.test_model(test_data)

LOGGER.info("Done.")
