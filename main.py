import json
import logging

from utils import data_prep, data_transform, logistic_regression as lr

LOGGER = logging.getLogger()
logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] | %(levelname)s | %(filename)s | %(funcName)s:%(lineno)d | %(message)s',
                    datefmt='%H:%M:%S'
                    )


with open("run_config.json", "r") as config_file:
    configs = json.load(config_file)

LOGGER.info(f"Loaded configs: {configs}")

if configs["run_data_setup"]:
    # SETUP - only needs to be run once to create the database
    data_prep.create_database(configs["add_external_sources"])

    # Add advanced stats to DB
    data_prep.create_advanced_statistics()

    # Standardize team names for external data sources
    if configs["add_external_sources"]:
        data_prep.standardize_team_names()

    if configs["aggregate_team_data"]:
        data_prep.build_team_aggregates(sport=configs["sport"])

if configs["transform_and_test"]:
    # Build regression training data
    data_transform.get_training_data(sport=configs["sport"])
    data_transform.get_test_data(sport=configs["sport"])

    # Classify training data
    training_data = lr.classify_data(training=True, sport=configs["sport"])
    # And test data
    test_data = lr.classify_data(training=False, sport=configs["sport"])
    lr.train_model(training_data)
    lr.test_model(test_data)
    lr.build_predictions(configs["execution_name"], sport=configs["sport"])


if configs["export_content"]:
    lr.export_to_csv(configs["execution_name"], sport=configs["sport"])
    lr.build_bracketeer_bracket(configs["execution_name"], sport=configs["sport"])

LOGGER.info("Done.")
