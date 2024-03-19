import json
import logging
import sys

from utils import data_prep, data_transform, logistic_regression as lr

LOGGER = logging.getLogger()
logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] | %(levelname)s | %(filename)s | %(funcName)s:%(lineno)d | PID: %(process)d | %(message)s',
                    datefmt='%H:%M:%S'
                    )


if __name__ == "__main__":

    with open("run_config.json", "r") as config_file:
        configs = json.load(config_file)

    if configs.get("ranking_system") and configs["sport"] == "women":
        raise ValueError(
            f"sport is women and ranking_system is a non-null value - " +
            "ranking systems are not currently supported for this sport"
        )

    LOGGER.info(f"Loaded configs: {configs}")

    if configs["run_data_setup"]:
        # SETUP - only needs to be run once to create the database
        # data_prep.create_database(configs["add_external_sources"])
        #
        # # Add advanced stats to DB
        # data_prep.create_advanced_statistics()
        #
        # # Standardize team names for external data sources
        # if configs["add_external_sources"]:
        #     data_prep.standardize_team_names()
        #
        # if configs["aggregate_team_data"]:
        #     data_prep.build_team_aggregates(sport=configs["sport"])
        #
        # # Add an index to the massey ordinal table
        if configs["sport"] == "men":
            data_prep.create_massey_ordinal_mapping(ranking_system=configs["ranking_system"])
            # data_prep.update_to_latest_ranking(sport="men")

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

        if configs.get("build_predictions"):
            lr.build_predictions(configs["execution_name"], sport=configs["sport"])
        else:
            LOGGER.info("build_predictions set to false - model will stop at testing")
            sys.exit(0)


    if configs["export_content"]:
        lr.export_to_csv(configs["execution_name"], sport=configs["sport"])
        lr.build_bracketeer_bracket(configs["execution_name"], sport=configs["sport"])

    LOGGER.info("Done.")
