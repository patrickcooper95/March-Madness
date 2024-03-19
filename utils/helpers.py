import json
import multiprocessing
from logging import getLogger


LOGGER = getLogger()


def run_workers(target_function, args_list):
    results = []
    with multiprocessing.Pool() as pool:
        for result in pool.starmap(target_function, args_list):
            results.append(result)
    return results


def create_chunks(data_set, chunk_size: int = 1000):
    """Divide the games into specified chunk size for multiprocessing."""
    divided_lists = []
    current_list = []

    for row in data_set:
        current_list.append(row)
        if len(current_list) >= chunk_size:
            divided_lists.append(current_list)
            current_list = []

    if current_list:  # If there are remaining items
        divided_lists.append(current_list)

    LOGGER.info(f"Created {len(divided_lists)} chunks with max chunk size of {chunk_size}")
    return divided_lists


def matchup_ordinal_worker(games, rankings_df, ranking_system: str = "POM"):

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

        winning_team_ranking = rankings_df.iloc[[winning_team_index]]["OrdinalRank"].values[0]
        losing_team_ranking = rankings_df.iloc[[losing_team_index]]["OrdinalRank"].values[0]

        LOGGER.info(f"Adding matchup {matchup_row_id} ranking mapping")
        matchup_ordinals.append(
            (
                winning_team_ranking,
                losing_team_ranking,
                matchup_row_id,
            )
        )
    return matchup_ordinals


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
