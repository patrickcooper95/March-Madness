import pendulum
from airflow import DAG


default_args = {
    'owner': 'Patrick'
}

dag = DAG(
    "bracket_predictor_dag",
    default_args=default_args,
    start_date=pendulum.now().add(days=-1),
    schedule_interval=None,
)
