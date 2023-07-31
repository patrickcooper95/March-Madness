import pendulum
from airflow import DAG
from airflow.decorators import task


default_args = {
    'owner': 'Patrick'
}

with DAG(
    "bracket_predictor_dag",
    default_args=default_args,
    start_date=pendulum.now().add(days=-1),
    schedule_interval=None,
) as dag:

    @task(task_id="setup")
    def setup_func(ds=None, **kwargs):
        """Gather DAG Run Configs and branch accordingly."""
        message = kwargs["dag_run"].conf

        branches = []
        tasks = [
            "add_external_sources",
            "add_external_sources",
            "run_data_setup",
            "aggregate_team_data",
            "transform_and_test",
            "export_content"
        ]
        for task in tasks:
            if message[task]:
                branches.append(task)
        return branches

    setup_task = setup_func()


    @task(task_id="print_the_context_two")
    def print_context_two(ds=None, **kwargs):
        """Print the Airflow context and ds variable from the context."""
        print(kwargs["dag_run"].conf)
        print(ds)
        return "Whatever you return gets printed in the logs"

    run_this_two = print_context_two()

    setup_task >> run_this_two
