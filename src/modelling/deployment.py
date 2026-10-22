from main import train_model_workflow
from prefect.schedules import Cron

TRAINSET_PATH = "data/abalone.csv"
# Every day at 1 pm, Paris time
RETRAIN_SCHEDULE = Cron("0 13 * * *", timezone="Europe/Paris")

if __name__ == "__main__":
    train_model_workflow.serve(
        name="train-model",
        schedule=RETRAIN_SCHEDULE,
        parameters={"trainset_path": TRAINSET_PATH},
    )
