from prefect.deployments import DeploymentSpec
from prefect.flow_runners import SubprocessFlowRunner
from prefect.orion.schemas.schedules import CronSchedule

DeploymentSpec(
    name="cron-schedule-deployment",
    flow_location="./homework.py",
    schedule=CronSchedule(cron="0 9 15 * *", timezone="America/New_York"),
    flow_runner=SubprocessFlowRunner(),
    tags=["ml"],
)
