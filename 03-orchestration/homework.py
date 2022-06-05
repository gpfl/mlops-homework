import pickle
from datetime import datetime
from urllib import request

import pandas as pd
from prefect import flow, get_run_logger, task
from prefect.task_runners import SequentialTaskRunner
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def get_file(year: str, month: str) -> str:

    url = f"https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_{year}-{month}.parquet"
    file_path = f"./data/fhv_tripdata_{year}-{month}.parquet"

    request.urlretrieve(url, filename=file_path)

    return file_path


@task
def get_paths(date: str):

    date_converted = datetime.strptime(date, "%Y-%m-%d")

    year = f"{date_converted.year}"
    month_train = f"{date_converted.month - 2}".zfill(2)
    month_valid = f"{date_converted.month - 1}".zfill(2)

    train_path = get_file(year, month_train)
    valid_path = get_file(year, month_valid)

    return train_path, valid_path


@task
def read_data(path):
    df = pd.read_parquet(path)
    return df


@task
def prepare_features(df, categorical, train=True):
    logger = get_run_logger()
    df["duration"] = df.dropOff_datetime - df.pickup_datetime
    df["duration"] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        logger.info(f"The mean duration of training is {mean_duration}")
    else:
        logger.info(f"The mean duration of validation is {mean_duration}")

    df[categorical] = df[categorical].fillna(-1).astype("int").astype("str")
    return df


@task
def train_model(df, categorical):
    logger = get_run_logger()
    train_dicts = df[categorical].to_dict(orient="records")
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)
    y_train = df.duration.values

    logger.info(f"The shape of X_train is {X_train.shape}")
    logger.info(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    logger.info(f"The MSE of training is: {mse}")

    return lr, dv


@task
def run_model(df, categorical, dv, lr):
    logger = get_run_logger()
    val_dicts = df[categorical].to_dict(orient="records")
    X_val = dv.transform(val_dicts)
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    logger.info(f"The MSE of validation is: {mse}")
    return


@flow(task_runner=SequentialTaskRunner())
def main(
    date: str,
):

    categorical = ["PUlocationID", "DOlocationID"]

    train_path, val_path = get_paths(date).result()

    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical)

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical, False)

    # train the model
    lr, dv = train_model(df_train_processed, categorical).result()
    run_model(df_val_processed, categorical, dv, lr)

    # save DictVetorizer()
    with open(f"models/dv-{date}.b", "wb") as f_out:
        pickle.dump(dv, f_out)

    # save model
    with open(f"models/model-{date}.bin", "wb") as f_out:
        pickle.dump(lr, f_out)


main(date="2021-08-15")
