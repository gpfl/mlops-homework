import os
import pickle
import sys
from datetime import datetime

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import pandas as pd
from batch import prepare_data


def dt(hour, minute, second=0):
    return datetime(2021, 1, 1, hour, minute, second)


def input_test(df_input, options):

    input_file = "s3://nyc-duration/input_test.parquet"

    df_input.to_parquet(
        input_file,
        engine="pyarrow",
        compression=None,
        index=False,
        storage_options=options,
    )


def output_test(df_input, options):
    output_file = "s3://nyc-duration/output_test.parquet"

    with open("model.bin", "rb") as f_in:
        dv, lr = pickle.load(f_in)

    categorical = ["PUlocationID", "DOlocationID"]

    df = prepare_data(df_input, categorical)

    year = df["pickup_datetime"].dt.year.unique()[0]
    month = df["pickup_datetime"].dt.month.unique()[0]
    df["ride_id"] = f"{year:04d}/{month:02d}_" + df.index.astype("str")

    dicts = df[categorical].to_dict(orient="records")  # type: ignore
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    print("predicted mean duration:", y_pred.sum())

    df_result = pd.DataFrame()
    df_result["ride_id"] = df["ride_id"]
    df_result["predicted_duration"] = y_pred

    df_result.to_parquet(
        output_file,
        engine="pyarrow",
        compression=None,
        index=False,
        storage_options=options,
    )


if __name__ == "__main__":

    s3_endpoint = os.getenv("S3_ENDPOINT_URL", "http://localhost:4566")
    options = {"client_kwargs": {"endpoint_url": s3_endpoint}}

    raw_data = [
        (None, None, dt(1, 2), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, 1, dt(1, 2, 0), dt(1, 2, 50)),
        (1, 1, dt(1, 2, 0), dt(2, 2, 1)),
    ]

    columns = ["PUlocationID", "DOlocationID", "pickup_datetime", "dropOff_datetime"]
    df_data = pd.DataFrame(raw_data, columns=columns)

    input_test(df_data, options)
    output_test(df_data, options)
