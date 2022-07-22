from datetime import datetime

import batch
import pandas as pd


def dt(hour, minute, second=0):
    return datetime(2021, 1, 1, hour, minute, second)


def test_prepare():
    data = [
        (None, None, dt(1, 2), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, 1, dt(1, 2, 0), dt(1, 2, 50)),
        (1, 1, dt(1, 2, 0), dt(2, 2, 1)),
    ]

    columns = ["PUlocationID", "DOlocationID", "pickup_datetime", "dropOff_datetime"]
    df = pd.DataFrame(data, columns=columns)

    expected_data = {
        "PUlocationID": {0: "-1", 1: "1"},
        "DOlocationID": {0: "-1", 1: "1"},
        "pickup_datetime": {
            0: pd.Timestamp("2021-01-01 01:02:00"),
            1: pd.Timestamp("2021-01-01 01:02:00"),
        },
        "dropOff_datetime": {
            0: pd.Timestamp("2021-01-01 01:10:00"),
            1: pd.Timestamp("2021-01-01 01:10:00"),
        },
        "duration": {0: 8.000000000000002, 1: 8.000000000000002},
    }

    expected_df = pd.DataFrame(expected_data)

    actual_df = batch.prepare_data(df, ["PUlocationID", "DOlocationID"])

    assert expected_df.to_dict() == actual_df.to_dict()
