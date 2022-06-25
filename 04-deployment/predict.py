import pickle

import pandas as pd
from flask import Flask, jsonify, request
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer

CATEGORICAL: list[str] = ["PUlocationID", "DOlocationID"]


def get_paths(year: int, month: int) -> tuple[str, str]:
    """Get the input and output path of the data based on inputs year and month.

    Parameters
    ----------
    year : int
        Desired year of the input data.
    month : int
        Desired month of the input data.

    Returns
    -------
    tuple[str, str]
        Strings for input file and posterior output file.
    """
    input_file = f"https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_{year:4d}-{month:02d}.parquet"
    output_file = f"./output/ride_predictions_{year:04d}_{month:02d}.parquet"

    return input_file, output_file


def get_model_dv(filename: str) -> tuple[DictVectorizer, RandomForestRegressor]:
    """Given a .bin file, returns a fitted vetorizer and trained model.

    Parameters
    ----------
    filename : str
        Path for the .bin file.

    Returns
    -------
    tuple[DictVectorizer, RandomForestRegressor]
        A vetorizer for the data and a trained RandomForest model.
    """
    with open(filename, "rb") as f_in:
        dv, lr = pickle.load(f_in)
    return dv, lr


def read_data(filename: str, year: int, month: int) -> pd.DataFrame:
    """Reads the data from filename URL and returns a pandas DataFrame.

    Parameters
    ----------
    filename : str
        URL of the input filename.

    year : int
        Desired year.

    month : int
        Desired month.

    Returns
    -------
    pd.DataFrame
        DataFrame of filename.
    """
    df = pd.read_parquet(filename)

    df["duration"] = df.dropOff_datetime - df.pickup_datetime
    df["duration"] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[CATEGORICAL] = df[CATEGORICAL].fillna(-1).astype("int").astype("str")

    df["ride_id"] = f"{year:04d}/{month:02d}_" + df.index.astype("str")

    return df


def transform_data(df: pd.DataFrame, dv: DictVectorizer) -> dict:
    """Transforms the data used for prediction.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame of the entries to be predicted.
    dv : DictVectorizer
        Vetorizer for the input data.

    Returns
    -------
    dict
        Dictionary of data to be predicted by the model.
    """
    dicts = df[CATEGORICAL].to_dict(orient="records")
    X_val = dv.transform(dicts)

    return X_val


app = Flask("duration-prediction")


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    ride_dict = request.get_json()

    year = ride_dict["year"]
    month = ride_dict["month"]

    input_file, _ = get_paths(year, month)

    df = read_data(input_file, year, month)

    dv, lr = get_model_dv("model.bin")

    X_val = transform_data(df, dv)
    y_pred = lr.predict(X_val)

    mean_pred = float(y_pred.mean())

    result = {"mean duration": mean_pred}

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
