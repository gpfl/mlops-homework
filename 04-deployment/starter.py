import pickle
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer

MODEL_FILE: str = "model.bin"
CATEGORICAL: list[str] = ["PUlocationID", "DOlocationID"]


def get_paths() -> tuple[str, str]:
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
    input_file = f"https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_{YEAR:4d}-{MONTH:02d}.parquet"
    output_file = f"./output/ride_predictions_{YEAR:04d}_{MONTH:02d}.parquet"

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


def read_data(filename: str) -> pd.DataFrame:
    """Reads the data from filename URL and returns a pandas DataFrame.

    Parameters
    ----------
    filename : str
        URL of the input filename.

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

    df["ride_id"] = f"{YEAR:04d}/{MONTH:02d}_" + df.index.astype("str")

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


def create_result_df(ids: pd.Series, y_pred: np.ndarray) -> pd.DataFrame:
    """Creates a pandas DataFrame with ID and respective predicted result.

    Parameters
    ----------
    ids : pd.Series
        Series of ID numbers.
    y_pred : np.ndarray
        Array of predictions.

    Returns
    -------
    pd.DataFrame
        DataFrame containing IDs and predictions.
    """
    result_dict = {"ride_id": ids, "ride_predictions": y_pred}

    return pd.DataFrame(result_dict)


def main():
    """Main function"""

    print("\nReading data for input.")
    df = read_data(INPUT_FILE)

    print("Loading model and vectorizer.")
    dv, lr = get_model_dv(MODEL_FILE)

    print("Transforming data to predict.")
    X_val = transform_data(df, dv)
    y_pred = lr.predict(X_val)

    print(f"\nMean predicted duration: {y_pred.mean()}\n")

    print(f"Saving file to parquet in path {OUTPUT_FILE}\n")
    df_result = create_result_df(df["ride_id"], y_pred)
    df_result.to_parquet(OUTPUT_FILE, engine="pyarrow", compression=None, index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--year", type=int)
    parser.add_argument("--month", type=int)

    my_args = parser.parse_args()

    YEAR = my_args.year
    MONTH = my_args.month
    INPUT_FILE, OUTPUT_FILE = get_paths()

    main()
