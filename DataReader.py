import pandas as pd


def csv_reader(data_path):
    df = pd.read_csv(data_path)
    return df
