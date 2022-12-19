import pandas as pd


def get_data(link: str) -> pd.DataFrame:
    df = pd.read_csv(link)
    return df