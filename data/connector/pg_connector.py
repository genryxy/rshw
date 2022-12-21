import logging

import pandas as pd


def get_data_by(link: str) -> pd.DataFrame:
    logging.info('Try to read csv-file')
    df = pd.read_csv(link)
    logging.info(f'Read csv-file size: `{len(df)}`')
    return df
