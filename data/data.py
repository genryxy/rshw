import logging
import pandas as pd

from config.conf import settings
from data.connector.pg_connector import get_data_by
from util.utils import split_data


def load_df() -> pd.DataFrame:
    """
    Load dataframe
    :return: Received dataframe.
    """
    link = settings.DATA.data_set
    logging.info(f'Extracting dataset from {link}')
    df = get_data_by(link)
    logging.info('Extracted dataset')
    return df


def get_train_test_data():
    """
    Get data for train and test
    :return: Splitted train and test data.
    """
    df = load_df()
    X_train, X_test, y_train, y_test = split_data(df)
    return X_train.values, X_test.values, y_train.values, y_test.values