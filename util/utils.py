import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from config.conf import settings


def split_data(df: pd.DataFrame):
    """
    Split source dataframe into train and test parts.
    :param df: Input dataframe with data
    :return: Splitted data.
    """
    logging.info('Defining X and y')
    X = df.iloc[:, :-1]
    y = df['target']

    rnd_state = settings.COMMON.random_state
    logging.info(f'Split X and y with random state `{rnd_state}`')
    X_train, X_test, y_train, y_test = train_test_split(
        X,  # independent variables
        y,  # dependent variable
        random_state=rnd_state
    )
    return X_train, X_test, y_train, y_test