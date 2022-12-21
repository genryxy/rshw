import logging
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split

from config.conf import settings


def load_model(filename: str):
    path = settings.MODEL.model_path
    logging.info(f'Upload model by path `{path}`')
    model = pickle.load(open(f'{path}{filename}', 'rb'))
    return model


def save_model(clf, filename: str):
    path = settings.MODEL.model_path
    logging.info(f'Save model by path `{path}` into `{filename}`')
    pickle.dump(clf, open(f'{path}{filename}', 'wb'))


def split_data(df: pd.DataFrame):
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