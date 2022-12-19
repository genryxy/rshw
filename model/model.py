import pickle

from dynaconf import Dynaconf, settings
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from conf.conf import logging
from connector.pg_connector import get_data
from util.utils import load_model


def my_train_test_split(df):
    logging.info('Defining X and y')
    X = df.iloc[:, :-1]
    y = df['target']

    logging.info('Split X and y')
    X_train, X_test, y_train, y_test = train_test_split(X,  # independent variables
                                                        y,  # dependent variable
                                                        random_state=3
                                                        )
    return X_train, X_test, y_train, y_test


def fit_decision_tree(X_train, y_train):
    clf = DecisionTreeClassifier(max_depth=3, random_state=3)
    logging.info('Train model')
    clf.fit(X_train, y_train)
    pickle.dump(clf, open('conf/decision_tree.pkl', 'wb'))
    return clf


def get_df():
    myconf = Dynaconf(settings_file='setting.toml')
    print(settings.DATA)
    logging.info(f'Extracting dataset from {settings.DATA.data_set}')
    df = get_data(settings.data_set)
    logging.info('Extracted dataset')
    # print(df)
    return df


if __name__ == '__main__':
    df = get_df()
    X_train, X_test, y_train, y_test = my_train_test_split(df)
    fit_decision_tree(X_train, y_train)
    clf = load_model()
    logging.info(f'Accuracy is {clf.score(X_test, y_test)}')
    logging.info(f'Prediction is {clf.predict(X_test)}')

