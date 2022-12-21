import logging

from config.conf import settings
from data.data import get_data
from model.sgd_clf import fit_sgd_clf
from model.random_forest_clf import fit_random_forest
from util.utils import split_data, load_model

if __name__ == '__main__':
    df = get_data()
    X_train, X_test, y_train, y_test = split_data(df)
    fit_random_forest(X_train, y_train)
    clf = load_model(settings.MODEL.rf_clf_pkl)
    logging.info(f'Accuracy RF is {clf.score(X_test, y_test)}')
    # logging.info(f'Prediction is {clf.predict(X_test)}')
    fit_sgd_clf(X_train, y_train)
    clf = load_model(settings.MODEL.sgd_clf_pkl)
    logging.info(f'Accuracy SGD is {clf.score(X_test, y_test)}')
