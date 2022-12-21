from sklearn.ensemble import RandomForestClassifier

from config.conf import logging, settings
from util.utils import save_model


def fit_random_forest(X_train: list, y_train: list):
    n_estimators_val = settings.RANDOM_FOREST.n_estimators
    max_depth_val = settings.RANDOM_FOREST.max_depth
    rnd_state = settings.COMMON.random_state
    logging.info('Create RandomForestClassifier model')
    logging.info(f'Params: n_estimators={n_estimators_val}, max_depth={max_depth_val}, random_state={rnd_state}')
    clf = RandomForestClassifier(n_estimators=n_estimators_val, max_depth=max_depth_val, random_state=rnd_state)
    logging.info('Train RandomForestClassifier model')
    clf.fit(X_train, y_train)
    save_model(clf, settings.MODEL.rf_clf_pkl)
    return clf
