import logging

from sklearn.linear_model import SGDClassifier

from config.conf import settings
from util.utils import save_model


def fit_sgd_clf(X_train: list, y_train: list):
    alpha_val = settings.SGD.alpha
    logging.info('Create SGDClassifier model')
    logging.info(f'Params: alpha={alpha_val}')
    clf = SGDClassifier(alpha=alpha_val)
    logging.info('Train SGDClassifier model')
    clf.fit(X_train, y_train)
    save_model(clf, settings.MODEL.sgd_clf_pkl)
    return clf