import logging

from sklearn.linear_model import SGDClassifier

from config.conf import settings
from util.models import save_model


def fit_sgd_clf(X_train: list, y_train: list):
    """"
    Fit and save the model of SGDClassifier.
    :return: Fitted model
    """
    alpha_val = settings.SGD.alpha
    logging.info('Create SGDClassifier model')
    logging.info(f'Params: alpha={alpha_val}')
    clf = SGDClassifier(alpha=alpha_val)
    logging.info('Train SGDClassifier model')
    clf.fit(X_train, y_train)
    save_model(clf, settings.MODEL.sgd_clf_pkl)
    return clf