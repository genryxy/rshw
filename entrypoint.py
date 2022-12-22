import argparse
import logging
import numpy as np

from config.conf import settings
from data.data import get_train_test_data
from model.random_forest_clf import fit_random_forest
from model.sgd_clf import fit_sgd_clf
from util.utils import load_model

# python entrypoint.py --prediction_model rf --prediction_params 55,0,0,180,327,0,2,117,1,3.4,1,0,2
parser = argparse.ArgumentParser(description='CLI for using classification models.')
parser.add_argument('--prediction_model', dest='model', metavar='m', type=str, nargs=1,
                    help='required model for classification (rf/sgd)')
parser.add_argument('--prediction_params', dest='params', metavar='p', type=str, nargs=1,
                    help='input parameters for the model')
args = parser.parse_args()

model = args.model[0]
params = np.reshape(list(map(float, args.params[0].split(','))), (1, -1))
if model == 'rf':
    X_train, X_test, y_train, y_test = get_train_test_data()
    fit_random_forest(X_train, y_train)
    clf = load_model(settings.MODEL.rf_clf_pkl)
    logging.info(f'Accuracy RF is {clf.score(X_test, y_test)}')
    logging.info(f'Prediction by params: {clf.predict(params)[0]}')
elif model == 'sgd':
    X_train, X_test, y_train, y_test = get_train_test_data()
    fit_sgd_clf(X_train, y_train)
    clf = load_model(settings.MODEL.sgd_clf_pkl)
    logging.info(f'Accuracy SGD is {clf.score(X_test, y_test)}')
    logging.info(f'Prediction by params: {clf.predict(params)[0]}')
else:
    logging.error(f'Unknown model `{model}`. Use `clf` or `sgd`')
