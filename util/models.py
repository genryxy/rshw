import logging
import pickle

from config.conf import settings


def load_model(filename: str):
    """
    Load model from file.
    :param filename: Filename with model
    :return: Loaded model
    """
    path = settings.MODEL.model_path
    logging.info(f'Upload model by path `{path}`')
    model = pickle.load(open(f'{path}{filename}', 'rb'))
    return model


def save_model(clf, filename: str):
    """
    Save model into the file.
    :param clf: Model for saving
    :param filename: Target filename with model
    """
    path = settings.MODEL.model_path
    logging.info(f'Save model by path `{path}` into `{filename}`')
    pickle.dump(clf, open(f'{path}{filename}', 'wb'))
