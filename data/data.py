import logging
import pandas as pd

from config.conf import settings
from data.connector.pg_connector import get_data_by


def get_data() -> pd.DataFrame:
    link = settings.DATA.data_set
    logging.info(f'Extracting dataset from {link}')
    df = get_data_by(link)
    logging.info('Extracted dataset')
    return df
