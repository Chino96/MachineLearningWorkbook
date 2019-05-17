import os
import tarfile
import pandas as pd
import numpy as np
from zlib import crc32
from six.moves import urllib

DOWNLOAD_ROOT = 'https://raw.githubusercontent.com/ageron/handson-ml/master/'
HOUSING_PATH = os.path.join('datasets', 'housing')
HOUSING_URL = DOWNLOAD_ROOT + 'datasets/housing/housing.tgz'


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
	if not os.path.isdir(housing_path):
		os.mkdir(housing_path)
	tgz_path = os.path.join(housing_path, 'housing.tgz')
	urllib.request.urlretrieve(housing_url, tgz_path)
	housing_tgz = tarfile.open(tgz_path)
	housing_tgz.extractall(path=housing_path)
	housing_tgz.close()


def load_housing_data(housing_path=HOUSING_PATH):
	csv_path = os.path.join(housing_path, 'housing.csv')
	return pd.read_csv(csv_path)


def test_set_check(identifier, test_ratio):
	return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32


def split_train_test(data, test_ratio, id_column):
	ids = data[id_column]
	in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
	return data.loc[~in_test_set], data.loc[in_test_set]

