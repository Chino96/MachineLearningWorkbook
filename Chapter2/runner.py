from data_proc import load_housing_data, split_train_test
from sklearn.model_selection import  train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LinearRegression
import numpy as np


rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
	def __init__(self, add_bedrooms_per_room=True):
		self.add_bedrooms_per_room = add_bedrooms_per_room

	def fit(self, X, y=None):
		return self

	def transform(self, X, y=None):
		rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
		population_per_household = X[:, population_ix] / X[:, households_ix]

		if self.add_bedrooms_per_room:
			bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
			return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
		else:
			return np.c_[X, rooms_per_household, population_per_household]


class DataFrameSelector(BaseEstimator, TransformerMixin):
	def __init__(self, attribute_names):
		self.attribute_names = attribute_names

	def fit(self, X, y=None):
		return self

	def transform(self, X):
		return X[self.attribute_names].values


housing = load_housing_data()
housing_with_id = housing.reset_index()

train, test = train_test_split(housing_with_id, test_size=0.2, random_state=42)

housing_copy = train.drop('median_house_value', axis=1)
housing_labels = train['median_house_value'].copy()

housing_num = housing_copy.drop('ocean_proximity', axis=1)


num_pipeline = Pipeline([
	('Dataframe_Selector', DataFrameSelector(list(housing_num))),
	('imputer', SimpleImputer(strategy='median')),
	('attribs_adder', CombinedAttributesAdder()),
	('std_scaler', StandardScaler()),
])

cat_pipeline = Pipeline([
	('selector', DataFrameSelector(['ocean_proximity'])),
	('cat_encoder', OneHotEncoder(sparse=False)),
])

full_pipeline = FeatureUnion(transformer_list=[
	('num_pipeline', num_pipeline),
	('cat_pipeline', cat_pipeline),
])

housing_prep = full_pipeline.fit_transform(housing_copy)

lin_reg = LinearRegression()
lin_reg.fit(housing_prep, housing_labels)

print(housing_prep)