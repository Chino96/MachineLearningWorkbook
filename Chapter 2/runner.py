from data_fetcher import load_housing_data
import matplotlib.pyplot as plt

housing = load_housing_data()
housing.hist(bins=50, figsize=(20,15))
plt.show()