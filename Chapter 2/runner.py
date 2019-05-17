from data_proc import load_housing_data, split_train_test
import matplotlib.pyplot as plt

housing = load_housing_data()
housing_with_id = housing.reset_index()
train, test = split_train_test(housing_with_id, .20, 'index')
print(len(test))
#housing.hist(bins=50, figsize=(20,15))
#plt.show()