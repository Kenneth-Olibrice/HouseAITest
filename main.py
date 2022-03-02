from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib
from tensorflow import feature_column as fc
import tensorflow as tf

trainsrc = pd.read_csv("C:\\Users\\Kjenneth\\Downloads\\Sources\\train.csv")
testsrc = pd.read_csv("C:\\Users\\Kjenneth\\Downloads\\Sources\\test.csv")
train_label = trainsrc.pop("SalePrice")
eval_label = train_label.drop(labels=0, axis=0)
trainsrc = trainsrc.replace(np.nan, b'1')
NUMERICAL_COLUMNS = [
"Id",
"GarageYrBlt",
"MasVnrArea",
"MSSubClass",
"LotFrontage",
"LotArea",
"OverallQual",
"OverallCond",
"YearBuilt",
"YearRemodAdd",
"BsmtFinSF1",
"BsmtFinSF2",
"BsmtUnfSF",
"TotalBsmtSF",
"1stFlrSF",
"2ndFlrSF",
"LowQualFinSF",
"GrLivArea",
"BsmtFullBath",
"BsmtHalfBath",
"FullBath",
"HalfBath",
"BedroomAbvGr",
"KitchenAbvGr",
"TotRmsAbvGrd",
"Fireplaces",
"GarageCars",
"GarageArea",
"WoodDeckSF",
"OpenPorchSF",
"EnclosedPorch",
"3SsnPorch",
"ScreenPorch",
"PoolArea",
"MiscVal",
"MoSold",
"YrSold",]
CATEGORICAL_COLUMNS = [
"MSZoning",
"Street",
"Alley",
"LotShape",
"LandContour",
"Utilities",
"LotConfig",
"LandSlope",
"Neighborhood",
"Condition1",
"Condition2",
"BldgType",
"HouseStyle",
"RoofStyle",
"RoofMatl",
"Exterior1st",
"Exterior2nd",
"MasVnrType",
"ExterQual",
"ExterCond",
"Foundation",
"BsmtQual",
"BsmtCond",
"BsmtExposure",
"BsmtFinType1",
"BsmtFinType2",
"Heating",
"HeatingQC",
"CentralAir",
"Electrical",
"KitchenQual",
"Functional",
"FireplaceQu",
"GarageType",
"GarageFinish",
"GarageQual",
"GarageCond",
"PavedDrive",
"PoolQC",
"Fence",
"MiscFeature",
"SaleType",
"SaleCondition",]




feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
    vocab = trainsrc[feature_name].unique()
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocab))

for feature_name in NUMERICAL_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.dtypes.float32))


def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
  def input_function():  # inner function, this will be returned
    ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))  # create tf.data.Dataset object with data and its label
    if shuffle:
      ds = ds.shuffle(1000)  # randomize order of data
    ds = ds.batch(batch_size).repeat(num_epochs)  # split dataset into batches of 32 and repeat process for number of epochs
    return ds  # return a batch of the dataset
  return input_function  # return a function object for use


train_func = make_input_fn(trainsrc, train_label)
eval_func = make_input_fn(testsrc, eval_label, shuffle=False, num_epochs=1)

regressor = tf.estimator.LinearRegressor(feature_columns=feature_columns)
regressor.train(train_func)
pred_dicts = list(regressor.predict(eval_func))
for i in range(len(pred_dicts)):
    print("Prediction", str(i + 1) + ":", pred_dicts[i]['predictions'])