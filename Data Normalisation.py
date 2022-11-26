import numpy as np
import pandas as pd

exo_train_df = pd.read_csv('https://student-datasets-bucket.s3.ap-south-1.amazonaws.com/whitehat-ds-datasets/kepler-exoplanets-dataset/exoTrain.csv')
exo_test_df = pd.read_csv('https://student-datasets-bucket.s3.ap-south-1.amazonaws.com/whitehat-ds-datasets/kepler-exoplanets-dataset/exoTest.csv')

# Getting a Data Description by Calling the 'describe()' Function
exo_train_df.describe()

# Normalising a Pandas series Using the Mean Normalisation Method
def normalisation(series):
  avg = series.mean()
  min_value = series.min()
  max_value = series.max()
  series1 = (series-avg)/(max_value-min_value)
  return series1

# Applying the 'mean_normalise' Function Horizontally on the Training DataFrame
norm_train_df =exo_train_df.iloc[:,1:].apply(normalisation,axis = 1)
