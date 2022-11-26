import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

exo_train_df = pd.read_csv('https://student-datasets-bucket.s3.ap-south-1.amazonaws.com/whitehat-ds-datasets/kepler-exoplanets-dataset/exoTrain.csv')
exo_test_df = pd.read_csv('https://student-datasets-bucket.s3.ap-south-1.amazonaws.com/whitehat-ds-datasets/kepler-exoplanets-dataset/exoTest.csv')

# Carried over from Fast Fourier Transformation

def mean_normalise(series):
  norm_series = (series - series.mean()) / (series.max() - series.min())
  return norm_series

norm_train_df = exo_train_df.iloc[:, 1:].apply(mean_normalise, axis=1)
norm_train_df.insert(loc=0, column='LABEL', value=exo_train_df['LABEL'])
norm_test_df = exo_test_df.iloc[:, 1:].apply(mean_normalise, axis=1)
norm_test_df.insert(loc=0, column='LABEL', value=exo_test_df['LABEL'])
exo_train_df.T

def fast_fourier_transform(star):
  fft_star = np.fft.fft(star, n=len(star))
  return np.abs(fft_star)
freq = np.fft.fftfreq(len(exo_train_df.iloc[0, 1:]))
x_fft_train_T = norm_train_df.iloc[:, 1:].T.apply(fast_fourier_transform)
x_fft_train = x_fft_train_T.T
x_fft_test_T = norm_test_df.iloc[:, 1:].T.apply(fast_fourier_transform)
x_fft_test = x_fft_test_T.T

# Oversampling For Classification Problems
#1 Getting the 'y_train' and 'y_test' Series from the 'norm_train_df' and 'norm_test_df' DataFrames Respectively
y_train = norm_train_df.iloc[:,0]
y_test = norm_test_df['LABEL']
#2 Apply the 'SMOTE()' Function to Balance the Training Data
sm = SMOTE(ratio=1)
x_fft_train_res , y_fft_train_res = sm.fit_sample(x_fft_train,y_train)

# The sum() Function
#1 Finding the Number of Occurrences of class '1' and class '2' Values in the 'y_fft_train_res' NumPy Array
print(sum(y_fft_train_res == 1))
print(sum(y_fft_train_res == 2))

# Applying the RandomForestClassifier Model
rf_cls = RandomForestClassifier(n_jobs=-1, n_estimators=50)
rf_cls.fit(x_fft_train_res, y_fft_train_res)
y_pred = rf_cls.predict(x_fft_test)

# The Confusion Matrix & Classification Report
cm = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))

# The XGBoost Classifier Model
#1 Deploying the XGBoost Classifier Model
import xgboost as xg
#2 Calling the 'XGBClassifier()' Function and Storing it in the 'model' Variable
model = xg.XGBClassifier()
#3 Calling the 'fit()' function with the 'x_fft_train_res' and 'y_fft_train_res' NumPy Arrays as Input
model.fit(x_fft_train_res,y_fft_train_res)
#4 Making Predictions on Test Data by Calling the 'predict()' Function with 'x_fft_test' Data as Input
y2_pred = model.predict(np.array(x_fft_test))
#5 Predicting the values of predicted values
print(y2_pred)
#6 Creating the Confusion Matrix Using the 'y_test' and 'y2_pred' Values as Inputs
cm1= confusion_matrix(y_test, y2_pred)
print(cm1)
#7 Printing the Classification Report Using the 'y_test' and 'y2_pred' Values as Inputs
print(classification_report(y_test, y2_pred))
