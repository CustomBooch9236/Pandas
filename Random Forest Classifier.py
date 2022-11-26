import panads as pd

# Collecting Data
import pandas as pd
exo_train_df = pd.read_csv('https://student-datasets-bucket.s3.ap-south-1.amazonaws.com/whitehat-ds-datasets/kepler-exoplanets-dataset/exoTrain.csv')
exo_test_df = pd.read_csv('https://student-datasets-bucket.s3.ap-south-1.amazonaws.com/whitehat-ds-datasets/kepler-exoplanets-dataset/exoTest.csv')

# Importing the 'RandomForestClassifier' Module from the 'sklearn.ensemble' Library
from sklearn.ensemble import RandomForestClassifier

# Extracting the Feature Variables from the Training Dataset
x_train = exo_train_df.iloc[:,1:]
y_train = exo_train_df.iloc[:,0]

# Fitting the Model
#1 Calling the 'RandomForestClassifier' Module with Inputs as 'n_jobs = - 1' & 'n_estimators=50'
rf_clf = RandomForestClassifier(n_jobs=-1, n_estimators=50)
#2 Calling the 'fit()' Function with 'x_train' and 'y_train' as Inputs\
print(rf_clf.fit(x_train,y_train))
#3 Calling the 'score()' Function with 'x_train' and 'y_train' as Inputs to Check the Accuracy Score of the Model
print(rf_clf.score(x_train,y_train))

# The predict() Function
x_test = exo_test_df.iloc[:,1:] #extracting the feature variables
y_test = exo_test_df.iloc[:,0] #extracting the target variable
y_predicted = rf_clf.predict(x_test)

# The Confusion Matrix
from sklearn.metrics import confusion_matrix , classification_report
cm = confusion_matrix(y_test,y_predicted)
#1 Printing the 'precision', 'recall' and 'f1-score' Values Using the 'classification_report()' Function
class_rep = classification_report(y_test,y_predicted)
print(class_rep)
