import pandas as pd

# Loading A CSV File
df = pd.read_csv('https://student-datasets-bucket.s3.ap-south-1.amazonaws.com/whitehat-ds-datasets/kepler-exoplanets-dataset/exoTrain.csv')

# Find Out the Number of Rows and Columns that Exist in a DataFrame
df.shape

# Check For The Missing Values
df.isnull()

# Find the Total Number of True Values in Each Column
df.isnull().sum()

# Check for Missing Values in the Entire DataFrame
num_missing_values = 0
for column in exo_train_df.columns:
  for item in exo_train_df[column].isnull():
    if item== True:
      num_missing_values +=1

# Replace 'True' with 'False' and see what is the Value of the 'num_missing_values' Variable
num_values = 0
for column in df.columns:
  for item in df[column].isnull():
    if item == False:
      num_values +=1 
num_values

# Slicing A DataFrame Using The iloc[] Function
star_0 = df.iloc[0,:]
