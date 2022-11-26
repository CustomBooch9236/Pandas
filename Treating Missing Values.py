# Importing the Necessary Libraries and Creating a DataFrame
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

met_df = pd.read_csv('https://student-datasets-bucket.s3.ap-south-1.amazonaws.com/whitehat-ds-datasets/meteorite-landings/meteorite-landings.csv')

# Slicing A DataFrame and the Ampersand (&) Logical Operator
correct_years_df = met_df[(met_df['year'] >= 860) & (met_df['year'] <= 2016)]
# Removing the Invalid 'reclong' Values
correct_long_df = correct_years_df[(correct_years_df['reclong'] >= -180) & (correct_years_df['reclong'] <= 180)]
# Removing the Rows Containing '0 N', '0 E' Values
correct_lat_long_df = correct_long_df[~((correct_long_df['reclat'] == 0 ) & (correct_long_df['reclong'] == 0))]

# Treating Missing Values
#1 Check the Missing Values
print(correct_lat_long_df.isnull().sum())
#2 Retrieving All the Rows Containing the Missing 'mass' Values in the 'correct_lat_long_df' DataFrame
print(correct_lat_long_df[correct_lat_long_df['mass'].isnull() == True])
#3 Getting Descriptive Statistics for the 'mass' Column in the 'correct_lat_long_df' DataFrame
print(correct_lat_long_df['mass'].describe())

#4 Creating a List of the Indices of the Above Rows
row_indices = correct_lat_long_df[correct_lat_long_df['mass'].isnull()== True].index
#5 Retrieving the Missing 'mass' Values from 'correct_lat_long_df' DataFrame Using the 'loc[]' Function
missing_mass_values = correct_lat_long_df.loc[row_indices,'mass']
#6 Replacing the Missing Values in the 'mass' Column in the 'correct_lat_long_df' DataFrame With Median of Mass
median_mass = correct_lat_long_df['mass'].median()
#7 Checking Whether All the Missing 'mass' Values have been Replaced by the Median of the 'mass' Values or Not
print(correct_lat_long_df.loc[row_indices,:])
