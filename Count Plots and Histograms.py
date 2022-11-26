import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

met_df = pd.read_csv('https://student-datasets-bucket.s3.ap-south-1.amazonaws.com/whitehat-ds-datasets/meteorite-landings/meteorite-landings.csv')

# Setting UP Dataset 
correct_years_df = met_df[(met_df['year'] >= 860) & (met_df['year'] <= 2016)
correct_long_df = correct_years_df[(correct_years_df['reclong'] >= -180) & (correct_years_df['reclong'] <= 180)]
correct_lat_long_df = correct_long_df[~((correct_long_df['reclat'] == 0 ) & (correct_long_df['reclong'] == 0))]
row_indices = correct_lat_long_df[correct_lat_long_df['mass'].isnull() == True].index
median_mass = correct_lat_long_df['mass'].median()
correct_lat_long_df.loc[:, 'year'] = correct_lat_long_df.loc[:, 'year'].astype('int')

# COUNT PLOT
#1 Creating a DataFrame Called 'met_after_1990_df' and Storing Data for the Meteorites Discovered After 1990                          
met_after_1990_df = correct_lat_long_df[correct_lat_long_df['year']>1990]
#2 Creating a Count Plot for the 'year' Values in the 'met_after_1990_df' DataFrame on the x-axis
plt.figure(figsize=(15,6))
sns.countplot(x='year', data=met_after_1990_df, hue = 'nametype')
plt.show()

# HISTOGRAM                          
#1 Creating a Pandas Series Containing the 'year' Values Between 1970 and 2000 Including Both of Them
correct_lat_long_df.loc[(correct_lat_long_df['year']>=1970) & (correct_lat_long_df['year']<=2000),'year']
#2 Creating a Histogram for the Pandas Series Containing the 'year' Values Between 1970 and 2000 Including Both of Them
plt.figure(figsize=(16,5))
plt.hist(correct_lat_long_df.loc[(correct_lat_long_df['year']>=1970) & (correct_lat_long_df['year']<=2000),'year'], 
         bins = 6)
plt.show()
