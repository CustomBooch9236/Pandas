import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

met_df = pd.read_csv('https://student-datasets-bucket.s3.ap-south-1.amazonaws.com/whitehat-ds-datasets/meteorite-landings/meteorite-landings.csv')

correct_years_df = met_df[(met_df['year'] >= 860) & (met_df['year'] <= 2016)]
correct_long_df = correct_years_df[(correct_years_df['reclong'] >= -180) & (correct_years_df['reclong'] <= 180)]
correct_lat_long_df = correct_long_df[~((correct_long_df['reclat'] == 0 ) & (correct_long_df['reclong'] == 0))]
row_indices = correct_lat_long_df[correct_lat_long_df['mass'].isnull() == True].index
median_mass = correct_lat_long_df['mass'].median()
correct_lat_long_df.loc[row_indices, 'mass'] = median_mass
correct_lat_long_df.loc[:, 'year'] = correct_lat_long_df.loc[:, 'year'].astype('int')

# Getting list containing the bar attributes for the count plot displaying the number of meteorites fallen from 1970 to 2000.
met_1970_2000_df = correct_lat_long_df[(correct_lat_long_df['year']>1970) & (correct_lat_long_df['year']<2000)]
plt.figure(figsize=(16,5))
cp = sns.countplot(x='year', data =met_1970_2000_df )
for p in cp.patches:
  print(p)

# Getting height, width, x and y coordinates of each bar in the count plot stored in the 'cp' variable
plt.figure(figsize=(16,5))
cp = sns.countplot(x='year', data =met_1970_2000_df )
for p in cp.patches:
  print("\nWidth : ",p.get_width(), "\nHeight :",p.get_height(),
        "\nX-coordinate:",p.get_x(),
        "\nY-coordinate" ,p.get_y() )

# Annotate bars in the count plot for the meteorites fallen between the years 1970 and 2000 (including both)
plt.figure(figsize=(16,5))
cp=sns.countplot(x='year',data=met_1970_2000_df)
for p in cp.patches:
  cp.annotate(p.get_height(),xy = ( p.get_x() + p.get_width()/2 , p.get_height()), ha ='center', va ='bottom')

# Annotate a histogram created using the 'distplot()' function
plt.figure(figsize=(20,5))
dp = sns.distplot(correct_lat_long_df.loc[(correct_lat_long_df['year']>1970)&(correct_lat_long_df['year']<2001), 'year'], bins=6, kde= False)
for p in dp.patches:
  dp.annotate(p.get_height(), xy = (p.get_x()+p.get_width()/2, p.get_height()), ha = 'center', va = 'center')
plt.show()
