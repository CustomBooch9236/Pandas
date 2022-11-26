import pandas as pd
# Importing the 'numpy' and 'matplotlib.pyplot' Modules
import numpy as np
import matplotlib.pyplot as plt

# Import the DF and check its dimensions
exo_train_df = pd.read_csv('https://student-datasets-bucket.s3.ap-south-1.amazonaws.com/whitehat-ds-datasets/kepler-exoplanets-dataset/exoTrain.csv')
exo_train_df.shape
# Slice the DF
star_0 = exo_train_df.iloc[0, :]

# Scatter Plot
#1 Call the 'figure()' function to resize the plot
plt.figure(figsize=(16,4))
#2 Create the inputs for the Scatter Plot
x_values_star0 = np.arange(1,3198)
y_values_star0 = star_0[1:]
#3 Call the Scatter Function
plt.scatter(x_values_star0,y_values_star0)
plt.show()

# Line Plot
plt.figure(figsize=(18,6))
plt.plot(x_values_star0,y_values_star0)
plt.show()
