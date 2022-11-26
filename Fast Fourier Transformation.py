import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


exo_train_df = pd.read_csv('https://student-datasets-bucket.s3.ap-south-1.amazonaws.com/whitehat-ds-datasets/kepler-exoplanets-dataset/exoTrain.csv')
exo_test_df = pd.read_csv('https://student-datasets-bucket.s3.ap-south-1.amazonaws.com/whitehat-ds-datasets/kepler-exoplanets-dataset/exoTest.csv')


norm_train_df = exo_train_df.iloc[:, 1:].apply(mean_normalise, axis=1)
norm_train_df.insert(loc=0, column='LABEL', value=exo_train_df['LABEL'])
norm_test_df = exo_test_df.iloc[:, 1:].apply(mean_normalise, axis=1)
norm_test_df.insert(loc=0, column='LABEL', value=exo_test_df['LABEL'])
exo_train_df.T #Trasnsposing exo_train DataFrame

# Creating a line plot for the first star in the 'norm_train_df' DataFrame
star_0 = norm_train_df.iloc[0,:]
x_axis = np.arange(1,3198)
y_axis = star_0.iloc[1:]

plt.figure(figsize=(16,5))
plt.plot(x_axis,y_axis)
plt.grid(True)
plt.show()

# Making a Line Plot Between the 'x' and 'y' Values
N = 600 # Number of sample points
T = 1.0 / 800.0 # sample spacing

t = np.linspace(0.0, N*T, N)
y = np.sin(50.0 * 2.0 * np.pi * t) + 0.5 * np.sin(75.0 * 2.0 * np.pi * t)

plt.figure(figsize=(20, 5))
plt.plot(t, y)
plt.grid()
plt.show()

# Making a Line Plot Between the 'xf' and 'yf' Values
yf = np.fft.fft(y)
tf = np.linspace(0.0, 1.0/(2.0*T), N//2)

plt.figure(figsize=(20, 5))
plt.plot(tf, 2.0/N * np.abs(yf[0:N//2]))
plt.grid() 
plt.show()

# Applying the 'fft.fft()' Function on the 'star_0' Pandas Series
fft_star_0 = np.abs(np.fft.fft(star_0.iloc[1:]))

# Applying the 'fft.fftfreq()' Function on 'len(star_0)' to Get the Frequencies of the 'star_0' Pandas Series
frequency = np.fft.fftfreq(len(star_0.iloc[1:]))

#Creating a Line Plot Between the 'fft_star_0' and 'freq' Values\
plt.figure(figsize=(15,6))
plt.plot(frequency,fft_star_0)
plt.show()

# Creating a Function and Naming it 'fast_fourier_transformation()' to apply Fast Fourier Transformation on the DataFrames
import numpy as np
def fast_fourier_transformation(star):
  fft_star = np.fft.fft(star,n=len(star))
  return np.abs(fft_star)

# Applying the 'fast_fourier_transform' function on the transposed 'norm_train_df' DataFrame
x_fft_train_T = norm_train_df.iloc[:,1:].T.apply(fast_fourier_transformation,axis =0)
x_fft_train = x_fft_train_T.T
