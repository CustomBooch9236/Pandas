#A Pandas series is a one-dimensional array which can hold various data types. It is similar to a Python list and a NumPy array.

import pandas as pd

#Python List To Pandas Series Conversion
import pandas as pd
import random
weights = pd.Series([random.randint(45,60) for i in range(30)])

#The mean(), min(), max(), mode(), median() Functions
w_mean = weights.mean()
w_min = weights.min()
w_max = weights.max()
w_mode = weights.mode()
w_median = weights.median()

#The sort_values() Function - arrange the numbers in a Pandas series either in an ascending order or in descending order
weights.sort_values(ascending=True) #sorts the the weights in increasing order
