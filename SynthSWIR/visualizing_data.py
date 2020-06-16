"""Pair-plot visualization of training data using Seaborn."""
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# read the csv
df = pd.read_csv('training_data.csv')

# drop unnamed column
df.pop('Unnamed: 0')

sns.set_style("dark")
sns.pairplot(df, x_vars=['Blue', 'Green', 'Red', 'NIR'],
             y_vars=['SWIR1', 'SWIR2'],
             height=2.0)
plt.show()
