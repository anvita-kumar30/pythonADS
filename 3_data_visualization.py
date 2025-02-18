import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("diamonds.csv")

# Strip spaces from column names
df.columns = df.columns.str.strip()

# Display some basic information about the dataset
# print(df.describe())
# print(df.info())
# print(df.corr())
# print(df.count())

# Scatter plot
plt.scatter(df['carat'], df['price'], c='blue')
plt.xlabel('Carat')
plt.ylabel('Price')
plt.title('Carat vs Price')
plt.show()

# Distribution plot
g = sns.displot(df['depth'])  # Distribution plot for depth
g.fig.suptitle('Distribution of Depth')
plt.show()

# Joint plot (Bivariate and Univariate)
sns.jointplot(x='carat', y='price', data=df)
plt.show()

# Pair plot (Pairwise relationships in the dataset)
sns.pairplot(df[['carat', 'price', 'depth', 'table', 'x']])
plt.show()

# Rug plot with scatter plot
plt.figure(figsize=(15, 5))
sns.scatterplot(data=df.head(20), x="carat", y="price")
sns.rugplot(data=df.head(20), x='carat')
plt.show()

# Histogram of 'price'
df['price'].hist()
plt.title('Histogram of Diamond Price')
plt.show()

# Andrews curve plot (Multivariate visualization)
df1 = df[['carat', 'depth', 'table', 'price']]
df1 = df1.sample(n=50)  # Sampling 50 random rows
andrew = pd.plotting.andrews_curves(df1, 'carat')
plt.title('Andrew\'s Curves')
plt.show()
