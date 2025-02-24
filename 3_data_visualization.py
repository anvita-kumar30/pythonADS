# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import bokeh.plotting as bp
from bokeh.models import HoverTool
from bokeh.io import output_notebook, show
from pandas.plotting import scatter_matrix

# Load dataset
file_path = "diamonds.csv"
df = pd.read_csv(file_path)

# Strip spaces from column names
df.columns = df.columns.str.strip()

# Display some basic information about the dataset
print(df.info())
print(df.describe())

# Univariate Visualization
# 1. Histogram
plt.figure(figsize=(8,5))
plt.hist(df['price'], bins=30, color='skyblue', edgecolor='black')
plt.title('Histogram of Diamond Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()

# 2. Bar Chart
plt.figure(figsize=(8,5))
df['cut'].value_counts().plot(kind='bar', color='coral', edgecolor='black')
plt.title('Count of Diamonds by Cut')
plt.xlabel('Cut')
plt.ylabel('Count')
plt.show()

# 3. Quartile Plot (Boxplot)
plt.figure(figsize=(8,5))
sns.boxplot(x=df['carat'])
plt.title('Boxplot of Diamond Carat')
plt.show()

# 4. Distribution Chart (KDE plot)
plt.figure(figsize=(8,5))
sns.kdeplot(df['depth'], shade=True, color='green')
plt.title('Distribution of Diamond Depth')
plt.show()

# Multivariate Visualization
# 1. Scatterplot
plt.figure(figsize=(8,5))
sns.scatterplot(x=df['carat'], y=df['price'], alpha=0.5)
plt.title('Carat vs Price Scatterplot')
plt.xlabel('Carat')
plt.ylabel('Price')
plt.show()

# 2. Scatter Matrix
scatter_matrix(df[['carat', 'depth', 'table', 'price']], figsize=(8, 8), alpha=0.2)
plt.show()

# 3. Bubble Chart using Plotly
fig = px.scatter(df, x="carat", y="price", size="depth", color="cut", hover_name="cut", title="Bubble Chart: Carat vs Price with Depth Size")
# fig.show()
# Save the plot as an HTML file for PyCharm users
fig.write_html("bubble_chart.html")
print("Bubble Chart saved as 'bubble_chart.html'. Open it in a browser to view.")

# 4. Density Chart
plt.figure(figsize=(8,5))
sns.kdeplot(x=df['carat'], y=df['price'], cmap="Blues", fill=True)
plt.title("Density Chart: Carat vs Price")
plt.show()

# 5. Heatmap
# Compute correlation matrix (Dropping non-numeric columns)
corr_matrix = df.select_dtypes(include=['number']).corr()
plt.figure(figsize=(8,5))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Heatmap of Correlation Matrix')
plt.show()

print("Data Visualization Completed Successfully!")
