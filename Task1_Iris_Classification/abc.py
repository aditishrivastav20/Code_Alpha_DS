import os

print("Current directory:", os.getcwd())
print(os.listdir())

import pandas as pd
data = pd.read_csv("iris copy.csv")
print(data.head())

print("\nRows and Columns:")
print(data.shape)

print("\nColumn Names:")
print(data.columns)

print("\nInformation:")
print(data.info())

print("\nStatistics")
print(data.describe())

print("Total Missing Values:")
print(data.isnull().sum())

print("\nFlower Types:")
print(data["Species"].unique())

print("\nNumber of Flowers in Each Species:")
print(data["Species"].value_counts())

import matplotlib.pyplot as plt
data["Species"].value_counts().plot(kind="bar")
plt.title("Number of Flowers in Each Species")
plt.xlabel("Species")
plt.ylabel("Count")
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt
sns.scatterplot(
    x="PetalLengthCm",
    y="PetalWidthCm",
    hue="Species",
    data=data
)
plt.title("Petal Length vs Petal Width")
plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.show()

numeric_data = data.select_dtypes(include=["float64", "int64"])
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()
sns.pairplot(data, hue="Species")
plt.show()

sns.pairplot(data, hue="Species")
plt.show()
