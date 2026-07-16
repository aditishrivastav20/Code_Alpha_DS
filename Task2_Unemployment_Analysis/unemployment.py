import pandas as pd

# Load Dataset
data = pd.read_csv("Unemployment in India.csv")

# First 5 Rows
print("First 5 Rows:")
print(data.head())

# Shape
print("\nRows and Columns:")
print(data.shape)

# Column Names
print("\nColumn Names:")
print(data.columns)

# Dataset Information
print("\nInformation:")
print(data.info())

# Statistics
print("\nStatistics:")
print(data.describe())

# Missing Values
print("\nMissing Values:")
print(data.isnull().sum())

# Remove extra spaces from column names
data.columns = data.columns.str.strip()

# Convert Date column to datetime
data["Date"] = pd.to_datetime(data["Date"], dayfirst=True)

# Check missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Remove missing values
data = data.dropna()

print("\nDataset Shape After Cleaning:")
print(data.shape)

import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(8,5))
sns.histplot(data["Estimated Unemployment Rate (%)"], bins=20, kde=True)
plt.title("Distribution of Unemployment Rate")
plt.savefig("graph1_distribution.png")
plt.show()
plt.figure(figsize=(10,6))

top = data.groupby("Region")["Estimated Unemployment Rate (%)"].mean().sort_values(ascending=False).head(10)

sns.barplot(x=top.values, y=top.index)

plt.title("Top 10 States by Average Unemployment Rate")
plt.savefig("graph2_top10_states.png")
plt.show()
plt.figure(figsize=(8,6))

numeric = data.select_dtypes(include=["float64","int64"])

sns.heatmap(numeric.corr(), annot=True, cmap="coolwarm")

plt.title("Correlation Heatmap")
plt.savefig("graph3_heatmap.png")
plt.show()
plt.figure(figsize=(12,6))

covid = data[data["Date"] >= "2020-03-01"]

monthly = covid.groupby("Date")["Estimated Unemployment Rate (%)"].mean()

plt.plot(monthly.index, monthly.values, marker="o")

plt.title("Covid-19 Impact on Unemployment")
plt.xlabel("Date")
plt.ylabel("Average Unemployment Rate (%)")
plt.xticks(rotation=45)

plt.savefig("graph4_covid.png")
plt.show()

plt.figure(figsize=(6,5))

sns.boxplot(
    x="Area",
    y="Estimated Unemployment Rate (%)",
    data=data
)

plt.title("Rural vs Urban Unemployment")

plt.savefig("graph5_rural_urban.png")
plt.show()