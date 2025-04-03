import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# import waterfall_chart

# Load dataset
df = pd.read_csv("cleaned_data.csv")

# Convert categorical to numeric where applicable
df['Experience_Bin'] = df['Experience_Bin'].astype(str)

# Select numeric columns for correlation
numeric_df = df.select_dtypes(include=['number'])

# 1. Heatmap (Correlation)
plt.figure(figsize=(8, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Heatmap")
plt.show()

# 2. Pie Chart (Experience Distribution)
experience_counts = df['Experience_Bin'].value_counts()
plt.figure(figsize=(6,6))
plt.pie(experience_counts, labels=experience_counts.index, autopct='%1.1f%%', colors=sns.color_palette("pastel"))
plt.title("Experience Level Distribution (Pie Chart)")
plt.show()

# 3. Donut Chart (Modified Pie Chart)
plt.figure(figsize=(6,6))
plt.pie(experience_counts, labels=experience_counts.index, autopct='%1.1f%%', colors=sns.color_palette("pastel"), wedgeprops=dict(width=0.4))
plt.title("Experience Level Distribution (Donut Chart)")
plt.show()

# # 4. Waterfall Chart (Salary Progression)
# df_sorted = df.sort_values(by="Salary").dropna()
# waterfall_chart.plot(df_sorted['Name'], df_sorted['Salary'], formatting='{:,.0f}')
# plt.title("Waterfall Chart: Salary Progression")
# plt.show()

# 5. Scatterplot (Age vs Salary)
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="Age", y="Salary", hue="Experience_Bin", palette="coolwarm")
plt.title("Scatterplot: Age vs Salary")
plt.xlabel("Age")
plt.ylabel("Salary")
plt.show()

# 6. Box Plot (Performance Score Outliers)
plt.figure(figsize=(6,6))
sns.boxplot(y=df["Performance_Score"])
plt.title("Boxplot: Performance Score Outliers")
plt.show()

# 7. Violin Plot (Performance Score Distribution)
plt.figure(figsize=(6,6))
sns.violinplot(y=df["Performance_Score"], inner="quartile")
plt.title("Violin Plot: Performance Score Distribution")
plt.show()

# 8. Pair Plot (Relationships Between Numeric Variables)
sns.pairplot(numeric_df)
plt.suptitle("Pair Plot of Numeric Features", y=1.02)
plt.show()