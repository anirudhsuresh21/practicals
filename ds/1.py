import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

# load dataset
df = pd.read_csv("updated_messy_data.csv")

# find missing values
print("Found missing values count")
a=df.isnull().sum()
print(df)
print(a)

#convert to int which is wrong data format
def convert(value):
    try:
        return float(value)
    except ValueError:
        return np.nan
    
df['Age']=df['Age'].apply(convert)
df['Salary']=df['Salary'].apply(convert)

# handle missing data    
df['Age'].fillna(df['Age'].mean(),inplace=True)
df['Salary'].fillna(df['Salary'].mean(),inplace=True)

#print after filling missing values
print(df)
print(df.isnull().sum())

# drop dupes
df.drop_duplicates(subset=['ID'],keep='first',inplace=True)
print(df)

#binning to remove noise data
df['Experience_new']= pd.cut(df['Experience'],bins=[0,2,5,10],labels=['Juniors','Mid','Senior'])
print(df)

#boxPlot
plt.figure(figsize=[6,4])
plt.boxplot(df['Performance_Score'],vert=False)
plt.title("Performance Score Outliers")
plt.show()

#interquartiles
Q1 = df['Performance_Score'].quantile(0.25)
Q3 = df['Performance_Score'].quantile(0.75) 
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df = df[(df['Performance_Score'] >= lower_bound) & (df['Performance_Score'] <= upper_bound)]
print(df)


#scaling zscore and minmax
scaler_minmax = MinMaxScaler()
df[['Age','Salary']] = scaler_minmax.fit_transform(df[['Age','Salary']])
print(df)

scaler_zscore = StandardScaler()
df[['Age','Salary']] = scaler_zscore.fit_transform(df[['Age','Salary']])
print(df)


#coorelation Heatmap
num_dcol = df.select_dtypes(include=['number'])
plt.figure(figsize=(8,6))
sns.heatmap(num_dcol.corr(),annot=True,cmap='coolwarm',fmt='.2f')
plt.title("Correlation Heatmap")
plt.show()

#chisquare test
contingency_table= pd.crosstab(df['Experience_new'],df['Performance_Score'])
chi2,p,dof,expected = chi2_contingency(contingency_table)
print("Chisqaure resultt")
print(chi2)
print(p)
print(dof)
print(expected)


# Save cleaned data
df.to_csv("cleaned_data1.csv", index=False)
print("Data cleaning complete! Cleaned data saved as 'cleaned_data.csv'")