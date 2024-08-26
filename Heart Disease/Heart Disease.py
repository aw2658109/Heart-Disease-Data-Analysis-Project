# Questions:
# 1. Import The Libraries And Dataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("heart.csv")
print(df)
"""Heart Disease Analysis | Python Pandas Project
Heart disease is the number one cause of death globally. Heart disease is concertedly contributed by hypertension, diabetes, overweight and unhealthy lifestyles.
This project covers manual exploratory data analysis and using pandas in pycharm."""

#  2. Display Top 5 Rows of The Dataset:
print(df.head(5))

#  3. Check The Last 5 Rows of The Dataset
print(df.tail(5))

#  4. Find Shape of Our Dataset (Number of Rows And Number of Columns)
print(np.shape(df))

#  5. Get Information About Our Dataset Like Total Number Rows, Total Number of Columns,
#  Datatypes of Each Column And Memory Requirement
print(df.info())

# 6. Check Null Values In The Dataset
print(df.isnull().sum())

# 7. Check For Duplicate Data and Drop Them
print(df.duplicated().any())

# 7. Check For Duplicate Data and Drop Them
print(df.drop_duplicates())

# 8. Get Overall Statistics About The Dataset
print(df.describe())

# 8. Get Overall Statistics About The Dataset
print(df.describe(include="all"))

# 9. Draw Correlation Matrix
plt.figure(figsize=(17,6))
sns.heatmap(df.corr(),annot=True)
plt.show()
# 10. How Many People Have Heart Disease,
# And How Many Don't Have Heart Disease In This Dataset?
print(df['target'].value_counts())
sns.countplot(df['target'].value_counts())
plt.show()

#  11. Find Count of  Male & Female in this Dataset:
print(df["sex"].value_counts())
sns.countplot(df['sex'].value_counts())
plt.xticks([1,0],['Female','Male'])
plt.show()

# 12. Find Gender Distribution According to The Target Variable
sns.countplot(x="sex",hue="target",data=df)
plt.xticks([1,0],['Female','Male'])
plt.legend(labels=["Diseas","No Diseas"])
plt.show()

# 13. Check Age Distribution In The Dataset:
sns.histplot(df["age"], bins=20, kde=True)
plt.show()

# 14. Check Chest Pain Type:
sns.countplot(df['cp'].value_counts())
plt.xticks([0, 1, 2, 3], ["typical angina", "atypical angina", "non-anginal pain", "asymptomatic"])
plt.xticks(rotation=75)
plt.show()

#  15. Show The Chest Pain Distribution As Per Target Variable:
sns.countplot(x="cp",hue="target",data=df)
plt.legend(labels=['Diesase','NoDiesase'])
plt.show()

#  16. Show Fasting Blood Sugar Distribution According To Target Variable:
sns.countplot(x='fbs',hue='target', data=df)
plt.legend(labels=['Disease','NoDisease'])
plt.show()

# 17.  Check Resting Blood Pressure Distribution:
print(df['trestbps'].hist())
plt.show()

#  18. Compare Resting Blood Pressure As Per Sex Column:
g = sns.FacetGrid(df, hue='sex', aspect=4)
g.map(sns.kdeplot, 'trestbps', shade=True)
plt.legend(labels=['Male','Female'])
plt.show()

#  19. Show Distribution of Serum cholesterol:
print(df['chol'].hist())
plt.show()


#  20. Plot Continuous Variables:
sns.pairplot(df, hue='target')
plt.show()



