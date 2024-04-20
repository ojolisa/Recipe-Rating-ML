import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

trainset = pd.read_csv('train.csv')
testset = pd.read_csv('test.csv')

#print(trainset.describe())
#print(testset.describe())

# print(trainset.isnull().sum())
# print(testset.isnull().sum())

# print(trainset.head())
# print(testset.head())

#sns.countplot(x='Rating', data=trainset)
#plt.show()

'''counts = trainset['ID'].value_counts()
print(counts)
if any(counts > 1):
    print("The column has common values.")
else:
    print("The column has no common values.")'''

print(trainset.loc[trainset['ID']==90])