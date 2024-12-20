#importing packages
import numpy as np
import pandas as pd

#importing data from kaggle
data = pd.read_csv("rawdataset/creditcard.csv")
print("\n\n display data : \n",data.head())

##datainformation:
print("\n\n data information : \n")
print(data.info())

##data columns :
print("\n\n datacolumns : " , data.columns)

###null values of data :
print("\n\n data null values: \n ", data.isnull().sum())

##describe data :
print("\n\n describe data : \n ",data.describe())

###
data = data.drop("Time", axis=1)
print("\n\n drop column : \n",data.head())

###no. of rows and columns:
print(" \n\n rows , columns : \n ",data.shape)

###We need to standardize the 'Amount' feature before modelling.
###For that, we use the StandardScaler function from sklearn. Then, we just have to drop the old feature :
from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
#standard scaling
data['std_Amount'] = scaler.fit_transform(data['Amount'].values.reshape (-1,1))

#removing Amount
data = data.drop("Amount", axis=1)


###data visualization:
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

##class values:
sns.countplot(x="Class", data=data)
plt.show()

###class counts:
fraud = data[data['Class']==1]
normal = data[data['Class']==0]
print(fraud.shape, normal.shape)

print("\n\n column names : \n" ,data.columns)
##export data to csv file:
from pandas import DataFrame
DataFrame(data).to_csv("finaldataset/updatedataset.csv" ,index=False , header=True)

print("data successfully updated:")