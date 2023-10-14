# Ex.No:1-Data-Preprocessing
## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## REQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

Kaggle :
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

Data Preprocessing:

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

Need of Data Preprocessing :

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
## Step 1:
Importing the libraries.

## Step 2:
Importing the dataset.

## Step 3:
Taking care of missing data.

## Step 4:
Encoding categorical data.

## Step 5:
Normalizing the data.

## Step 6:
Splitting the data into test and train.

## Step 7:
End the program.

## PROGRAM:
## Developed By:B.Pavizhi
## Register Number:212221230077
```
import pandas as pd

df=pd.read_csv("/content/Churn_Modelling.csv")

df.head()

df.isnull().sum()

df.drop(["RowNumber","Age","Gender","Geography","Surname"],inplace=True,axis=1)

print(df)

x=df.iloc[:,:-1].values

y=df.iloc[:,-1].values

print(x)

print(y)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

df1 = pd.DataFrame(scaler.fit_transform(df))

print(df1)

from sklearn.model_selection import train_test_split

xtrain,ytrain,xtest,ytest=train_test_split(x,y,test_size=0.2,random_state=2)

print(xtrain)

print(len(xtrain))

print(xtest)

print(len(xtest))

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

df1 = sc.fit_transform(df)

print(df1)
```

## OUTPUT:
## df.head() :
![1](https://github.com/pavizhi/Ex.No.1---Data-Preprocessing/assets/95067176/00a680e7-e976-4243-aba6-55690de8c395)

## df.isnull().sum() :
![2](https://github.com/pavizhi/Ex.No.1---Data-Preprocessing/assets/95067176/b55e491d-24b1-4cf4-a0f5-c1ef489c7e09)

## df value :
![3](https://github.com/pavizhi/Ex.No.1---Data-Preprocessing/assets/95067176/24dcda4c-f1d0-4ea2-948a-a8c4b6f81981)

## VALUES OF INPUT AND OUTPUT DATA ON VAR X AND Y :
![4](https://github.com/pavizhi/Ex.No.1---Data-Preprocessing/assets/95067176/40468c90-61ae-476e-bb4b-d47148a63d7e)

## NORMALIZING DATA:
![5](https://github.com/pavizhi/Ex.No.1---Data-Preprocessing/assets/95067176/5fe3f3ac-9dcb-49e3-a58d-8e65976c0a98)

## X_TRAIN AND Y_TRAIN VALUES :
![6](https://github.com/pavizhi/Ex.No.1---Data-Preprocessing/assets/95067176/a0234a7a-b907-45e4-870d-a5d1bb6c1991)

## X AND Y VALUES :
![71](https://github.com/pavizhi/Ex.No.1---Data-Preprocessing/assets/95067176/c7a23d5f-9d15-4b41-a53f-72bebb95fcf5)

## X_TEST AND Y_TEST VALUES :
![8](https://github.com/pavizhi/Ex.No.1---Data-Preprocessing/assets/95067176/fefa7d1d-1e5c-41aa-9447-f6d9e8d00e4d)



## RESULT:
Thus,the program to perform Data preprocessing in a data set downloaded from Kaggle is implemented successfully .

