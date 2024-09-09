# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data.
2. Print the placement data and salary data.
3. Find the null and duplicate values.
4. Using logistic regression find the predicted values of accuracy , confusion matrices.
5. Display the results.

## Program:
```
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: HARSHITHA V
RegisterNumber: 212223230074
```
```
import pandas as pd
data=pd.read_csv(r"C:\Users\admin\Desktop\Placement_Data.csv")
data.head()
data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()
data1.isnull().sum()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1
x=data1.iloc[:,:-1]
x
y=data1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy
from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion
from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```
# Output:

## HEAD
![Screenshot (25)(1)](https://github.com/user-attachments/assets/0cfd79ee-7899-4f9f-9456-545463b2fc43)


## COPY
![Screenshot (26)(1)](https://github.com/user-attachments/assets/b7aad1da-a91a-4806-8f55-2d8f0608f92c)


## FIT TRANSFORM
![Screenshot (27)(1)](https://github.com/user-attachments/assets/04437a70-e1c1-4a92-a2f9-db9adc49bb2a)

## LOGISTIC REGRESSION 
![Screenshot (28)(1)](https://github.com/user-attachments/assets/03ccb8d6-99ba-454a-8c25-dc15144d4570)


## ACCURACY SCORE
![Screenshot (29)(1)](https://github.com/user-attachments/assets/d9c3c32a-5f29-432a-8dba-e52f4bb24641)


## CONFUSION MATRIX
![Screenshot (30)(1)](https://github.com/user-attachments/assets/3ac1dde5-7761-4e89-9c7c-2f890ee5115c)


## CLASSFICATION REPORT
![image](https://github.com/user-attachments/assets/2b5c2137-5901-4a58-a629-eb5018b23890)

## PREDICTION
![image](https://github.com/user-attachments/assets/949aebd5-531e-4f59-b2c8-82db99b9f3fa)


# Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
