# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required packages and print the present data.

2.Print the placement data and salary data. 

3.Find the null and duplicate values.

4.Using logistic regression find the predicted values of accuracy , confusion matrices. 

5.Display the results.

## Program:
```

Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: SRI SAI PRIYA.S 
RegisterNumber: 212222240103
```
```
import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
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
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
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
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
## Placement Data:

![319872339-3c0cd4cc-3f4f-44ea-b777-60cb31b490a4](https://github.com/SriSaiPriyaSenthilvel/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119475702/df1d9712-9882-4b02-abb8-1a5aea43294f)

## Salary Data:

![319872577-e930e9b4-b5e2-4e57-a38b-05107d92cd88](https://github.com/SriSaiPriyaSenthilvel/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119475702/0d584771-6e4b-4fbf-8c21-db346b37789b)

## Checking the null() function:

![319872795-a0955fef-d69a-4926-98d0-e6c9a47c2fbb](https://github.com/SriSaiPriyaSenthilvel/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119475702/c1b9349b-337e-4de6-bcae-74ce93516437)

## Data Duplicate:

![319872982-7bf51b24-0782-4c98-afba-5cf91227a732](https://github.com/SriSaiPriyaSenthilvel/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119475702/08f0dd61-6391-4423-a868-a827e9d18026)

## Print Data:

![319873253-fd8c8015-b304-443c-b045-f81817407bc5](https://github.com/SriSaiPriyaSenthilvel/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119475702/9ebf4040-53da-4f94-aa3a-3e426eaccbc8)

## Data-Status:

![319873656-013624fd-a58d-4235-9f98-289788fa9173](https://github.com/SriSaiPriyaSenthilvel/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119475702/22022172-69a8-429e-8673-50a2b7eed46e)

![319873737-a9b4c894-4feb-4423-aa83-dba95b47d44d](https://github.com/SriSaiPriyaSenthilvel/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119475702/1f008542-0446-4d28-b787-c61e5970b71f)

## Y_prediction Array:

![319873949-450eefbe-457a-4843-9511-f3bafe38946c](https://github.com/SriSaiPriyaSenthilvel/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119475702/0d3b72ab-0f40-4ebe-ba9f-854b3f71a1db)

## Accuracy value:

![319874216-1f3b35cc-505f-4865-b9bc-cbdb6280e0be](https://github.com/SriSaiPriyaSenthilvel/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119475702/977c8fc0-3998-4c60-ba8c-6d00a5d8a967)

## Confusion Array:

![319874503-6e1cdaf7-5851-4c6a-8328-127f460698dc](https://github.com/SriSaiPriyaSenthilvel/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119475702/9c7f9ea1-24d7-4fed-a88a-d0f654796b4e)

## Classification Report:

![319874715-1d5a71ab-05bc-42b4-a1f9-b82f01be910f](https://github.com/SriSaiPriyaSenthilvel/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119475702/1416fd9a-ec0b-459d-9347-7b7d2483638b)

## Prediction of LR:

![319874945-500a9e75-e1a5-4d97-9c92-5e8f4dbbb1d4](https://github.com/SriSaiPriyaSenthilvel/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119475702/b88dbb94-9a0d-443e-af79-a340d1734c96)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
