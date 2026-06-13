import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score,accuracy_score,confusion_matrix,classification_report
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#loading iris datatset
iris = load_iris(as_frame=True)
df = iris.frame
print(df.head())

#changing the species names
df["target"] = df["target"].replace({0:"setosa",1:"versicolor",2:"virginica"})

#printing all the columns
print(df.head())
x = df.iloc[:,0:4]
y = df.iloc[:,4]

#using scalar to minimize the values to a standard scale so that it will be easy for the knc algoritm ot train
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

#assigning the data
x_train,x_test,y_train,y_test = train_test_split(x_scaled,y,test_size=0.2,random_state=42,stratify=y)
print("\n\n")
#printing the data
print(x_train[0])#after scaling
print(x_test[0])#after scaling
print("\n\n")
#model training
model = KNeighborsClassifier()
#the lengths and the species name are used to train the model
model.fit(x_train,y_train) 

#for some value let us test the model
sample_predict = model.predict(x_test)
print(x_test[0]," : ",sample_predict[0])

#finding the accuracy of the model accuracy score,f1_score
accuracy = accuracy_score(y_test,sample_predict)
f1 = f1_score(y_test,sample_predict,average="weighted")
print("\nThe accuracy")
print("accuracy : ",accuracy) 
print("f1_score : ",f1)

#data visualization of our prediction to real data through confusion matrix
print("\n\n")
print("DATA VISUALIZATION BETWEEN ACTUAL AND PREDICTED")
cm = confusion_matrix(y_test,sample_predict)
plt.figure(figsize=(5,5))
sb.heatmap(cm,annot=True,fmt="d",cmap="Blues",xticklabels=model.classes_,yticklabels=model.classes_)
plt.title("Confusion matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
print("\n\n")

while True:
    choice=input("Enter p-predict (or) e-exit : ")
    if choice=="e":
        break
    print("\nEnter data:\n")
    sl = float(input("Enter sepal length : "))
    sw = float(input("Enter sepal width : "))
    pl = float(input("Enter petal length : "))
    pw = float(input("Enter petal width : "))
    flower = np.array([[sl,sw,pl,pw]])
    prediction = model.predict(scaler.fit_transform(flower))
    print("The prediction species : ",prediction[0])