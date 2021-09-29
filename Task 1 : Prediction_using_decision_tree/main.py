#import all the libraries required

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

#Loading The .csv Dataset

df = pd.read_csv('iris.csv')

print("The data is:")

print(df.head(10))

#getting dataset info
print("Dataset Info")
df.info()

#visualizing dataset
sns.set(style="ticks", color_codes=True)
iris = sns.load_dataset("iris")
g = sns.pairplot(iris)
plt.show()


#Spliting The Dataset Into Test And Train Data
x = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values
y = df[['Species']].values
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.3, random_state=42)

#Fitting The Train Data Into Decision Tree Model And Predicting The Test Data

model = DecisionTreeClassifier()
model.fit(x_train, y_train)

predict = model.predict(x_test)

#Checking The Accuracy
accuracy = accuracy_score(y_test, predict)
print("Accuracy is: ",accuracy)

#Giving A New Input And Checking The Output
new = []
sepalLength = input('Enter the Sepal Length (max range-8cm): ')
new.append(sepalLength)
sepalWidth = input('Enter the Sepal Width (max range-4.5cm): ')
new.append(sepalWidth)
petalLength = input('Enter the Petal Length (max range-7cm): ')
new.append(petalLength)
petalWidth = input('Enter the Petal Width (max range-2.5cm): ')
new.append(petalWidth)
new_array = [new]
output_prediction = model.predict(new_array)


species = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
for item in range(len(output_prediction)):
    print(output_prediction[item])
    
    
fig = plt.figure(figsize=(25,20))
figure = tree.plot_tree(model,
                       feature_names=(['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']),
                       class_names=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'],
                       rounded=True,
                       filled=True)
fig.savefig("decistion_tree.png")
