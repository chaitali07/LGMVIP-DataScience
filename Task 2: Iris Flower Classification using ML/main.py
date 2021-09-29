import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
import seaborn as sns

#read
dataset=pd.read_csv('IRIS.csv')
print(dataset)

print(dataset.min())
print(dataset.max())

dataset.info()

dataset.describe()

dataset.mean()
dataset.corr()

sns.countplot(x='Species',data=dataset)
plt.title('Species',fontsize=20)
plt.show()

sns.set(style='whitegrid')
sns.stripplot(x='Species',y='SepalLengthCm',data=dataset)
plt.title('Iris Dataset')
plt.show()


sns.boxplot(x='Species',y='SepalLengthCm',data=dataset)
plt.title('Iris Dataset')
plt.show()
sns.violinplot(x='Species',y='SepalLengthCm',data=dataset)
plt.title('Iris Dataset')
plt.show()
sns.boxplot(x='Species',y='PetalLengthCm',data=dataset)
plt.title('Iris Dataset')
plt.show()
sns.violinplot(x='Species',y='PetalWidthCm',data=dataset)
plt.title('Iris Dataset')
plt.show()


sns.set_style('whitegrid')
sns.pairplot(dataset,hue='Species')
plt.figure(figsize=(10,8))
plt.show()

sns.boxplot(data=dataset)

x=dataset.drop('Species',axis=1)
x.head()

y=dataset['Species']
y.head()

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=1)

from sklearn.preprocessing import StandardScaler
st=StandardScaler()
x_train=st.fit_transform(x_train)
x_test=st.transform(x_test)

from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_train)
print("Training done")

y_pred=model.predict(x_test)

disp_df=pd.DataFrame({'Actual':y_test,'Predicted':y_pred})
print(disp_df)

from sklearn.metrics import confusion_matrix,accuracy_score
print("Confusion Matrix")
print(confusion_matrix(y_test,y_pred))
print("Accuracy:",accuracy_score(y_test,y_pred))


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))

x=dataset.iloc[:,[0,1,2,3]].values
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state =0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss,color='red')
plt.title('The Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

#applying kmeans to the dataset
kmeans=KMeans(n_clusters=3,init='k-means++',max_iter=300,n_init=10,random_state =0)
y_kmeans=kmeans.fit_predict(x)
print(y_kmeans)

#cluster centers
print(kmeans.cluster_centers_)

plt.scatter(x[y_kmeans == 0,0],x[y_kmeans == 0,1], s=60 ,c='green' ,label="Iris-setosa")
plt.scatter(x[y_kmeans == 1,0],x[y_kmeans == 1,1], s=60 ,c='red' ,label="Iris-versicolor")
plt.scatter(x[y_kmeans == 2,0],x[y_kmeans == 2,1], s=60 ,c='blue' ,label="Iris-virginics")
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],c='black',label="Centroids")
plt.title('Iris-Flower Cluster')
plt.xlabel('Sapel Length')
plt.ylabel('Petal Length')
plt.legend()
plt.show()


