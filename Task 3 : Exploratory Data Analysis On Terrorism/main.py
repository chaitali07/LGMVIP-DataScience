import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


data=pd.read_csv('globalterrorismdb_0718dist.csv',encoding ='latin1')

print(data.head())

print(data.describe())

print(data.info())

print(data.dtypes)

print(data.columns)

print(data.columns.values)

data=data[['eventid', 'iyear','region','provstate','city','crit1', 'crit2', 'crit3','success', 'suicide', 'attacktype1','targtype1','natlty1','gname','guncertain1','claimed','weaptype1','nkill','nwound']]
print(data.head())

data.isnull().sum()

data['nkill']=data['nkill'].fillna(0)
data['nwound']=data['nwound'].fillna(0)
data['casualities']=data['nkill']+data['nwound']
print(data.isnull().sum())


print(data.describe())

year=data['iyear'].unique()
year_count=data['iyear'].value_counts(dropna=False).sort_index()
plt.figure(figsize=(15,10))
sns.barplot(x=year,y=year_count,palette="tab10")
plt.xticks(rotation=50)
plt.xlabel('Attacking Year',fontsize=20)
plt.ylabel('Number of attacks each year',fontsize=22)
plt.show()

yearc=data[['iyear','casualities']].groupby('iyear').sum()
yearc.plot(kind='bar',color='red',figsize=(15,6))
plt.title("Casualities")
plt.xlabel('Years',fontsize=15)
plt.ylabel('No. of Casualities',fontsize=15)
plt.show()

data['attacktype1'].value_counts().plot(kind='bar',figsize=(20,10),color='blue')
plt.xticks(rotation=50)
plt.xlabel('Attack Type',fontsize=20)
plt.ylabel('Number os attacks')
plt.title('Number of attacks')
plt.show()

plt.subplots(figsize=(20,10))
sns.countplot(data['targtype1'],order=data['targtype1'].value_counts().index,palette='gist_heat',edgecolor=sns.color_palette("deep"));
plt.xticks(rotation=90)
plt.xlabel('Attack type',fontsize=20)
plt.ylabel('count')
plt.title('Type of attack')
plt.show()

sattk=data.success.value_counts()[:10]
print(sattk)

print(data.gname.value_counts()[1:11])

#So the conclusion is 
#1.Taliban has done most of attacks 2.Most of the attacks were made in the year 2014 3.bombing type attack were used most of time
