#Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

#Loading Dataset
churn_df=pd.read_csv('telecom_churn_clean.csv')
#Subsetting data
X=churn_df[['total_day_charge', 'total_eve_charge']].values
y=churn_df['churn'].values
#Splitting data for training and testing
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=21,stratify=y)
#loop that will give different test ans training accuracies for different values of k
test_accuracies={}
train_accuracies={}
neighbors=np.arange(1,26)
for neighbor in neighbors:
    knn=KNeighborsClassifier(n_neighbors=neighbor)
    knn.fit(X_train,y_train)
    test_accuracies[neighbor] = knn.score(X_test,y_test)
    train_accuracies[neighbor] = knn.score(X_train,y_train)
print(test_accuracies,train_accuracies)
#plotting the accuracies for analysis
plt.figure(figsize=(10,10))
plt.title("KNN:Varying Number of K")
plt.plot(neighbors,train_accuracies.values(),label='Training Accuracy')
plt.plot(neighbors,test_accuracies.values(),label='Test Accuracy')
plt.legend()
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracies')
plt.show()