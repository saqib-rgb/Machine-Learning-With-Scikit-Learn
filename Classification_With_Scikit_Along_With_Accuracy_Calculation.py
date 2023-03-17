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
knn=KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train,y_train)
score=knn.score(X_test,y_test)
print(score)