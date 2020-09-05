######### Nimesha Asintha Muthunayake 
####      M.Sc In DS (Machine Learning )
##        Fare-Classification-MSc2020-master
#         02-07-2020
#         This screipt is tesed in Google Colab notebook 
import pandas as pd
import numpy as np
import datetime
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV

##Data loding part goes here ---------------------------------------------->>


# open file with pd.read_csv , Loading Data file 
df = pd.read_csv("/content/drive/My Drive/Nimesha/Ride Fare Classification/train.csv")
df_test = pd.read_csv("/content/drive/My Drive/Nimesha/Ride Fare Classification/test.csv")

##plotting histoghrams for analysis 
df.hist(alpha=0.5, figsize=(20, 10))
plt.tight_layout()
plt.show()

##Data Clencing part goes here ---------------------------------------------->>

#Removing NaN /Null values from the training data set 
df= df.dropna()

#Clencing data 
X = df.drop(['label','pickup_time','drop_time'], axis=1)
#encoding variable into 1 and 0
y= df['label'].map({'correct': 1, 'incorrect': 0 })
df_test = df_test.drop(['pickup_time','drop_time'], axis=1)

##Data Preparation for training part goes here ------------------------------->>

## Split the data into training and test sets (0.33 for test with random state =0 )
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

##ML Model Preparation part goes here ---------------------------------------->>

#Making model
#The generalization error variance is decreasing to zero in the Random Forest when
#more trees are added to the algorithm. However, the bias of the generalization does not change.
#To avoid overfitting in Random Forest the hyper-parameters of the algorithm should be tuned.
#For example, the number of samples in the leaf.
#Making model
rfc = RandomForestClassifier(n_estimators=1000, max_depth=100, max_features='sqrt')
rfc.fit(X_train,y_train)
rfc_predict = rfc.predict(df_test)

#print(classification_report(y_test, rfc_predict))
print('Accuracy  :',rfc.score(X_test, y_test))

##Output Preparation part goes here ---------------------------------------->>

#preparing the output fil format 
df_test['prediction'] = rfc_predict

df_test = df_test.drop(['additional_fare', 'duration', 'meter_waiting', 'meter_waiting_fare', 'meter_waiting_till_pickup', 'pick_lat', 'pick_lon', 'drop_lat', 'drop_lon', 'fare'] ,axis=1)
print(df_test)
##Create the output file and save it
df_test.to_csv('/content/drive/My Drive/Nimesha/Ride Fare Classification/fair_ride_Prediction_7.csv')
        

    