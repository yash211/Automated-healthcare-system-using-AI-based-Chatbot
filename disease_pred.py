import pickle
import pandas as pd
import numpy as np
data = pd.read_csv('datasets/dataset.csv',skipinitialspace=True)
data_1 = pd.read_csv('datasets/Symptom-severity.csv',skipinitialspace=True)
data=data.fillna(0)
data.head()
symptoms=data_1['Symptom'].unique()
print(len(symptoms))
sev_data=data_1.values
pickle.dump(sev_data,open('symptomns.pkl','wb'))
print(len(sev_data))
data=data.replace(symptoms[0],sev_data[0][1])
# data=data.replace(" skin_rash",sev_data[1][1])
for i in range(1,len(sev_data)):
    data=data.replace(sev_data[i][0],sev_data[i][1])
data.head()
data = data.replace('dischromic _patches', 0)
data = data.replace('spotting_ urination',0)
data = data.replace('foul_smell_of urine',0)
data.head()
X = data.iloc[:,1:].values
y = data['Disease'].values
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42,shuffle=True)
print(len(X))
print(len(X_train),len(X_test),len(y_train),len(y_test))
from sklearn.metrics import accuracy_score,f1_score,classification_report
from sklearn.ensemble import RandomForestClassifier

randomness=RandomForestClassifier()
randomness.fit(X_train,y_train)

predicts=randomness.predict(X_test)
print(accuracy_score(y_test,predicts))
print(classification_report(y_test,predicts))


Model_file = 'model.pkl'

pickle.dump(randomness, open(Model_file, 'wb'))
