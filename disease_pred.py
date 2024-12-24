import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('datasets/dataset.csv', on_bad_lines='skip')
data_1 = pd.read_csv('datasets/Symptom-severity.csv', skipinitialspace=True)

data = data.fillna(0)
data = data.applymap(lambda x: x.strip() if isinstance(x, str) else x)
data_1['Symptom'] = data_1['Symptom'].str.strip()

symptom_severity = dict(zip(data_1['Symptom'], data_1['weight']))

for col in data.columns[1:]:
    data[col] = data[col].apply(lambda x: symptom_severity.get(x.strip() if isinstance(x, str) else x, 0))

for col in data.columns:
    print(f"Column: {col}")
    print(data[col].unique())

# data = data.apply(pd.to_numeric, errors='coerce')
# data = data.dropna()


disease_encoder = LabelEncoder()
data['Disease'] = disease_encoder.fit_transform(data['Disease'])

X = data.iloc[:, 1:]
y = data['Disease']

print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

print(X_train.shape,X_test.shape,y_test.shape,y_train.shape)

model = RandomForestClassifier()
model.fit(X_train, y_train)

predicts = model.predict(X_test)
print(accuracy_score(y_test, predicts))
print(classification_report(y_test, predicts))

pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(disease_encoder, open('disease_encoder.pkl', 'wb'))