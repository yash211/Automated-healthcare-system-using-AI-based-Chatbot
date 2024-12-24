import pandas as pd
import numpy as np
import nltk


from nltk.stem import WordNetLemmatizer
import json
import pickle
from nltk.corpus import stopwords
# import tensorflow
# import tflearn
import re
data = open('datasets/intents.json')

nltk.download('stopwords')
nltk.download('wordnet')

dataset=json.load(data)
#print(dataset)

labels=[]
questions=[]
response={}
for intent in dataset["intents"]:
    for question in intent["patterns"]:
        questions.append(question)
        labels.append(intent["tag"])
    response[intent["tag"]]=intent["responses"]
pickle.dump(response, open('gans.pkl', 'wb'))
df=pd.DataFrame()
df["Questions"]=questions
df["labels"]=labels
#labels encoding basically 
df["labels_id"]=df["labels"].factorize()[0]
category_id_df = df[['labels', 'labels_id']].drop_duplicates()


# Dictionaries for future use
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['labels_id', 'labels']].values)
print(id_to_category)

#print(df)

#preprocessing with nltk
wl=WordNetLemmatizer()
def data_preprocess(x):
    modified_messages = []
    stop_words = set(stopwords.words('english'))
    for i in range(len(x)):
        word = re.sub('[^a-zA-Z]',' ',x[i])
        word = word.lower()
        word = word.split()
        word = [wl.lemmatize(final_word) for final_word in word if not final_word in stop_words]
        word = ' '.join(word)
        modified_messages.append(word)
    df['Questions']=modified_messages
data_preprocess(df['Questions'])

df=df.drop(columns="labels")
print(df)

#tf-idf
from sklearn.feature_extraction.text import CountVectorizer
bow = CountVectorizer(max_features=100,lowercase=True)
X=bow.fit_transform(df['Questions']).toarray()
pickle.dump(bow, open('bog.pkl', 'wb'))
y=df["labels_id"].to_numpy()
y.reshape(-1,1)
print(X.shape,y.shape)

#Spliting dataset into train and test
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
print(y_test.shape)
#model
#multinomial Naive bayes
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB().fit(X_train,y_train)

#Random forest
from sklearn.ensemble import RandomForestClassifier
rfx=RandomForestClassifier(max_depth=10)
rfx.fit(X_train,y_train)

#Getting accuracy
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
predict_x=rfx.predict(X_test)
predict = model.predict(X_test)
print(accuracy_score(y_test,predict))
print(accuracy_score(y_test,predict_x))

# confusion_matrix(model,X_test,y_test)

# import matplotlib.pyplot as plt
# plt.show()

#Random Forest Performs well,now storing useful data
pickle.dump(rfx, open('model1.pkl', 'wb'))
pickle.dump(id_to_category, open('idtocat.pkl', 'wb'))
