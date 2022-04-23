import pandas as pd
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
from nltk.corpus import stopwords
import re
sev_data=pickle.load(open('symptomns.pkl','rb'))
model=pickle.load(open('model.pkl', 'rb'))
gqmodel=pickle.load(open('model1.pkl', 'rb'))
tags=pickle.load(open('idtocat.pkl', 'rb'))
gans=pickle.load(open('gans.pkl', 'rb'))
bog=pickle.load(open('bog.pkl', 'rb'))
#vomiting,fever,chills,joint_pain,neck_pain
def answer(inputs):
    list_s=inputs.split(",")
    final=[]
    for j in range(len(sev_data)):
        for i in list_s:
            if(i==sev_data[j][0]):
                final.append(sev_data[j][1])
    len_f=17-len(final)
    nulls = [0]*len_f
    # while(len(final)<5):
    #     final.append(0)
    last = [final+nulls]
    # print(len(last))
    ans = model.predict(last)
    return ans[0]

#General Question Answering Model
wl=WordNetLemmatizer()
def data_preprocess(x):
    word = re.sub('[^a-zA-Z]',' ',x)
    word = word.lower()
    word = word.split()
    word = [wl.lemmatize(final_word) for final_word in word if not final_word in set(stopwords.words('english'))]
    word = ' '.join(word)
    return word

def general_question(inputs):
    cleaned_sentence=data_preprocess(inputs)
    cleaned_sentence=[cleaned_sentence]
    feature_vectors=bog.transform(cleaned_sentence)
    predict_ans=gqmodel.predict(feature_vectors)
    ans=tags[predict_ans[0]]
    return gans[ans]
#print(general_question("Hey i got cuts,so what should i do"))