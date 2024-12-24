import pandas as pd
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
# import json
import pickle
from nltk.corpus import stopwords
import re
sev_data=pickle.load(open('symptomns.pkl','rb'))
model=pickle.load(open('model.pkl', 'rb'))
gqmodel=pickle.load(open('model1.pkl', 'rb'))
tags=pickle.load(open('idtocat.pkl', 'rb'))
gans=pickle.load(open('gans.pkl', 'rb'))
bog=pickle.load(open('bog.pkl', 'rb'))
disease_encoder = pickle.load(open('disease_encoder.pkl','rb'))
#vomiting,fever,chills,joint_pain,neck_pain

def map_symptoms_to_severity(symptoms, sev_data):
    final = []
    for symptom in symptoms:
        for sev in sev_data:
            if symptom.strip() == sev[0]:
                final.append(sev[1])
    return final

# def answer(inputs):
#     list_s=inputs.split(",")
#     final=[]
#     for j in range(len(sev_data)):
#         for i in list_s:
#             if(i==sev_data[j][0]):
#                 final.append(sev_data[j][1])
#     len_f=17-len(final)
#     nulls = [0]*len_f
#     # while(len(final)<5):
#     #     final.append(0)
#     last = [final+nulls]
#     # print(len(last))
#     ans = model.predict(last)
#     return ans[0]

def answer(inputs):
    list_s = inputs.split(",")  # Split the input by commas to get symptoms
    final = map_symptoms_to_severity(list_s, sev_data)
    
    # Handle case where the number of symptoms is less than the model's expected input size (17)
    len_f = 17 - len(final)
    nulls = [0] * len_f  # Add 0 for missing symptoms

    # Combine the final severity values with null values (to fill up to 17 features)
    last = [final + nulls]  # Ensure it's a 2D array for prediction
    
    # Make prediction
    ans = model.predict(last)
    
    # Get the predicted disease label (index)
    predicted_index = ans[0]
    
    # Convert index to the disease name
    predicted_disease = disease_encoder.inverse_transform([predicted_index])[0]
    
    # Return the predicted disease name
    return predicted_disease

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