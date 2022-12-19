import streamlit as st
import pickle
from preprocessing import *

#pickle load
pick_model = pickle.load(open("model pickle/model_mlp.pkl", 'rb'))     
pick_cv = pickle.load(open("model pickle/count_vect.pkl",'rb')) 
pick_tf = pickle.load(open("model pickle/tf_transformer.pkl", 'rb'))   

def process_text(word):
    #proses analysis sentiment   
    list_word = [word]
    count_vect_data = pick_cv.transform(list_word)
    tfidf_data = pick_tf.transform(count_vect_data).toarray()
    result = pick_model.predict(tfidf_data)
    if result == 1:
        st.markdown('Bersifat **POSITIF**')
    elif result == 0:
        st.markdown('Bersifat **NEUTRAL**')
    else:
        st.markdown('Bersifat **NEGATIF**')

def process_file(dummy_df):
    first_column = dummy_df.columns[0] #mendapatkan nama first column untuk digunakan proses 
    dummy_df[first_column] = dummy_df[first_column].apply(cleansing)
    data_result = []
    for i in range(len(dummy_df)):  
        data = [dummy_df[first_column][i]]
        count_vect_data = pick_cv.transform(data)
        tfidf_data = pick_tf.transform(count_vect_data).toarray()
        result = pick_model.predict(tfidf_data)
        if result == 1:
            data_result.append("Positive")
        elif result == 0:
            data_result.append("Neutral")
        else:
            data_result.append("Negative")
    dummy_df['Label'] = data_result
        
    st.dataframe(dummy_df)