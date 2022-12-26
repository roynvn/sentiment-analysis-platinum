import streamlit as st
import pickle
from preprocessing.text_processing import TextProcessing
from preprocessing.text_processing_file import *
from keras.models import load_model
from keras.utils import pad_sequences
import numpy as np
import pandas as pd

from inference.predict import PredictSentiment
from main import mapping_result

import sqlite3 as sq

#grafik
import plotly.express as px
import plotly.graph_objects as go

TP = TextProcessing()
predict_model = PredictSentiment()


#load model v2 with tfidf
pick_model = pickle.load(open("models/best_tfidf_model_mlp.pkl", 'rb'))     
pick_vect = pickle.load(open("models/best_tfidf_vectorizer.pkl",'rb')) 

def mapping_sql_file(result_sql_file):
    sql_data = 'STORE_FILE.db'
    conn = sq.connect(sql_data)
    cur = conn.cursor()
    cur.execute('''DROP TABLE IF EXISTS FILE_DATA''')
    result_sql_file.to_sql('FILE_DATA', conn, if_exists='replace', index=False) # - writes the df to SQLIte DB
    conn.commit()
    conn.close()


def precocessing_to_file(dummy_df):
    #preprocessing file
    #dummy_df = pd.DataFrame(dummy_df)
    dummy_df['case_fold'] = dummy_df['Tweet'].apply(case_fold)
    dummy_df['tokenize'] = dummy_df['case_fold'].apply(multiword_tokenize) 
    dummy_df['normalisasi'] = dummy_df['tokenize'].apply(normalized_term)
    dummy_df['stopwords'] = dummy_df['normalisasi'].apply(stopwords_removal)
    #dummy_df["stemmer"] = dummy_df["stopwords"].apply(get_stemmed_term).apply(normalized_term).apply(stopwords_removal)
    dummy_df['clean'] = dummy_df['normalisasi'].apply(combine_text)
    
    return dummy_df

def grafik_data_after(dummy_df):
    #Grafik Pie
    sentiment_count = dummy_df["label"].value_counts()
    print(sentiment_count)
    df_graf = pd.DataFrame({"Sentimen" :sentiment_count.index, "Label" :sentiment_count.values})

    graf = px.pie(df_graf, values = "Label", names='Sentimen',
    color_discrete_map={'Positive':'cyan', 'Negative':'royalblue','Netral': 'green'} )
    graf.update_layout(
    title="<b>Perbandingan Komentar Positif,Negatif dan Netral </b>")
    st.plotly_chart(graf)

def akurasi(dummy_df):
    comparison_column = np.where(dummy_df['before'] == dummy_df['label'], "Benar", "Salah")
    dummy_df["hasil"] = comparison_column
    #Akurasi Data
    res_true = 0
    res_false = 0
    for enum in dummy_df['hasil']:
        if enum == "Benar": res_true += 1  
        else: res_false +=1
    res_total = len(dummy_df['Tweet'])
    st.write("Jumlah Tebakan yang benar", res_true)
    st.write("Jumlah Tebakan yang salah", res_false)
    accuracy = (res_true/res_total) * 100
    st.write("Hasil akurasinya", round(accuracy,2), "%")




#PROSES ANALYSIS SENTIMENT
def process_file_ann(dummy_df):

    precocessing_to_file(dummy_df)
    data_result = []
    for i, row in dummy_df.iterrows():
        clean_text,bow = TP.get_bow(row['clean'])
        result_prediction = predict_model.predict_text_ann(bow)
        result_prediction = mapping_result(result_prediction)
        data_result.append(result_prediction)
    
    dummy_df ['label'] = data_result
    dummy_df= dummy_df.drop(['case_fold', 'tokenize','normalisasi','stopwords','clean'], axis=1)
    akurasi(dummy_df)
    #store to database
    mapping_sql_file(dummy_df)
    print("successfull")
    #grafik
    grafik_data_after(dummy_df)
    return st.dataframe(dummy_df)


def process_file_lstm(dummy_df):
    
    precocessing_to_file(dummy_df)
    data_result = []
    for i, row in dummy_df.iterrows():
        clean_text,input_ids = TP.get_tokenizer(row['clean'])
        result_prediction = predict_model.predict_text_lstm(input_ids)
        result_prediction = mapping_result(result_prediction)
        data_result.append(result_prediction)
    
    dummy_df ['label'] = data_result
    dummy_df= dummy_df.drop(['case_fold', 'tokenize','normalisasi','stopwords','clean'], axis=1)
    akurasi(dummy_df)
    #store to database
    mapping_sql_file(dummy_df)
    print("successfull")
    #grafik
    grafik_data_after(dummy_df)

    return st.dataframe(dummy_df)

def process_file_ann_v2(dummy_df):
    data_result = []
    precocessing_to_file(dummy_df)
    for i in range(len(dummy_df)):  
        data = dummy_df['clean'][i]
        v_data = pick_vect.transform([data]).toarray()
        result = pick_model.predict(v_data)
        if result == 1:
            data_result.append("positive")
        elif result == 0:
            data_result.append("neutral")
        else:
            data_result.append("negative")
    dummy_df['label'] = data_result
    dummy_df= dummy_df.drop(['case_fold', 'tokenize','normalisasi','stopwords','clean'], axis=1)
    akurasi(dummy_df)
    #store to database
    mapping_sql_file(dummy_df)
    print("successfull")
    grafik_data_after(dummy_df)
    return st.dataframe(dummy_df)