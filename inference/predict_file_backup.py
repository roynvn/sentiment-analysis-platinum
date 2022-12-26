import streamlit as st
import pickle
from preprocessing.text_processing import TextProcessing
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
# #load model
# #ann
# pick_model = pickle.load(open("models/model_mlp.pkl", 'rb'))     
# pick_cv = pickle.load(open("models/count_vect.pkl",'rb')) 
# pick_tf = pickle.load(open("models/tf_transformer.pkl", 'rb'))   

#lstm 
# pick_tokenizer = pickle.load(open("models/tokenizer.pkl",'rb')) 
# model = load_model("models/model_lstm.h5")


def mapping_sql_file(result_sql_file):
    sql_data = 'STORE_FILE.db'
    conn = sq.connect(sql_data)
    cur = conn.cursor()
    cur.execute('''DROP TABLE IF EXISTS FILE_DATA''')
    result_sql_file.to_sql('FILE_DATA', conn, if_exists='replace', index=False) # - writes the df to SQLIte DB
    conn.commit()
    conn.close()

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

# def process_file_ann(dummy_df):
#     first_column = dummy_df.columns[0] #mendapatkan nama first column untuk digunakan proses 
#     dummy_df[first_column] = dummy_df[first_column].apply(TP.case_fold)
#     data_result = []
#     for i in range(len(dummy_df)):  
#         data = [dummy_df[first_column][i]]
#         count_vect_data = pick_cv.transform(data)
#         tfidf_data = pick_tf.transform(count_vect_data).toarray()
#         result = pick_model.predict(tfidf_data)
#         if result == 1:
#             data_result.append("Positive")
#         elif result == 0:
#             data_result.append("Neutral")
#         else:
#             data_result.append("Negative")
#     dummy_df['Label'] = data_result
#     #store to database
#     mapping_sql_file(dummy_df)
#     print("successfull")
#     return st.dataframe(dummy_df)

def process_file_ann(dummy_df):
    first_column = dummy_df.columns[0] #mendapatkan nama first column untuk digunakan proses 
    #dummy_df[first_column] = dummy_df[first_column].apply(TP.case_fold)
    data_result = []
    
    for i, row in dummy_df.iterrows():
        clean_text,bow = TP.get_bow(row[first_column])
        result_prediction = predict_model.predict_text_ann(bow)
        result_prediction = mapping_result(result_prediction)
        data_result.append(result_prediction)
    
    dummy_df ['label'] = data_result
    akurasi(dummy_df)
    #store to database
    mapping_sql_file(dummy_df)
    print("successfull")
    #grafik
    grafik_data_after(dummy_df)
    return st.dataframe(dummy_df)




# def process_file_lstm(dummy_df):
#     sentiment = ['Neutral','Positive','Negative']
#     first_column = dummy_df.columns[0] #mendapatkan nama first column untuk digunakan proses 
#     dummy_df[first_column] = dummy_df[first_column].apply(TP.case_fold)
#     data_result = []
#     for i in range(len(dummy_df)):  
#         data = [dummy_df[first_column][i]]
#         print(data)
#         sequence = pick_tokenizer.texts_to_sequences(data)
#         print(sequence)
#         test = pad_sequences(sequence, maxlen=128)
#         result = sentiment[np.around(model.predict(test), decimals=0).argmax(axis=1)[0]]
#         data_result.append(result)
#         print(result)
#     dummy_df ['Label'] = data_result
#     #store to database
#     mapping_sql_file(dummy_df)
#     print("successfull")
#     return st.dataframe(dummy_df)

def process_file_lstm(dummy_df):
    first_column = dummy_df.columns[0] #mendapatkan nama first column untuk digunakan proses 
    #dummy_df[first_column] = dummy_df[first_column].apply(TP.case_fold)
    data_result = []

    for i, row in dummy_df.iterrows():
        clean_text,input_ids = TP.get_tokenizer(row[first_column])
        result_prediction = predict_model.predict_text_lstm(input_ids)
        result_prediction = mapping_result(result_prediction)
        data_result.append(result_prediction)
    
    dummy_df ['label'] = data_result
    akurasi(dummy_df)
    #store to database
    mapping_sql_file(dummy_df)
    print("successfull")
    #grafik
    grafik_data_after(dummy_df)

    return st.dataframe(dummy_df)

