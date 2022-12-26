import streamlit as st
import requests
import pandas as pd

from loadpage import *

from inference.predict_file import process_file_ann,process_file_lstm, process_file_ann_v2

#grafik
import plotly.express as px
import plotly.graph_objects as go


def call_api(text, path):
    url = f"http://127.0.0.1:5555/{path}/v1"

    data_payload = {
        "text":text
    }

    response = requests.post(url,json=data_payload)
    result = response.json()
    return result

def grafik_data_asli(df_file):
    #Grafik Pie
    sentiment_count = df_file["before"].value_counts()
    print(sentiment_count)
    df_graf = pd.DataFrame({"Sentimen" :sentiment_count.index, "Label" :sentiment_count.values})

    graf = px.pie(df_graf, values = "Label", names='Sentimen',
    color_discrete_map={'Positive':'cyan', 'Negative':'royalblue','Netral': 'green'} )
    graf.update_layout(
    title="<b>Perbandingan Komentar Positif,Negatif dan Netral </b>")
    st.plotly_chart(graf)


def main():
    model = ['Loadpage','ANN','LSTM']
    choice = st.sidebar.selectbox("Select Model",model)
    
    uploaded_file = st.sidebar.file_uploader("Upload Dataset",type=["csv","xlsx","xls"])
    if uploaded_file is not None:
        #df_file = pd.read_csv(uploaded_file, encoding='latin-1')
        df_file = pd.read_csv(uploaded_file,engine='python',encoding='cp1252')
    
    if choice == 'ANN':
        st.title("Model Neural Network")

        st.header("PROCESS TEXT")
        text = st.text_input('Masukkan kalimat dalam bahasa indonesia')
        path = 'predict_sentiment_ann'
        result = call_api(text,path)
        st.write("Hasil: ", result)

        #without API
        st.header("PROCESS FILE")

        if st.button("Lihat Data"):
            try:
                st.dataframe(df_file)
                grafik_data_asli(df_file)
            except:
                st.error("DATAFRAME BELUM DI UPLOAD")

        if st.button("Process the data"):
            try:
                #dummy_df = df_file.iloc[:, :1] #selalu menggunakan first column, return as dataframe
                dummy_df = df_file.copy()
                #process_file_ann(dummy_df)
                process_file_ann_v2(dummy_df)
            except:
                st.error("BELUM ADA DATAFRAME")
    
    


    elif choice == 'LSTM':
        st.title("Model LSTM") 
        path = "predict_sentiment_lstm"
        text = st.text_input('Masukkan kalimat dalam bahasa indonesia')
        result = call_api(text,path)
        st.write("Hasil: ", result)

        #without API
        st.header("PROCESS FILE")

        if st.button("Lihat Data"):
            try:
                st.dataframe(df_file)
                grafik_data_asli(df_file)
            except:
                st.error("DATAFRAME BELUM DI UPLOAD")

        if st.button("Process the data"):
            try:
                #dummy_df = df_file.iloc[:, :1] #selalu menggunakan first column, return as dataframe
                dummy_df = df_file.copy()
                process_file_lstm(dummy_df)
            except:
                st.error("BELUM ADA DATAFRAME")
    
    else:
        loadpage()


if __name__=="__main__":
    main()