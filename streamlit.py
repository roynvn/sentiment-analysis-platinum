import streamlit as st
import requests
import pandas as pd

from loadpage import *

from inference.predict_file import process_file_ann,process_file_lstm


def call_api(text, path):
    url = f"http://127.0.0.1:5555/{path}/v1"

    data_payload = {
        "text":text
    }

    response = requests.post(url,json=data_payload)
    result = response.json()
    return result


def main():
    model = ['Loadpage','ANN','LSTM']
    choice = st.sidebar.selectbox("Select Model",model)
    
    uploaded_file = st.sidebar.file_uploader("Upload Dataset",type=["csv","xlsx","xls"])
    if uploaded_file is not None:
        #df_file = pd.read_csv(uploaded_file, encoding='latin-1')
        df_file = pd.read_csv(uploaded_file,engine='python',encoding='cp1252')
    if st.button("Lihat Data"):
        try:
            st.dataframe(df_file)
        except:
            st.error("DATAFRAME BELUM DI UPLOAD")
    
    if choice == 'ANN':
        st.title("Model Neural Network")

        st.header("PROCESS TEXT")
        text = st.text_input('Masukkan kalimat dalam bahasa indonesia')
        path = 'predict_sentiment_ann'
        result = call_api(text,path)
        st.write("Hasil: ", result)

        #without API
        st.header("PROCESS FILE")
        if st.button("Process the data"):
            try:
                dummy_df = df_file.iloc[:, :1] #selalu menggunakan first column, return as dataframe
                process_file_ann(dummy_df)
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
        if st.button("Process the data"):
            try:
                dummy_df = df_file.iloc[:, :1] #selalu menggunakan first column, return as dataframe
                process_file_lstm(dummy_df)
            except:
                st.error("BELUM ADA DATAFRAME")

    else:
        loadpage()


    
# st.title('predict sentiment analysis in Bahasa Indonesia')

# option = st.selectbox(
#     'pakai model apa?',
#     ('LSTM', 'ANN'))
# text = st.text_input('Masukkan kalimat dalam bahasa indonesia')

# if text:
#     if option == 'LSTM':
#         path = 'predict_sentiment_lstm'
#         result = call_api(text,path)
#     else:
#         path = "predict_sentiment"
#         result = call_api(text,path)
#     st.write("your result: ", result)

if __name__=="__main__":
    main()