import streamlit as st
import pandas as pd


def app():
    
    st.title("Model LSTM")
    
    word = st.text_input('Masukkan kalimat')
    st.write('Hasilnya adalah: ',word)

    uploaded_file = st.file_uploader('Choose a file', type = {"csv","xlsx"})
    if uploaded_file is not None:
        df_file = pd.read_csv(uploaded_file, encoding='latin-1')
        #df_file = pd.read_excel(uploaded_file, sep='\t')
        st.dataframe(df_file)
    else:
        st.warning('you need to upload a csv or excel file')

    
    if st.button("Process the data"):
        df_final = df_file[['Tweet']].copy()
        st.dataframe(df_final)