#app.py
import streamlit as st
import pandas as pd


from loadpage import *
from cleansing import *
from preprocessing import *

from apps.process_ann import process_text,process_file



def main():
    model = ['Loadpage','ANN','LSTM']
    choice = st.sidebar.selectbox("Select Model",model)
    
    uploaded_file = st.sidebar.file_uploader("Upload Dataset",type=["csv","xlsx","xls"])
    if uploaded_file is not None:
        df_file = pd.read_csv(uploaded_file, encoding='latin-1')
    if st.button("Lihat Data"):
        try:
            st.dataframe(df_file)
        except:
            st.error("DATAFRAME BELUM DI UPLOAD")

    if choice == 'ANN':
        st.title("Model Neural Network")

        st.header("PROCESS TEXT")
        word= st.text_input('Masukkan kalimat')
        word = cleansing(word)
        st.write('Cleansing: ', word)
        #process text
        #process_text(word)

        st.header("PROCESS FILE")
        if st.button("Process the data"):
            try:
                dummy_df = df_file[['Tweet']].copy()
                process_file(dummy_df)
            except:
                st.error("BELUM ADA DATAFRAME")


    
    # di command dulu
    # elif choice == 'LSTM':
    #     st.title("Model LSTM")

    #     st.write("Input Text")
    #     word= st.text_input('Masukkan kalimat')
    #     st.write('Hasilnya adalah: ', word)

    #     if st.button("Process the data"):
    #         df_final = df_file[['Tweet','HS']].copy()
    #         st.dataframe(df_final)

    else:
        loadpage()

if __name__=="__main__":
    main()