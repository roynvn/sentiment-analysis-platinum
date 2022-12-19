import streamlit as st
import pandas as pd


def loadpage():
    st.title("Model Neural Network")

    word= st.text_input('Masukkan kalimat')
    st.write('Hasilnya adalah: ', word)

    