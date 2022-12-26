import streamlit as st


def loadpage():
    st.title("WEBSITE ANALISIS SENTIMEN MENGGUNAKAN ANN dan LSTM")
    st.markdown("""
            <div>
                <p class="abstrak", align="justify">
                Website ini digunakan untuk melakukan analisis sentimen dengan hasil akhir akan menampilkan komentar yang termasuk kedalam komentar positif, negatif atau netral.
                </br>
                </br>
                Terdapat 2 menu yang dapat digunakan yaitu: </br>
                1.  ANN</br>
                    Untuk melakukan klasifikasi komentar menjadi positif atau negatif menggunakan model ANN
                    dan menampilkan grafik perbandingan komentar positif, negatif atau netral yang dimiliki pada data.
                    </br>
                2.  LSTM</br>
                    Untuk melakukan klasifikasi komentar menjadi positif atau negatif menggunakan model LSTM
                    dan menampilkan grafik perbandingan komentar positif, negatif atau netral yang dimiliki pada data.
                </br>
                </br>
            </div>               
    """,unsafe_allow_html=True)