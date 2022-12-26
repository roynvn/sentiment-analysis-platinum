from flask import Flask, request, jsonify
from preprocessing.text_processing import TextProcessing
from inference.predict import PredictSentiment
import sqlite3 as sq
import pandas as pd 

TP = TextProcessing()
predict_model = PredictSentiment()
app = Flask(__name__)

def mapping_result(result_prediction):
    if result_prediction == 0:
        return "negative"
    elif  result_prediction == 1:
        return "positive"
    else:
        return "neutral"

def mapping_sql(result_sql):
    sql_data = 'STORE_TEXT.db'
    conn = sq.connect(sql_data)
    result_sql.to_sql('TEXT_DATA', conn, if_exists='append', index=False)
    conn.commit()
    conn.close()

@app.route("/predict_sentiment_ann/v1", methods=["POST"])
def predict_sentiment_ann():
    text = request.get_json()['text']
    clean_text,bow = TP.get_bow(text)
    result_prediction = predict_model.predict_text_ann(bow)
    result_prediction = mapping_result(result_prediction)
    result = jsonify({"text":clean_text, "result_sentiment":result_prediction})
    #db
    dict_result ={"text":[clean_text], "result_sentiment":[result_prediction]}
    df_text =  pd.DataFrame.from_dict(dict_result)
    mapping_sql(df_text)
    print('succes to insert sql')
    return result

@app.route("/predict_sentiment_lstm/v1", methods=["POST"])
def predict_sentiment_lstm():
    text = request.get_json()['text']
    clean_text,input_ids = TP.get_tokenizer(text)
    result_prediction = predict_model.predict_text_lstm(input_ids)
    result_prediction = mapping_result(result_prediction)
    result = jsonify({"text":clean_text, "result_sentiment":result_prediction})
    #db
    dict_result ={"text":[clean_text], "result_sentiment":[result_prediction]}
    df_text =  pd.DataFrame.from_dict(dict_result)
    mapping_sql(df_text)
    print('succes to insert sql')
    return result

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5555, debug=True)