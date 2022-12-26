import re
import pickle
import pandas as pd
import os
from unidecode import unidecode
import string

from keras.utils import pad_sequences

class TextProcessing():
    def __init__(self) -> None:
        with open(os.path.join("models", "best_count_vect.pkl"), 'rb') as f:
            self.count_vect = pickle.load(f)

        with open(os.path.join("models", "best_tf_transformer.pkl"), 'rb') as f:
            self.tf_transformer = pickle.load(f)

        with open(os.path.join("models",'tokenizer.pkl'),'rb') as f:
            self.tokenizer = pickle.load(f)
        pass

    def case_fold(self, text):
        #membuat semua komentar menjadi huruf kecil 
        text = text.lower()
        # remove tab, new line, ans back slice
        text = text.replace('\\t'," ").replace('\\n'," ").replace('\\u'," ").replace('\\',"")
        #replace - with space
        text = text.replace('-',' ')
        # remove non ASCII (emoticon, chinese word, .etc)
        text = text.encode('ascii', 'replace').decode('ascii')
        # remove consecutive word
        text = re.sub(r'\b(\w+\s*)\1{1,}', '\\1', text)
        # remove mention, link, hashtag
        text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)"," ", text).split())
        # remove incomplete URL
        text.replace("http://", " ").replace("https://", " ")
        #remove punctuation
        text = text.translate(str.maketrans("","",string.punctuation))
        #remove whitestrip
        text = text.strip()
        #remove multiple whitespace into single whitespace
        text = re.sub('\s+',' ',text)
        #remove single char
        text = re.sub(r"\b[a-zA-Z]\b", "", text)
        #remove url
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        text = url_pattern.sub(r'', text)
        return text

    def get_bow(self,text):
        text = text.lower()
        text = self.case_fold(text)
        clean_text = text
        new_df = pd.DataFrame([text],columns=['text'])
        target_predict = self.count_vect.transform(new_df['text'])
        target_predict = self.tf_transformer.transform(target_predict)
        return clean_text,target_predict

    def get_tokenizer(self,text) -> list:
        text = text.lower()
        text = self.case_fold(text)
        clean_text = text
        text = self.tokenizer.texts_to_sequences([text])
        return clean_text, pad_sequences(text, maxlen=128)