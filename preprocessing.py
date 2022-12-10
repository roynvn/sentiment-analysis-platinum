import numpy as np
import re
from unidecode import unidecode
import string
import pandas as pd

import nltk
from nltk.tokenize import word_tokenize,MWETokenizer
from nltk.probability import FreqDist
nltk.download('punkt')
nltk.download('stopwords')


#case fold 
def cleansing(text):
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