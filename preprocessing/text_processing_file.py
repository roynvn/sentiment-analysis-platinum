import string
import re

#toknize
import nltk
from nltk.tokenize import word_tokenize,MWETokenizer
from nltk.probability import FreqDist
nltk.download('punkt')
nltk.download('stopwords')


import pandas as pd

#stopword 
from nltk.corpus import stopwords
#stemmer
#from mpstemmer import MPStemmer


#case fold 
def case_fold(text):
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


#TOKENIZE
def multiword_tokenize(text):
    mwe = open("./data/mwe.txt", "r",).read().split("\n")
    protected_tuples = [word_tokenize(word) for word in mwe]
    protected_tuples_underscore = ['_'.join(word) for word in protected_tuples]
    tokenizer = MWETokenizer(protected_tuples)
    # Tokenize the text.
    tokenized_text = tokenizer.tokenize(word_tokenize(text))
    # Replace the underscored protected words with the original MWE
    for i, token in enumerate(tokenized_text):
        if token in protected_tuples_underscore:
            tokenized_text[i] = mwe[protected_tuples_underscore.index(token)]
    return tokenized_text

#normalisasi
normalized_word = pd.read_excel("./data/normalisasi.xlsx")

normalized_word_dict = {}

for index, row in normalized_word.iterrows():
    if row[0] not in normalized_word_dict:
        print(row)
        normalized_word_dict[row[0]] = row[1] 

def normalized_term(text):
    return [normalized_word_dict[term] if term in normalized_word_dict else term for term in text]


#STOPWORDS
dump_stopwords = stopwords.words('indonesian')
extend_stopword = open("./data/extend_stopword.txt", "r",).read().split("\n")
for element_es in extend_stopword:
    dump_stopwords.append(element_es)
delete_from_stopword = open("./data/delete_from_stopword.txt", "r",).read().split("\n")
for element in delete_from_stopword:
    if element in dump_stopwords:
        dump_stopwords.remove(element)
list_stopwords = set(dump_stopwords)

def stopwords_removal(text):
    return [word for word in text if word not in list_stopwords]


# #STEMMER
# stemmer = MPStemmer()
# def stemmed_wrapper(term):
#     return stemmer.stem(term)

# def get_stemmed_term(text):
#     term_dict = {}
#     for document in df["stopwords"]:
#         for term in document:
#             if term not in term_dict:
#                 term_dict[term] = " "
#         for term in term_dict:
#             term_dict[term] = stemmed_wrapper(term) 
# # apply stemmed term to dataframe
#     return [term_dict[term] for term in text]

#clean
def combine_text(text):
    komentar = " "
    return (komentar.join(text))