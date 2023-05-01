
import numpy as np
import spacy 
import streamlit as st
from streamlit_option_menu import option_menu
import pickle
import pandas as pd
import gensim.downloader as api                                         # The api module in Gensim allows for easy access to pre-trained word embeddings, which are word representation vectors learned from large corpora of text. These pre-trained word embeddings can be used as features in natural language processing (NLP) and text mining tasks, such as word similarity, document classification, and text generation.
                 # The Pipeline function is used to define a sequence of data preprocessing and model training steps in a machine learning pipeline. Each step is defined as a tuple containing a name for the step and an instance of a transformer or an estimator.
                                                                        # The make_pipeline function is a convenient function for creating a pipeline without explicitly specifying names for the steps. It automatically generates names for the steps based on the names of the classes of the transformer or estimator objects.
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
                                                                        # The CountVectorizer function is used to convert a collection of text documents to a matrix of token counts. It creates a sparse matrix representation where each row represents a document and each column represents a unique token, with the cell value indicating the count of that token in the corresponding document.
                                                                        # The TfidfVectorizer function is used to convert a collection of text documents to a matrix of TF-IDF (Term Frequency-Inverse Document Frequency) features. It creates a sparse matrix representation where each row represents a document and each column represents a unique token, with the cell value indicating the TF-IDF score of that token in the corresponding document.
nlp = spacy.load("spacy_model_updated_v2")                                      # loads the pre-trained English language model "en_core_web_lg" from the Spacy library. 
# wv = api.load('word2vec-google-news-300')   

from gensim.models import KeyedVectors

# load the saved model from file
wv = KeyedVectors.load_word2vec_format('word2vec-google-news-300.bin', binary=True)

# set page title
st.set_page_config(page_title='AXON CUSTOMER REVIEW CATEGORIZATION')

# page configuration
hide_st_style="""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)

original_title = '<p style="color:#FED80C; font-size: 36px; font-weight:bold">CUSTOMER REVIEW CATEGORIZATION</p>'
st.markdown(original_title, unsafe_allow_html=True)
#st.title(':blue[AXON] CUSTOMER REVIEW CATEGORIZATION')
#st.title('A title with _italics_ :blue[colors] and emojis :sunglasses:')

st.sidebar.image('data:image/png;base64,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')

option = st.sidebar.selectbox(
    'Select the binary Model',
    ('Cost', 'Product', 'Sales','Customer Service','Others','Training'))

st.sidebar.write('You selected:', option)

pickle_in = open('models', 'rb')                                                # Opening Classifier in read-byte mode. 
models = pickle.load(pickle_in)


def generate_word_embeddings(sent):
    words = sent.split()                                               # Splitting the input sentence into words
    word_vectors = [wv[word] for word in words if word in wv]          # Extracting word vectors for words present in the pre-trained word vectors
    if word_vectors:
        return np.mean(word_vectors, axis=0)                           # Calculating the average word vector
    else:
        return np.zeros((300,))                                        # Returning a zero vector if no word vectors are found for the input sentence

labels = list(nlp.get_pipe('ner').labels)

def generate_ner_tags(doc):
    ner_dict = dict.fromkeys(labels, 0)                                # Creating a dictionary to store NER labels and initializing all values to 0
    # Using a dictionary comprehension
    ner_dict = {labels: 0 for labels in labels}
    doc = nlp(doc)
    for tokens in doc.ents:
        if tokens.label_ in ner_dict:                                  # Checking if the NER label is present in the dictionary
            ner_dict[tokens.label_]=1                                  # Setting the value to 1 if the NER label is found
    return(list(ner_dict.values()))       
                                    

def normalize(review, lowercase, remove_stopwords, punctuations_rm):
    if lowercase:
        review = review.lower()                                        # Converting the text to lowercase if the `lowercase` flag is set
    doc = nlp(review)
    lemmatized = list()
    for token in doc:
          if remove_stopwords and not token.is_stop:                   # Checking if the token is a stopword and the `remove_stopwords` flag is set
            if punctuations_rm:
                if not token.is_punct:                                 # Checking if the token is a punctuation and the `punctuations_rm` flag is set
                    lemmatized.append(token.lemma_)                    # Lemmatizing the token and appending to the list
            else: 
                lemmatized.append(token.lemma_)
    return " ".join(lemmatized)                                        # Joining the lemmatized tokens with space and returning as normalized text


def testing(sent):
    sent = normalize(sent, lowercase=True, remove_stopwords=True, punctuations_rm=True)
    sent_length = len(nlp(sent))
    sent_vector = generate_word_embeddings(sent)

    ner_tag = generate_ner_tags(sent)
    x3 = np.reshape(np.array(sent_length), (-1, 1))
    x1 = np.reshape(np.array(sent_vector), (-1, 1))
    x2 = np.reshape(np.array(ner_tag), (-1, 1))
    test = np.concatenate((x1, x2, x3),axis=0)
    #test = np.concatenate((x2,x3),axis=0)
    return test.T

def predictions_final(sent):
    mydict = {}
    if models['Product'][0].predict_proba(sent)[:,1] > (models['Product'][1]):
        mydict["Product"] = (models['Product'][0].predict_proba(sent)[:,1])
    if models['Cost'][0].predict_proba(sent)[:,1] > models['Cost'][1]:
        mydict["Cost"] = (models['Cost'][0].predict_proba(sent)[:,1])
    if models['Sales'][0].predict_proba(sent)[:,1] > models['Sales'][1]:
        mydict["Sales"] = (models['Sales'][0].predict_proba(sent)[:,1])
    if models['Customer Service'][0].predict_proba(sent)[:,1] > models['Customer Service'][1]:
        mydict["Customer Service"] = (models['Customer Service'][0].predict_proba(sent)[:,1])
    if models['Training'][0].predict_proba(sent)[:,1] > models['Training'][1]:
        mydict["Training"] = (models['Training'][0].predict_proba(sent)[:,1])
    
    mydict = dict(sorted(mydict.items(), key=lambda x: x[1], reverse = True))
    len(mydict)
    if len(mydict) == 0:
        st.write("Others")
    else:
        for i,j in mydict.items():
            st.write(i, " -> ", '{:.2%}'.format(j[0]))

user_input = st.text_input('Customer review', 'Enter text here')
#st.write('The current movie title is', title)


if st.button('Predict'):
    sent = testing(user_input)
    predictions_final(sent)
    #st.write(option)
else:
    st.write('None')

threshold = st.sidebar.slider('Select the probability threshold value', 0.0, 1.0, 0.50)
st.sidebar.write("The model will predict the model at ", threshold,'threshold')
