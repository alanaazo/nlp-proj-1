import math
from collections import Counter
import pandas as pd
import sklearn
from sklearn import *
import numpy as np
import os
import joblib
# from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
import sys
import scipy
import scipy.linalg
from scipy.sparse import lil_matrix, csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from scipy.sparse import hstack


if not hasattr(scipy.linalg, 'triu'):
    def triu(m, k=0):
        m = np.asanyarray(m)
        mask = np.tri(m.shape[0], m.shape[1], k=k, dtype=bool)
        return np.where(mask, m, 0)

    scipy.linalg.triu = triu

################# Existing methods from the deliverable notebook startup:
def cast_list_as_strings(mylist):
    """
    return a list of strings
    """
    mylist_of_strings = []
    for x in mylist:
        mylist_of_strings.append(str(x))

    return mylist_of_strings


def get_features_from_df(df, count_vectorizer):
    """
    returns a sparse matrix containing the features build by the count vectorizer.
    Each row should contain features from question1 and question2.
    """
    q1_casted =  cast_list_as_strings(list(df["question1"]))
    q2_casted =  cast_list_as_strings(list(df["question2"]))
    
    ############### Begin exercise ###################
    # what is kaggle                  q1
    # What is the kaggle platform     q2
    X_q1 = count_vectorizer.transform(q1_casted)
    X_q2 = count_vectorizer.transform(q2_casted)    
    X_q1q2 = scipy.sparse.hstack((X_q1,X_q2))
    ############### End exercise ###################

    return X_q1q2

def print_mistake_k(k, mistake_indices, predictions):
    print(train_df.iloc[mistake_indices[k]].question1)
    print(train_df.iloc[mistake_indices[k]].question2)
    print("true class:", train_df.iloc[mistake_indices[k]].is_duplicate)
    print("prediction:", predictions[mistake_indices[k]])


def get_mistakes(clf, X_q1q2, y):

    ############### Begin exercise ###################
    predictions = clf.predict(X_q1q2)
    incorrect_predictions = predictions != y 
    incorrect_indices,  = np.where(incorrect_predictions)
    ############### End exercise ###################
    
    if np.sum(incorrect_predictions)==0:
        print("no mistakes in this df")
    else:
        return incorrect_indices, predictions


######################### New methods (Ralitsa):

def get_features_from_df_tfidf(df, tfidf_vectorizer):
    """
    returns a sparse matrix containing the features build by the count vectorizer.
    Each row should contain features from question1 and question2.
    """
    q1_casted =  cast_list_as_strings(list(df["question1"]))
    q2_casted =  cast_list_as_strings(list(df["question2"]))
    
    ############### Begin exercise ###################
    # what is kaggle                  q1
    # What is the kaggle platform     q2
    X_q1 = tfidf_vectorizer.transform(q1_casted)
    X_q2 = tfidf_vectorizer.transform(q2_casted)    
    X_q1q2 = scipy.sparse.hstack((X_q1,X_q2))
    ############### End exercise ###################
    return X_q1q2


class TFIDF_Vectorizer:
    def __init__(self, max_features=None):
        self.max_features = max_features
        self.vocab = {}  # token -> index
        self.IDF = {}    # IDF: term -> IDF value

    def fit(self, data):
        # start fresh:
        self.vocab = {}  # token -> index
        self.IDF = {}    # IDF: term -> IDF value
        DF = {}

        for sentence in data:
            tokens = sentence.lower().split()
            for token in tokens:
                if token in DF:
                    DF[token] += 1
                else:
                    DF[token] = 1

        # word to index mapping
        for idx, token in enumerate(DF):
            self.vocab[token] = idx
        
        # Calculate IDF for each token in the vocabulary
        # USe log, to prevent very small IDF values
        for token, count in DF.items():
            self.IDF[token] = math.log(len(data) / count)


    def transform(self, data):
        n_docs = len(data)
        n_tokens = len(self.vocab)
        matrix = lil_matrix((n_docs, n_tokens))

        for doc_idx, sentence in enumerate(data):
            tokens = sentence.lower().split()
            TF = Counter(tokens)

            for token, count in TF.items():
                if token in self.vocab:
                    tf = count / len(tokens)
                    idf = self.IDF.get(token, 0)
                    tfidf = tf * idf
                    term_idx = self.vocab[token]
                    matrix[doc_idx, term_idx] = tfidf

        return matrix.tocsr()  # Convert to efficient CSR format

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)


def cosine_similarity_vectors(vec1, vec2):
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    
    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product/(norm1 * norm2)


def euclidean_distance_vectors(vec1, vec2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(vec1, vec2)))


def save_model(model, path="models", filename="model.pkl"):
    """
    Saves a scikit-learn model to the given path and filename.
    """
    os.makedirs(path, exist_ok=True)  # create directory if it doesn't exist
    full_path = os.path.join(path, filename)
    joblib.dump(model, full_path)
    print(f"Model saved to {full_path}")


def load_model(path="models", filename="model.pkl"):
    """
    Loads a scikit-learn model from the given path and filename.
    """
    full_path = os.path.join(path, filename)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"No model found at {full_path}")
    model = joblib.load(full_path)
    print(f"Model loaded from {full_path}")
    return model


def check_model_saved(path="models", filename="model.pkl"):
    model_path = os.path.join(path, filename)
    return os.path.isfile(filename)


def clean_text(text):
    return str(text).lower().strip()


def char_ngram_similarity(q1, q2, vectorizer):
    q1, q2 = clean_text(q1), clean_text(q2)
    tfidf = vectorizer.transform([q1, q2])
    return cosine_similarity(tfidf[0], tfidf[1])[0][0]


def starts_with_indicator(text):
    text = str(text).strip().lower()
    return {f"starts_with_{word}": int(text.startswith(word)) for word in start_words}

start_words = ['how', 'can', 'what', 'why', 'are', 'do', 'does', 'is', 'should', 'could']

def jaccard_similarity(q1, q2):
    # Tokenize the text into sets of words
    set_q1 = set(str(q1).lower().split())
    set_q2 = set(str(q2).lower().split())
    
    # Calculate Jaccard similarity
    intersection = len(set_q1 & set_q2)  # common words
    union = len(set_q1 | set_q2)         # all unique words
    return intersection / union if union != 0 else 0  # Avoid division by zero


# For use with LDA and LSI, but can't make versions of gensim, numpy and scipy work together.....

# def gensim_preprocess_en(texts):
#     stop_words = set(stopwords.words('english'))
#     return [
#         [word for word in simple_preprocess(doc) if word not in stop_words]
#         for doc in texts
#     ]


# def get_embedding(model, dictionary, text):
#     bow = dictionary.doc2bow(text)
#     topic_dist = model[bow]
#     dense_vector = np.zeros(model.num_topics)
#     for idx, val in topic_dist:
#         dense_vector[idx] = val
#     return dense_vector


# def compute_similarity_features(q1_list, q2_list, model, dictionary):
#     features = []
#     for q1, q2 in zip(q1_list, q2_list):
#         q1_vec = get_embedding(model, dictionary, q1)
#         q2_vec = get_embedding(model, dictionary, q2)
#         sim = cosine_similarity_vectors([q1_vec], [q2_vec])[0][0]
#         features.append(sim)
#     return np.array(features).reshape(-1, 1)

######################### New methods (Alana):