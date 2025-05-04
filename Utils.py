import math
from collections import Counter
import scipy
from scipy.sparse import lil_matrix, csr_matrix
import pandas as pd
import sklearn
from sklearn import *
import numpy as np
import os
import joblib

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

######################### New methods (Alana):