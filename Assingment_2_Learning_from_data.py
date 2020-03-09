#!/usr/bin/python3
# Author: J.F.P. (Richard) Scholtens
# Studentnr: s2956586
# Date: 23/09/2019

import numpy as np
import time
import string
import sys
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from matplotlib import pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


translator = str.maketrans('', '', string.punctuation)
stemmer = PorterStemmer()


def show_report(classifier, vec, Xtrain, Xtest, Ytrain, Ytest):
    """This function trains a classifier and prints a report of the results.
    It also returns the predictions."""
    classifier = Pipeline([('vec', vec), ('cls', classifier)])
    t0 = time.time()

    # Here the classifier learns which feautures are linked to what label.
    classifier.fit(Xtrain, Ytrain)
    train_time = time.time() - t0
    print("Training time: ", train_time)
    t1 = time.time()

    # Here the classifier predicts the label of features based on the
    # learned process in the step before.
    Yguess = classifier.predict(Xtest)
    test_time = time.time() - t1
    print("Test time: ", test_time)

    # Here the classifier compares the gold standard labels with the
    # predict labels retrieved from the step before.
    print("Accuracy score: ", accuracy_score(Ytest, Yguess))

    # Here the system tries to predict the labels of the test data, where
    # Yguess are the predicted labels and Xtest is the data
    print(classification_report(Ytest, Yguess))
    return Yguess


def show_plot(matrix):
    """This function shows a plot of a matrix which has four rows."""

    # Creates plot
    plt.title("Accuracy vs F-score ")
    plt.plot(matrix[0], matrix[1], label="Accuracy")
    plt.plot(matrix[0], matrix[2], label="F-score Wheigted")
    plt.plot(matrix[0], matrix[3], label="F-score Micro")
    plt.plot(matrix[0], matrix[4], label="F-score Macro")
    plt.xlabel("Number of neighbors")
    plt.ylabel("Accuracy/F-score")
    plt.legend()
    plt.show()


# This function opens a file and retrieves all lines in this file.
# It then removes all whitespace from this line and then creates a list with
# where the first item is genre, the second item is the sentiment, and the
# third is the id number of the review. Everything after this are the words
# of the review. To retrieve sentiment the variable use_sentiment must be True.
# To use genre's the variable use_sentiment must be False. One of these
# variables will be used as labels. It then returns the documents and labels.
def preprocessing(file, use_sentiment):
    """This function pre-process the data by removing stop words, white space
    and punctuation, stemming, tokenization. It returns documents and label
    lists then creates a list with where the first item is genre, the second
    item is the sentiment, and the third is the id number of the review.
    Everything after this are the words of the review. To retrieve sentiment
    the variable use_sentiment must be True. To use genre's the variable
    use_sentiment must be False. One of these variables will be used as labels.
    It then returns the documents and labels."""
    documents = []
    labels = []
    with open(file, encoding='utf-8') as f:
        for line in f:
            line = line.lower().strip()
            line = line.translate(translator)
            tokens = word_tokenize(line)
            filtered_tokens = [stemmer.stem(w) for w in tokens]
            documents.append(filtered_tokens[3:])
            if use_sentiment:
                # 2-class problem: positive vs negative
                labels.append(tokens[1])
            else:
                # 6-class problem: books, camera, dvd, health, music, software
                labels.append(tokens[0])
    return documents, labels


# a dummy function that just returns its input
def identity(x):
    return x


def main(argv):
    """The program reads a textfile and retrieves the data
    and the labels linked to the data. After this it
    splits the data in training data and test data.
    The same goes for the labels."""
    X = sys.argv[1]
    Y = sys.argv[2]
    Xtrain, Ytrain = preprocessing(X, use_sentiment=False)
    Xtest, Ytest = preprocessing(Y, use_sentiment=False)

    # A TF_IDF vectorizer creates a score scale based on frequency of input
    # within different documents. Every word will have a different score for
    # a different document. This score can be used as feature for the
    # classifier.The classifier learns from these features in order to make
    # calculated predictions.
    # let's use the TF-IDF vectorizer

    print("Baseline results")
    # Create dummy classifer baseline
    dummy = DummyClassifier(strategy='stratified')
    # "Train" model
    t0 = time.time()
    dummyscore = dummy.fit(Xtrain, Ytrain)
    train_time = time.time() - t0
    print("Training time: ", train_time)

    t1 = time.time()
    baselineguess = dummyscore.predict(Xtest)
    test_time = time.time() - t1
    print("Test time: ", test_time)

    # Get score
    print("accuracy score: ", accuracy_score(Ytest, baselineguess))
    print(classification_report(Ytest, baselineguess))

    vec = TfidfVectorizer(preprocessor=identity,
                          tokenizer=identity,
                          min_df=0.001,
                          max_df=1000,
                          smooth_idf='18',
                          max_features=3000)

    MNB = MultinomialNB(alpha=0.1, class_prior=None, fit_prior=False)
    print("MultinomialNB results")
    show_report(MNB, vec, Xtrain, Xtest, Ytrain, Ytest)

    DTC = DecisionTreeClassifier(random_state=0, min_samples_leaf=6)
    print("Decision Tree results")
    show_report(DTC, vec, Xtrain, Xtest, Ytrain, Ytest)

    KNC = KNeighborsClassifier(n_neighbors=8)
    print("KNeighborsClassifier results with K = 8")
    show_report(KNC, vec, Xtrain, Xtest, Ytrain, Ytest)

    # Use this code when wanting multiple K's.
    # k_acc = []
    # k_f1_w = []
    # k_f1_micro = []
    # k_f1_macro = []
    # k = []
    # for i in range(30):
    #     i += 1
    #     KNC = KNeighborsClassifier(n_neighbors=i)
    #     print("KNeighborsClassifier results with K = {}".format(i))
    #     Yguess = show_report(KNC, vec, Xtrain, Xtest, Ytrain, Ytest)
    #     k_acc.append(accuracy_score(Ytest, Yguess))
    #     k_f1_w.append(f1_score(Ytest, Yguess, average='weighted'))
    #     k_f1_micro.append(f1_score(Ytest, Yguess, average='micro'))
    #     k_f1_macro.append(f1_score(Ytest, Yguess, average='macro'))
    #     k.append(i)
    # k_acc = np.array(k_acc)
    # k_f1_w = np.array(k_f1_w)
    # k_f1_micro = np.array(k_f1_micro)
    # k_f1_macro = np.array(k_f1_macro)
    # k = np.array(k)
    # matrix = [k, k_acc, k_f1_w, k_f1_micro, k_f1_macro]
    # show_plot(matrix)
if __name__ == '__main__':
    main(sys.argv)
