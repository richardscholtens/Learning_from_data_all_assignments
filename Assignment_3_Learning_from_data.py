#!/usr/bin/python3
# Author: J.F.P. (Richard) Scholtens
# Studentnr: s2956586
# Author: Remy Wang
# Studentnr: s2212781
# Date: 30/09/2019

import sys
import string
import nltk
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
#from nltk.corpus import stopwords
from sklearn.svm import SVC, LinearSVC
#from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import KFold
from statistics import mean
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import FeatureUnion


def identity(x):
    """A dummy function that just returns its input"""
    return x


def preprocessing(file, use_sentiment):
    """This function pre-process the data by removing stop words, white space
    and punctuation, stemming, tokenization. It returns documents and label
    lists then creates a list with where the first item is genre, the second
    item is the sentiment, and the third is the id number of the review.
    Everything after this are the words of the review. To retrieve sentiment
    the variable use_sentiment must be True. To use genre's the variable
    use_sentiment must be False. One of these variables will be used as labels.
    It then returns the documents and labels."""
    translator = str.maketrans('', '', string.punctuation)
    #stemmer = PorterStemmer()
    #stop_words = set(stopwords.words('english'))
    documents = []
    labels = []
    with open(file, encoding='utf-8') as f:
        for line in f:
            line = line.lower().strip()
            line = line.translate(translator)
            tokens = line.split()

            #filtered_tokens = [stemmer.stem(w) for w in tokens] # if w not in stop_words]
            documents.append(tokens[3:])
            if use_sentiment:
                # 2-class problem: positive vs negative
                labels.append(tokens[1])
            else:
                # 6-class problem: books, camera, dvd, health, music, software
                labels.append(tokens[0])
    return documents, labels


def get_count_vectorizer():
  """Retuns a CountVectorizer"""
  vectorizer = CountVectorizer(preprocessor = identity,
                               tokenizer = tokenize_pos, 
                               ngram_range=(1, 3))
  return vectorizer


def get_tfidf_vectorizer():
    """Returns a TfidfVectorizer."""
    vectorizer = TfidfVectorizer(preprocessor=identity,
                                 tokenizer=tokenize_pos,
                                 ngram_range=(1, 3))

                                #Use min_df to ignore the terms that appear in less than 0.1% documents 
                                #min_df=0.001, 
                                #Use max_df to ignore the terms that appear in more than 1000 documents
                                #max_df=1000,
                                #Use smooth_idf to allow words not seen in training data to be processed
                                #smooth_idf='18',
                                #Keep other parameters default  as this seems to lead to the best score
    return vectorizer


def run_gridsearch(classifier, parameters, Xtrain, Ytrain, Xtest, Ytest):
    """Returns best parameter for classifier."""
    print("### USING GRID-SEARCH")
    clf = GridSearchCV(classifier, parameters, cv=5, n_jobs=-1)
    clf.fit(Xtrain, Ytrain)
    Yguess2 = clf.predict(Xtest)
    print(accuracy_score(Ytest, Yguess2))
    print(clf.best_params_)
    return clf.best_params_, True


def run_classifier(X, Y, classifier, parameters=False, best_parameters=False):
  """Divides the training data in k-folds subsets. Each time, one of the k
  subsets is used as the test set/ validation set and the other k-1 subsets are
  put together to form a training set. This significantly reduces bias as we are
  using most of the data for fitting, and also significantly reduces variance as
  most of the data is also being used in validation set. It also prints,
  confusion matrix, classification report, accuracy and the average of all
  accuracies"""
  c = 0
  X = np.asarray(X)
  Y = np.asarray(Y)
  accuracies = []
  kf = KFold(n_splits=10) # Define the split - into 10 folds 
  for train_index, test_index in kf.split(X):
      c += 1
      Xtrain, Xtest = X[train_index], X[test_index]
      Ytrain, Ytest = Y[train_index], Y[test_index]
      
      if parameters:
          best_parameters, check = run_gridsearch(classifier, parameters, Xtrain,
                                                Ytrain, Xtest, Ytest)
          parameters = False
          classifier.set_params(**best_parameters)
      elif best_parameters:
          classifier.set_params(**best_parameters)
      classifier.fit(Xtrain, Ytrain)
      Yguess = classifier.predict(Xtest)
      
      print("Using {0}-fold.".format(c))
      print(confusion_matrix(Ytest, Yguess)) 
      print(classification_report(Ytest, Yguess))
      print("Accuracy score of the model: ", 
            accuracy_score(Ytest, Yguess))

      accuracies.append(accuracy_score(Ytest, Yguess))
  print("Average accuracy classifier: ", mean(accuracies))
  return classifier


def tokenize_pos(tokens):
    return [token+"_POS-"+tag for token, tag in nltk.pos_tag(tokens)]


def show_results(cls, Xtest, Ytest):
    """Shows the results of two classifiers."""
    predict =  cls.predict(Xtest)
    print("Best model accuracy: ", accuracy_score(Ytest, predict))
    print(confusion_matrix(Ytest, predict)) 
    print(classification_report(Ytest, predict))


def main(argv):
    """The program reads a textfile and retrieves the data
    and the labels linked to the data. After this it
    splits the data in training data and test data.
    The same goes for the labels."""
    X = sys.argv[1]
    Y = sys.argv[2]

    Xtrain, Ytrain = preprocessing(X, use_sentiment=True)
    Xtest, Ytest = preprocessing(Y, use_sentiment=True)   

    # dummy = DummyClassifier(strategy='stratified')
    # run_classifier(Xtrain, Ytrain, dummy)

    count_vec = get_count_vectorizer()
    tfidf_vec = get_tfidf_vectorizer()
    vec = FeatureUnion([("count", count_vec), ("tfidf", tfidf_vec)])

                                           
    # 3.1.1 Default settings  
    # SVC_cls = Pipeline([('vec', vec),
    #                       ('cls', SVC(kernel='linear', C=5))])
    
    # run_classifier(Xtrain, Ytrain, SVC_cls)
    
    # 3.1.2 Setting C

    # lst = [0.1, 0.3, 0.5, 0.8, 1, 3, 5]
    # for i in lst:
    #     SVC_cls = Pipeline([('vec', vec),
    #                            ('cls', SVC(kernel='linear', C=i))])
    #     run_classifier(Xtrain, Ytrain, SVC_cls)

    # parameters = {'cls__C':[1, 5, 10]}
    # run_classifier(Xtrain, Ytrain, SVC_cls, parameters)
    

    # 3.1.3 Using a non-linear kernel

    # parameters = {'cls__kernel':('linear', 'rbf'),
    #               'cls__gamma':(0.1, 0.5, 1.0)     
    #              }
    # classifier = Pipeline([('vec', vec),
    #                    ('cls', SVC())])
    
    # run_classifier(Xtrain, Ytrain, classifier, parameters)



    # 3.1.4 Implementation differences
    
    # Running default SVC classifier.
    # run_classifier(Xtrain, Ytrain, SVC())
    
    # Running default LinearSVC classifier.
    # run_classifier(Xtrain, Ytrain, SVC())

    # 3.1.5 Best SVM model
    # Running SVC classifier with best GridSearch hyperparameters.
    # parameters = {'cls__kernel':('linear', 'rbf'),
    #               'cls__gamma':('scale', 'auto'),
    #               'cls__C': [1, 5]  
    #              }
    # classifier = Pipeline([('vec', vec),
    #                        ('cls', SVC())])
    # tweaked_cls = run_classifier(Xtrain, Ytrain, classifier, parameters)
    
    # Running LinearSVC classifier with best GridSearch hyperparameters.
    # parameters = {'cls__penalty': ('l2',),
    #               'cls__max_iter': (3000, 6000),
    #               'cls__C': (0.1, 0.5, 1.0)
    #              }
    # classifier = Pipeline([('vec', vec),
    #                        ('cls', LinearSVC())])
    # tweaked_cls2 = run_classifier(Xtrain, Ytrain, classifier, parameters)

    # classifier = Pipeline([('vec', vec),
    #                        ('cls', SVC(C=0.02, gamma='scale', kernel='linear'))])
    # run_classifier(Xtrain, Ytrain, classifier)

    classifier = Pipeline([('vec', vec),
                       ('cls', LinearSVC(C=0.02, fit_intercept=True))])
    run_classifier(Xtrain, Ytrain, classifier)

    show_results(classifier, Xtest, Ytest)

if __name__ == '__main__':
    main(sys.argv)