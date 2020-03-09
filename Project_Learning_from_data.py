#lfd project group 4

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import KFold
from statistics import mean
from sklearn.pipeline import FeatureUnion
from sklearn.svm import SVC
import numpy as np
import sys
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


stopWords = set(stopwords.words("english"))


def run_gridsearch(classifier, parameters, Xtrain, Ytrain, Xtest, Ytest):
    """Returns best parameter for classifier."""
    print("### USING GRID-SEARCH")
    clf = GridSearchCV(classifier, parameters, cv=5, n_jobs=-1)
    clf.fit(Xtrain, Ytrain)
    Yguess2 = clf.predict(Xtest)
    print(accuracy_score(Ytest, Yguess2))
    print(clf.best_params_)
    return clf.best_params_, True



def tokenize_pos(tokens):
    return [token+"_POS-"+tag for token, tag in nltk.pos_tag(tokens)]


#A dummy function that just returns its input
def identity(x):
    return x
  

def get_count_vectorizer():
    """Retuns a CountVectorizer"""
    vectorizer = CountVectorizer(stop_words=stopWords, preprocessor = identity, ngram_range=(1, 3))

    #CountVectorizer(stop_words=stopWords,ngram_range=(3,3))
    return vectorizer


def get_tfidf_vectorizer():
    """Returns a TfidfVectorizer."""
    vectorizer = TfidfVectorizer(stop_words=stopWords,preprocessor=identity,ngram_range=(1, 3))

                                #Use min_df to ignore the terms that appear in less than 0.1% documents 
                                #min_df=0.001, 
                                #Use max_df to ignore the terms that appear in more than 1000 documents
                                #max_df=1000,
                                #Use smooth_idf to allow words not seen in training data to be processed
                                #smooth_idf='18',
                                #Keep other parameters default  as this seems to lead to the best score
    return vectorizer

def run_classifier(X, Y, classifier, parameters=False, best_parameters=False, pre=False):
    """Divides the training data in k-folds subsets. Each time, one of the k
    subsets is used as the test set/ validation set and the other k-1 subsets are
    put together to form a training set. This significantly reduces bias as we are
    using most of the data for fitting, and also significantly reduces variance as
    most of the data is also being used in validation set. It also prints,
    confusion matrix, classification report, accuracy and the average of all
    accuracies"""
    c = 0

    accuracies = []
    f1scores = []
    kf = KFold(n_splits=5) # Define the split - into 10 folds 
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
        
        t0 = time.time()
        classifier.fit(Xtrain, Ytrain)
        train_time = time.time() - t0
        print("training time = ", train_time)
        Yguess = classifier.predict(Xtest)
        accuracies.append(accuracy_score(Ytest, Yguess))
        f1scores.append(f1_score(Ytest, Yguess, average='macro')) 
    
    print("Average accuracy classifier: ", mean(accuracies))
    print("Average macro f1-score classifier: ", mean(f1scores))
    print(classification_report(Ytest,Yguess))
    return Yguess, Ytest

def unison_shuffled_copies(a, b,c):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p], c[p]

def main():

    #read command line arguments
    train_set = ''.join(sys.argv[1])
    test_set = ''.join(sys.argv[2])

    train = pd.read_csv(train_set,compression='xz',sep='\t',encoding='utf-8',index_col=0).dropna()
    test = pd.read_csv(test_set,compression='xz',sep='\t',encoding='utf-8',index_col=0).dropna()


    #initialize the count vectorizer
    count_vec = get_count_vectorizer()
    vec = count_vec
    
    ### IF YOU WANT TO COMBINE FEATURES ###
    #tfidf_vec = get_tfidf_vectorizer()
    #vec = FeatureUnion([("count", count_vec), ("tfidf", tfidf_vec)])

    #preparing all needed data
    text_array = np.asarray(train.text)
    b_label_array = np.asarray(train.hyperp,dtype='str')
    m_label_array = np.asarray(train.bias,dtype='str')
    test_text = np.asarray(test.text)
    
    b_test_labels = np.asarray(test.hyperp,dtype='str')
    m_test_labels = np.asarray(test.bias,dtype='str')
    id_array = np.asarray(test.id)

    ### COUNTVECTORIZER LOGISTIC REGRESSION ###
    pipeline = Pipeline([('vec', vec), ('clf', LogisticRegression(penalty='l2',multi_class='auto',solver='liblinear'))])

    #shuffle data if needed (k-fold)
    #text_array, b_label_array, m_label_array = unison_shuffled_copies(text_array,b_label_array,m_label_array)
    
    #if you want to run k-fold for development    
    #bYguess, b_test_labels = run_classifier(text_array, b_label_array, pipeline)
    #mYguess, m_test_labels = run_classifier(text_array, m_label_array, pipeline)
    

    ### BINARY CLASSIFICATION ###
    t0 = time.time()
    model = pipeline.fit(text_array,b_label_array)
    train_time = time.time() - t0
    print("training time = ", train_time)
    bYguess = model.predict(test_text)
    print(classification_report(b_test_labels,bYguess))
    print("Accuracy Score binary: ",accuracy_score(b_test_labels,bYguess))
    
    ### MULTI-CLASS CLASSIFICATION ###
    t0 = time.time()
    model = pipeline.fit(text_array,m_label_array)
    train_time = time.time() - t0
    print("training time = ", train_time)
    mYguess = model.predict(test_text)
    print(classification_report(m_test_labels,mYguess))
    print("Accuracy Score multi-class: ",accuracy_score(m_test_labels,mYguess))

    #calculating & printing the scores on the combined labels
    combined_gold = [i +' '+ j for i, j in zip(b_test_labels, m_test_labels)]
    combined_pred = [i +' '+ j for i, j in zip(bYguess, mYguess)]

    print(classification_report(combined_gold,combined_pred,labels=['True left', 'True right', 'False left-center', 'False least', 'False right-center']))
    print("Accuracy Score both labels: ",accuracy_score(combined_gold,combined_pred))

    #Writing output file
    out_data=[]
    f = open('predictions.txt', 'w')
    for ids,prediction in zip(id_array,combined_pred):
        pred = str(ids) + ' ' + prediction + '\n'
        f.write(pred)
    f.close()

if __name__ == '__main__':
    main()