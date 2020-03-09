#!/usr/bin/python3
# Author: J.F.P. (Richard) Scholtens
# Studentnr: s2956586
# Date: 30/09/2019

import gensim
from gensim.models import word2vec
import numpy, json, argparse
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt
from collections import Counter
from sklearn.metrics import confusion_matrix


numpy.random.seed(1337)

def show_plot(matrix, xlabel):
    """This function shows a plot of a matrix which has two rows."""

    # Creates plot
    plt.title("Accuracy vs Loss ")
    plt.plot(matrix[0], matrix[1], label="Loss")
    plt.plot(matrix[0], matrix[2], label="Accuracy")
    plt.xlabel(xlabel)
    plt.ylabel("Accuracy/Loss")
    plt.legend()
    plt.show()


# Read in the NE data, with either 2 or 6 classes
def read_corpus(corpus_file, binary_classes):
    print('Reading in data from {0}...'.format(corpus_file))
    words = []
    labels = []
    with open(corpus_file, encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            words.append(parts[0])
            if binary_classes:
                if parts[1] in ['GPE', 'LOC']:
                    labels.append('LOCATION')
                else:
                    labels.append('NON-LOCATION')
            else:
                labels.append(parts[1]) 
    print('Done!')
    return words, labels

# Read in word embeddings 
def read_embeddings(embeddings_file):
    print('Reading in embeddings from {0}...'.format(embeddings_file))
    embeddings = json.load(open(embeddings_file, 'r'))
    embeddings = {word:numpy.array(embeddings[word]) for word in embeddings}
    print('Done!')
    return embeddings

# Turn words into embeddings, i.e. replace words by their corresponding embeddings
def vectorizer(words, embeddings):
    vectorized_words = []
    for word in words:
        try:
            vectorized_words.append(embeddings[word.lower()])
        except KeyError:
            vectorized_words.append(embeddings['UNK'])
    return numpy.array(vectorized_words)

# Prints the most common label for a vector.
def print_label(model, vector):
    a = encoder.inverse_transform(model.predict(vector))
    b = Counter(a)
    print(b.most_common()[0][0])

# Pretty print for confusion matrixes.
def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """"""
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth
    
    # Begin CHANGES
    fst_empty_cell = (columnwidth-3)//2 * " " + "t/p" + (columnwidth-3)//2 * " "
    
    if len(fst_empty_cell) < len(empty_cell):
        fst_empty_cell = " " * (len(empty_cell) - len(fst_empty_cell)) + fst_empty_cell
    # Print header
    print("    " + fst_empty_cell, end=" ")
    # End CHANGES
    
    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ")
        
    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=" ")
        for j in range(len(labels)):
            cell = "%{0}.1f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=" ")
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='KerasNN parameters')
    parser.add_argument('data', metavar='named_entity_data.txt', type=str, help='File containing named entity data.')
    parser.add_argument('embeddings', metavar='embeddings.json', type=str, help='File containing json-embeddings.')
    parser.add_argument('-b', '--binary', action='store_true', help='Use binary classes.')
    args = parser.parse_args()
    
    # Read in the data and embeddings
    X, Y = read_corpus(args.data, binary_classes = args.binary)
    embeddings = read_embeddings(args.embeddings)

    # Transform words to embeddings
    X = vectorizer(X, embeddings)

    # Transform string labels to one-hot encodings
    encoder = LabelBinarizer()

    Y = encoder.fit_transform(Y) # Use encoder.classes_ to find mapping of one-hot indices to string labels
    if args.binary:
        Y = numpy.where(Y == 1, [0,1], [1,0])

    # Split in training and test data
    split_point = int(0.75*len(X))
    Xtrain = X[:split_point]
    Ytrain = Y[:split_point]
    Xtest = X[split_point:]
    Ytest = Y[split_point:]

    # Creating a baseline.
    # dummy = DummyClassifier(strategy='stratified')
    # dummy.fit(Xtrain, Ytrain)
    # Yguess = dummy.predict(Xtest)
    # print("### BASELINE")
    # print("Accuracy score of baseline: ", accuracy_score(Ytest, Yguess))
    # print(classification_report(Ytest, Yguess))

    # Define the properties of the perceptron model
    model = Sequential()
    model.add(Dense(input_dim=X.shape[1], units=Y.shape[1]))
    model.add(Activation("tanh"))
    sgd = SGD(lr=0.01)
    loss_function = 'mean_squared_error'
    model.compile(loss=loss_function, optimizer=sgd, metrics=['accuracy'])

    # Train the perceptron

    # ### CHECKING EPOCHS
    # e = 20
    # history = model.fit(Xtrain, Ytrain, verbose=1, epochs=e, batch_size=32)

    # # Get predictions
    # Yguess = model.predict(Xtest)

    # # Convert to numerical labels to get scores with sklearn in 6-way setting
    # Yguess = numpy.argmax(Yguess, axis = 1)
    # Ytest = numpy.argmax(Ytest, axis = 1)

    # ### PRINTING RESULTS
    # print('Classification accuracy on test: {0}'.format(accuracy_score(Ytest, Yguess)))
    # print(classification_report(Ytest, Yguess))

    # ### MAKE ACCURACY / LOSS PLOT
    # acc = numpy.array(history.history['accuracy'])
    # loss = numpy.array(history.history['loss'])
    # matrix = [[(i + 1 ) for i in range(e)], loss, acc]
    # show_plot(matrix, "Number of Epochs")

    # ### MAKE BATCHSIZE PLOT
    # batchsizes = [8, 16, 32, 64, 128, 256]
    # acc_lst = []
    # loss_lst = []
    # h1 = model.fit(Xtrain, Ytrain, verbose = 1, epochs = 1, batch_size = batchsizes[0])
    # h2 = model.fit(Xtrain, Ytrain, verbose = 1, epochs = 1, batch_size = batchsizes[1])
    # h3 = model.fit(Xtrain, Ytrain, verbose = 1, epochs = 1, batch_size = batchsizes[2])
    # h4 = model.fit(Xtrain, Ytrain, verbose = 1, epochs = 1, batch_size = batchsizes[3])
    # h5 = model.fit(Xtrain, Ytrain, verbose = 1, epochs = 1, batch_size = batchsizes[4])
    # h6 = model.fit(Xtrain, Ytrain, verbose = 1, epochs = 1, batch_size = batchsizes[5])
    # a1 = numpy.array(h1.history['accuracy'])
    # a2 = numpy.array(h2.history['accuracy'])
    # a3 = numpy.array(h3.history['accuracy'])
    # a4 = numpy.array(h4.history['accuracy'])
    # a5 = numpy.array(h5.history['accuracy'])
    # a6 = numpy.array(h6.history['accuracy'])
    # acc_lst.append(numpy.mean(a1))
    # acc_lst.append(numpy.mean(a2))
    # acc_lst.append(numpy.mean(a3))
    # acc_lst.append(numpy.mean(a4))
    # acc_lst.append(numpy.mean(a5))
    # acc_lst.append(numpy.mean(a6))
    # l1 = numpy.array(h1.history['loss'])
    # l2 = numpy.array(h2.history['loss'])
    # l3 = numpy.array(h3.history['loss'])
    # l4 = numpy.array(h4.history['loss'])
    # l5 = numpy.array(h5.history['loss'])
    # l6 = numpy.array(h6.history['loss'])
    # loss_lst.append(numpy.mean(l1))
    # loss_lst.append(numpy.mean(l2))
    # loss_lst.append(numpy.mean(l3))
    # loss_lst.append(numpy.mean(l4))
    # loss_lst.append(numpy.mean(l5))
    # loss_lst.append(numpy.mean(l6))    
    # acc = numpy.array(history.history['accuracy'])
    # loss = numpy.array(history.history['loss'])
    # matrix = [batchsizes, loss_lst, acc_lst]
    # show_plot(matrix, "Batchsize")


    # ### CHECKING ACTIVATION FUNCTION
    # e = 20
    # history = model.fit(Xtrain, Ytrain, verbose = 1, epochs = 20, batch_size = 32)

    # # Get predictions
    # Yguess = model.predict(Xtest)

    # # Convert to numerical labels to get scores with sklearn in 6-way setting
    # Yguess = numpy.argmax(Yguess, axis = 1)
    # Ytest = numpy.argmax(Ytest, axis = 1)

    # print('Classification accuracy on test: {0}'.format(accuracy_score(Ytest, Yguess)))
    # print(classification_report(Ytest, Yguess))
    # acc = numpy.array(history.history['accuracy'])
    # loss = numpy.array(history.history['loss'])
    # matrix = [[(i + 1 ) for i in range(e)], loss, acc]
    # show_plot(matrix, "Number of epochs")


    model.fit(Xtrain, Ytrain, verbose=1, epochs=10, batch_size=32)
    # Specific labeling
    # wb1 = vectorizer("Antwerpen", embeddings)
    # wb2 = vectorizer("KFC", embeddings)
    # wb3 = vectorizer("thirteen", embeddings)
    # print_label(model, wb1)
    # print_label(model, wb2)
    # print_label(model, wb3)

    Yguess = model.predict(Xtest)

    # Convert to numerical labels to get scores with sklearn in 6-way setting
    Yguess = numpy.argmax(Yguess, axis=1)
    Ytest = numpy.argmax(Ytest, axis=1)

    labels = ['CARDINAL', 'DATE', 'GPE', 'LOC', 'ORG', 'PERSON']
    cm = confusion_matrix(Ytest, Yguess)
    print("\n\n")
    print_cm(cm, labels)
