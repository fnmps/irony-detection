from __future__ import print_function

import random

from gensim.models.word2vec import Word2Vec
from keras.layers import Embedding
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.core import Dense , Dropout , Activation, Flatten
from keras.layers.recurrent import LSTM
from keras.models import Graph, Sequential
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
import keras.preprocessing.text
from sklearn import preprocessing
from sklearn.cross_validation import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.metrics
import theano.sandbox.cuda

import fnmps_dataset
import numpy as np
import riloff_dataset
import wallace_dataset


#===================================
def _get_entries(a_list, indices):
    return [a_list[i] for i in indices]

def get_data(dataset_name, affective_norms=False, author=False, subreddit=False,punctuation=False, emoticons=False):
    if dataset_name == 'wallace':
        comments, targets = wallace_dataset.dataset().get_data(affective_norms=affective_norms, punctuation=punctuation, emoticons=emoticons)
    elif dataset_name == 'fnmps':
        comments, targets =  fnmps_dataset.dataset().get_data(author=author, subreddit=subreddit, affective_norms=affective_norms, punctuation=punctuation, emoticons=emoticons)
    elif dataset_name == 'riloff':
        comments, targets =  riloff_dataset.dataset().get_data(affective_norms=affective_norms, punctuation=punctuation, emoticons=emoticons)
    return zip(comments, targets)



data = get_data('riloff', affective_norms=False, author=False, subreddit=False)
#===============================================================

print ("")
print ("Reading pre-trained word embeddings...")
embeddings = dict( )
embeddings = Word2Vec.load_word2vec_format( "GoogleNews-vectors-negative300.bin" , binary=True ) 

# size of the word embeddings
embeddings_dim = 300

# maximum number of words to consider in the representations
max_features = 30000

# maximum length of a sentence
max_sent_len = 50

# percentage of the data used for model training
percent = 0.75

# number of classes
num_classes = 2


batch_size = 64
nb_epoch = 30
nb_folds = 5

random.shuffle( data )
train_size = int(len(data) * percent)
train_texts = [ txt.lower() for ( txt, label ) in data[0:train_size] ]
test_texts = [ txt.lower() for ( txt, label ) in data[train_size:-1] ]
train_labels = [ label for ( txt , label ) in data[0:train_size] ]
test_labels = [ label for ( txt , label ) in data[train_size:-1] ]
num_classes = len( set( train_labels + test_labels ) )
tokenizer = Tokenizer(nb_words=max_features, filters=keras.preprocessing.text.base_filter(), lower=True, split=" ")
tokenizer.fit_on_texts(train_texts)
train_sequences = sequence.pad_sequences( tokenizer.texts_to_sequences( train_texts ) , maxlen=max_sent_len )
test_sequences = sequence.pad_sequences( tokenizer.texts_to_sequences( test_texts ) , maxlen=max_sent_len )
train_matrix = tokenizer.texts_to_matrix( train_texts, mode='tfidf' )
test_matrix = tokenizer.texts_to_matrix( test_texts, mode='tfidf' )
embedding_weights = np.zeros( ( max_features , embeddings_dim ) )
for word,index in tokenizer.word_index.items():
    if index < max_features:
        try: embedding_weights[index,:] = embeddings[word]
        except: embedding_weights[index,:] = np.random.rand( 1 , embeddings_dim )
le = preprocessing.LabelEncoder( )
le.fit( train_labels + test_labels )
train_labels = le.transform( train_labels )
test_labels = le.transform( test_labels )
print ("Classes that are considered in the problem : " + repr( le.classes_ ))


theano.sandbox.cuda.use("gpu0")
print ("Method = Stack of two LSTMs")
np.random.seed(0)

# vectorizer = TfidfVectorizer(ngram_range=(1,2), binary=True, stop_words="english", max_features=max_features)
# X = vectorizer.fit_transform([x[0] for x in data ])
# y = [x[1] for x in data ]

X = np.vstack( (train_sequences,test_sequences) )
y = np.hstack( (train_labels,  test_labels) )
predictions = []
result_labels = []
kf = KFold(len(y), n_folds=nb_folds, shuffle=True)
fold_no = 0
for train, test in kf:
    fold_no += 1
    print("kfold: " + str(fold_no) + "/5")
    model = Sequential()
    model.add(Embedding(max_features, embeddings_dim, input_length=max_sent_len, mask_zero=True, weights=[embedding_weights] ))
    model.add(Dropout(0.25))
    model.add(LSTM(output_dim=embeddings_dim , activation='sigmoid', inner_activation='hard_sigmoid', return_sequences=True))
    model.add(Dropout(0.25))
    model.add(LSTM(output_dim=embeddings_dim , activation='sigmoid', inner_activation='hard_sigmoid'))
    model.add(Dropout(0.25))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    if num_classes == 2: model.compile(loss='binary_crossentropy', optimizer='adam', class_mode='binary')
    else: model.compile(loss='categorical_crossentropy', optimizer='adam')
    X_train, X_test = X[train], X[test]
    y_train = _get_entries(y, train)
    y_test = _get_entries(y, test)
    model.fit( X_train , y_train , nb_epoch=nb_epoch, batch_size=batch_size)
    results = model.predict_classes( X_test )
    predictions = predictions + results.tolist()
    result_labels = result_labels + y_test
print ("Accuracy = " + repr( sklearn.metrics.accuracy_score( result_labels , predictions )  ))
print (sklearn.metrics.classification_report( result_labels , np.asarray(predictions) ))
 
 
print ("Method = MLP with bag-of-words features")
np.random.seed(0)
X = np.vstack( (train_matrix, test_matrix) )
y = np.hstack( (train_labels,  test_labels) )
predictions = []
result_labels = []
kf = KFold(len(y), n_folds=nb_folds, shuffle=True)
fold_no = 0
for train, test in kf:
    fold_no += 1
    print("kfold: " + str(fold_no) + "/5")
    model = Sequential()
    model.add(Dense(embeddings_dim, input_dim=train_matrix.shape[1], init='uniform', activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(embeddings_dim, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(1, activation='sigmoid'))
    if num_classes == 2: model.compile(loss='binary_crossentropy', optimizer='adam', class_mode='binary')
    else: model.compile(loss='categorical_crossentropy', optimizer='adam')
    X_train, X_test = X[train], X[test]
    y_train = _get_entries(y, train)
    y_test = _get_entries(y, test)
    model.fit( X_train , y_train , nb_epoch=nb_epoch, batch_size=batch_size)
    results = model.predict_classes( X_test )
    predictions = predictions + results.tolist()
    result_labels = result_labels + y_test
print ("Accuracy = " + repr( sklearn.metrics.accuracy_score( result_labels , predictions )  ))
print (sklearn.metrics.classification_report( result_labels , np.asarray(predictions) ))
  
  
print ("Method = CNN from the paper 'Convolutional Neural Networks for Sentence Classification'")
np.random.seed(0)
nb_filter = embeddings_dim
X = np.vstack( (train_sequences,test_sequences) )
y = np.hstack( (train_labels,  test_labels) )
predictions = []
result_labels = []
kf = KFold(len(y), n_folds=nb_folds, shuffle=True)
fold_no = 0
for train, test in kf:
    fold_no += 1
    print("kfold: " + str(fold_no) + "/5")
    model = Graph()
    model.add_input(name='input', input_shape=(max_sent_len,), dtype=int)
    model.add_node(Embedding(max_features, embeddings_dim, input_length=max_sent_len, mask_zero=False, weights=[embedding_weights] ), name='embedding', input='input')
    model.add_node(Dropout(0.25), name='dropout_embedding', input='embedding')
    for n_gram in [3, 5, 7]:
        model.add_node(Convolution1D(nb_filter=nb_filter, filter_length=n_gram, border_mode='valid', activation='relu', subsample_length=1, input_dim=embeddings_dim, input_length=max_sent_len), name='conv_' + str(n_gram), input='dropout_embedding')
        model.add_node(MaxPooling1D(pool_length=max_sent_len - n_gram + 1), name='maxpool_' + str(n_gram), input='conv_' + str(n_gram))
        model.add_node(Flatten(), name='flat_' + str(n_gram), input='maxpool_' + str(n_gram))
    model.add_node(Dropout(0.25), name='dropout', inputs=['flat_' + str(n) for n in [3, 5, 7]])
    model.add_node(Dense(1, input_dim=nb_filter * len([3, 5, 7])), name='dense', input='dropout')
    model.add_node(Activation('sigmoid'), name='sigmoid', input='dense')
    model.add_output(name='output', input='sigmoid')
    if num_classes == 2: model.compile(loss={'output': 'binary_crossentropy'}, optimizer='adam')
    else: model.compile(loss={'output': 'categorical_crossentropy'}, optimizer='adam') 
    model.fit({'input': train_sequences, 'output': train_labels}, batch_size=batch_size, nb_epoch=nb_epoch)
    results = np.array(model.predict({'input': test_sequences}, batch_size=batch_size)['output'])
    if num_classes != 2: results = results.argmax(axis=-1)
    else: results = (results > 0.5).astype('int32')
    predictions = predictions + results.tolist()
    result_labels = result_labels + y_test
print ("Accuracy = " + repr( sklearn.metrics.accuracy_score( result_labels , predictions )  ))
print (sklearn.metrics.classification_report( result_labels , np.asarray(predictions) ))
  
  
print ("Method = Bidirectional LSTM")
np.random.seed(0)
X = np.vstack( (train_sequences,test_sequences) )
y = np.hstack( (train_labels,  test_labels) )
predictions = []
result_labels = []
kf = KFold(len(y), n_folds=nb_folds, shuffle=True)
fold_no = 0
for train, test in kf:
    fold_no += 1
    print("kfold: " + str(fold_no) + "/5")
    model = Graph()
    model.add_input(name='input', input_shape=(max_sent_len,), dtype=int)
    model.add_node(Embedding( max_features, embeddings_dim, input_length=max_sent_len, mask_zero=True, weights=[embedding_weights] ), name='embedding', input='input')
    model.add_node(LSTM(embeddings_dim, activation='sigmoid', inner_activation='hard_sigmoid', return_sequences=True), name='forward1', input='embedding')
    model.add_node(Dropout(0.25), name="dropout1", input='forward1')
    model.add_node(LSTM(embeddings_dim, activation='sigmoid', inner_activation='hard_sigmoid'), name='forward2', input='forward1')
    model.add_node(LSTM(embeddings_dim, activation='sigmoid', inner_activation='hard_sigmoid', go_backwards=True, return_sequences=True), name='backward1', input='embedding')
    model.add_node(Dropout(0.25), name="dropout2", input='backward1') 
    model.add_node(LSTM(embeddings_dim, activation='sigmoid', inner_activation='hard_sigmoid', go_backwards=True), name='backward2', input='backward1')
    model.add_node(Dropout(0.25), name='dropout', inputs=['forward2', 'backward2'])
    model.add_node(Dense(1, activation='sigmoid'), name='sigmoid', input='dropout')
    model.add_output(name='output', input='sigmoid')
    if num_classes == 2: model.compile(loss={'output': 'binary_crossentropy'}, optimizer='adam')
    else: model.compile(loss={'output': 'categorical_crossentropy'}, optimizer='adam')
    model.fit({'input': train_sequences, 'output': train_labels}, batch_size=batch_size, nb_epoch=nb_epoch)
    results = np.array(model.predict({'input': test_sequences}, batch_size=batch_size)['output'])
    if num_classes != 2: results = results.argmax(axis=-1)
    else: results = (results > 0.5).astype('int32')
    predictions = predictions + results.tolist()
    result_labels = result_labels + y_test
print ("Accuracy = " + repr( sklearn.metrics.accuracy_score( result_labels , predictions )  ))
print (sklearn.metrics.classification_report( result_labels , np.asarray(predictions) ))
  
print ("Method = CNN-LSTM")
np.random.seed(0)
filter_length = 3
nb_filter = embeddings_dim
pool_length = 2
X = np.vstack( (train_sequences,test_sequences) )
y = np.hstack( (train_labels,  test_labels) )
predictions = []
result_labels = []
kf = KFold(len(y), n_folds=nb_folds, shuffle=True)
fold_no = 0
for train, test in kf:
    fold_no += 1
    print("kfold: " + str(fold_no) + "/5")
    model = Sequential()
    model.add(Embedding(max_features, embeddings_dim, input_length=max_sent_len, weights=[embedding_weights]))
    model.add(Dropout(0.25))
    model.add(Convolution1D(nb_filter=nb_filter, filter_length=filter_length, border_mode='valid', activation='relu', subsample_length=1))
    model.add(MaxPooling1D(pool_length=pool_length))
    model.add(LSTM(embeddings_dim))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    if num_classes == 2: model.compile(loss='binary_crossentropy', optimizer='adam', class_mode='binary')
    else: model.compile(loss='categorical_crossentropy', optimizer='adam')  
    model.fit( train_sequences , train_labels , nb_epoch=nb_epoch, batch_size=batch_size)
    results = model.predict_classes( test_sequences )
    predictions = predictions + results.tolist()
    result_labels = result_labels + y_test
print ("Accuracy = " + repr( sklearn.metrics.accuracy_score( result_labels , predictions )  ))
print (sklearn.metrics.classification_report( result_labels , np.asarray(predictions) ))

