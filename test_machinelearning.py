from sklearn import metrics
from sklearn.cross_validation import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm.classes import LinearSVC

import bamman_dataset
import fnmps_dataset
import numpy as np
import riloff_dataset
import wallace_dataset


def _get_entries(a_list, indices):
    return [a_list[i] for i in indices]


def get_data(dataset_name, affective_norms=False, author=False, subreddit=False,punctuation=False, emoticons=False):
    if dataset_name == 'wallace':
        return wallace_dataset.dataset().get_data(affective_norms=affective_norms, punctuation=punctuation, emoticons=emoticons)
    elif dataset_name == 'fnmps':
        return fnmps_dataset.dataset().get_data(author=author, subreddit=subreddit, affective_norms=affective_norms, punctuation=punctuation, emoticons=emoticons)
    elif dataset_name == 'riloff':
        return riloff_dataset.dataset().get_data(affective_norms=affective_norms, punctuation=punctuation, emoticons=emoticons)
    elif dataset_name == 'bamman':
        return bamman_dataset.dataset().get_data(affective_norms=affective_norms, punctuation=punctuation, emoticons=emoticons)
    
def classify(X, targets):
    
    svm_y_tests = []
    svm_predicted = []
    for train, test in kf:
        X_train, X_test = X[train], X[test]
        y_train = _get_entries(targets, train)
        y_test = _get_entries(targets, test)
         
        svm = SGDClassifier(loss="hinge", penalty="l2", class_weight="balanced", alpha=.01)
        parameters = {'alpha':[.001, .01,  .1]}
        clf = GridSearchCV(svm, parameters, scoring='f1')
        clf.fit(X_train, y_train)
         
        predicted = clf.predict(X_test)
         
        svm_predicted = svm_predicted + predicted.tolist()
        svm_y_tests = svm_y_tests + y_test

    
    #===============================================================================
    
    nb_y_tests = []
    nb_predicted = []
    for train, test in kf:
        X_train, X_test = X[train], X[test]
        y_train = _get_entries(targets, train)
        y_test = _get_entries(targets, test)
         
        nb = MultinomialNB()
        parameters = {'alpha':[.001, .01,  .1]}
        clf = GridSearchCV(nb, parameters, scoring='f1')
        clf.fit(X_train, y_train)
         
        predicted = clf.predict(X_test)
         
        nb_predicted = nb_predicted + predicted.tolist()
        nb_y_tests = nb_y_tests + y_test
       
     
    
    #===============================================================================
    
    log_y_tests = []
    log_predicted = []
    for train, test in kf:
        X_train, X_test = X[train], X[test]
        y_train = _get_entries(targets, train)
        y_test = _get_entries(targets, test)
         
        log = LogisticRegression(penalty="l2", class_weight="balanced")
        parameters = {}
        clf = GridSearchCV(log, parameters, scoring='f1')
        clf.fit(X_train, y_train)
         
        predicted = clf.predict(X_test)
        log_predicted = log_predicted + predicted.tolist()
        log_y_tests = log_y_tests + y_test
      
    
    nbsvm_y_tests = []
    nbsvm_predicted = []
    for train, test in kf:
        X_train, X_test = X[train], X[test]
        y_train = _get_entries(targets, train)
        y_test = _get_entries(targets, test)         
        model = MultinomialNB( fit_prior=False )
        model.fit( X_train , y_train )
        X_train = np.hstack( (X_train.toarray(), model.predict_proba( X_train ) ) )
        X_test = np.hstack( (X_test.toarray(), model.predict_proba( X_test ) ) )
        model = SGDClassifier(loss="hinge", penalty="l2", class_weight="balanced", alpha=.01)
        model.fit( X_train , y_train )
        predicted = model.predict( X_test )
        nbsvm_predicted = nbsvm_predicted + predicted.tolist()
        nbsvm_y_tests = nbsvm_y_tests + y_test
    

    
    return (svm_predicted, svm_y_tests),(nb_predicted, nb_y_tests),(log_predicted, log_y_tests),(nbsvm_predicted, nbsvm_y_tests)

#===============================================================================

#features to use
# maximum number of words to consider in the representations
max_features = 30000
affective_norms = False
author = False
subreddit = False
emoticons = False
punctuation = False
np.random.seed(seed=1234)
print("Features being used:")
print("affective_norms = " + str(affective_norms) + ", author = " + str(author) + ", subreddit = " + str(subreddit) + ", emoticons = " + str(emoticons) + ", punctuation = " + str(punctuation))
print("=========================")
print("USING WALLACE DATA...")

comments, targets = get_data('wallace', affective_norms=affective_norms, punctuation=punctuation, emoticons=emoticons)

vectorizer = TfidfVectorizer(ngram_range=(1,2), binary=True, stop_words="english", max_features=max_features)
X = vectorizer.fit_transform(comments)
kf = KFold(len(targets), n_folds=5, shuffle=True)

(svm_result, svm_tests),(nb_result, nb_tests),(log_result, log_tests),(nbsvm_result, nbsvm_tests) = classify(X, targets)

print "--------SVM-----------"
print(metrics.classification_report(svm_tests, svm_result, target_names=["ironic", "not-ironic"]))
print "accuracy: " + str(metrics.accuracy_score(svm_tests, svm_result) )
print "---------------------"

print "-------NB----------"
print(metrics.classification_report(nb_tests, nb_result, target_names=["ironic", "not-ironic"]))
print "accuracy: " + str(metrics.accuracy_score(nb_tests, nb_result) )
print "-------------------"
    
print "------LOG----------"
print(metrics.classification_report(log_tests, log_result, target_names=["ironic", "not-ironic"]))
print "accuracy: " + str(metrics.accuracy_score(log_tests, log_result) )
print "-------------------"

print "------NB-SVM----------"
print(metrics.classification_report(nbsvm_tests, nbsvm_result, target_names=["ironic", "not-ironic"]))
print "accuracy: " + str(metrics.accuracy_score(nbsvm_tests, nbsvm_result) )
print "-------------------"
#===============================================================

print("=========================")
print("USING RILOFF DATA...")
 
comments, targets = get_data('riloff', affective_norms=affective_norms, punctuation=punctuation, emoticons=emoticons)
 
vectorizer = TfidfVectorizer(ngram_range=(1,2), binary=True, stop_words="english", max_features=max_features)
X = vectorizer.fit_transform(comments)
kf = KFold(len(targets), n_folds=5, shuffle=True)
 
(svm_result, svm_tests),(nb_result, nb_tests),(log_result, log_tests),(nbsvm_result, nbsvm_tests) = classify(X, targets)
 
print "--------SVM-----------"
print(metrics.classification_report(svm_tests, svm_result, target_names=["ironic", "not-ironic"]))
print "accuracy: " + str(metrics.accuracy_score(svm_tests, svm_result) )
print "---------------------"
 
print "-------NB----------"
print(metrics.classification_report(nb_tests, nb_result, target_names=["ironic", "not-ironic"]))
print "accuracy: " + str(metrics.accuracy_score(nb_tests, nb_result) )
print "-------------------"
     
print "------LOG----------"
print(metrics.classification_report(log_tests, log_result, target_names=["ironic", "not-ironic"]))
print "accuracy: " + str(metrics.accuracy_score(log_tests, log_result) )
print "-------------------"
 
print "------NB-SVM----------"
print(metrics.classification_report(nbsvm_tests, nbsvm_result, target_names=["ironic", "not-ironic"]))
print "accuracy: " + str(metrics.accuracy_score(nbsvm_tests, nbsvm_result) )
print "-------------------"
#===============================================================


print("=========================")
print("USING BAMMAN DATA...")
 
tweets, targets = get_data('bamman', affective_norms=affective_norms, punctuation=punctuation, emoticons=emoticons)

vectorizer = TfidfVectorizer(ngram_range=(1,2), binary=True, stop_words="english", max_features=max_features)
X = vectorizer.fit_transform(tweets)

kf = KFold(len(targets), n_folds=5, shuffle=True)
 
(svm_result, svm_tests),(nb_result, nb_tests),(log_result, log_tests),(nbsvm_result, nbsvm_tests) = classify(X, targets)
 
print "--------SVM-----------"
print(metrics.classification_report(svm_tests, svm_result, target_names=["ironic", "not-ironic"]))
print "accuracy: " + str(metrics.accuracy_score(svm_tests, svm_result) )
print "---------------------"
 
print "-------NB----------"
print(metrics.classification_report(nb_tests, nb_result, target_names=["ironic", "not-ironic"]))
print "accuracy: " + str(metrics.accuracy_score(nb_tests, nb_result) )
print "-------------------"
     
print "------LOG----------"
print(metrics.classification_report(log_tests, log_result, target_names=["ironic", "not-ironic"]))
print "accuracy: " + str(metrics.accuracy_score(log_tests, log_result) )
print "-------------------"
 
print "------NB-SVM----------"
print(metrics.classification_report(nbsvm_tests, nbsvm_result, target_names=["ironic", "not-ironic"]))
print "accuracy: " + str(metrics.accuracy_score(nbsvm_tests, nbsvm_result) )
print "-------------------"
#===============================================================


# 
# print("=========================")
# print("USING FNMPS DATA...")
# 
# comments, targets = get_data('fnmps', author=author, subreddit=subreddit, affective_norms=affective_norms, punctuation=punctuation, emoticons=emoticons)
# 
# vectorizer = TfidfVectorizer(ngram_range=(1,2), binary=True, stop_words="english")
# X = vectorizer.fit_transform(comments)
# kf = KFold(len(targets), n_folds=5, shuffle=True)
# 
# (svm_result, svm_tests),(nb_result, nb_tests),(log_result, log_tests) = classify(X, targets)
# print "--------SVM-----------"
# print(metrics.classification_report(svm_tests, svm_result, target_names=["ironic", "not-ironic"]))
# print "accuracy: " + str(metrics.accuracy_score(svm_tests, svm_result) )
# print "---------------------"
# 
# print "-------NB----------"
# print(metrics.classification_report(nb_tests, nb_result, target_names=["ironic", "not-ironic"]))
# print "accuracy: " + str(metrics.accuracy_score(nb_tests, nb_result) )
# print "-------------------"
#     
# print "------LOG----------"
# print(metrics.classification_report(log_tests, log_result, target_names=["ironic", "not-ironic"]))
# print "accuracy: " + str(metrics.accuracy_score(log_tests, log_result) )
# print "-------------------"
#===============================================================


