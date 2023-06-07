#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 21:33:41 2023

@author: hbonen
"""

import pandas as pd
import regex as re
import numpy as np
from sklearn import model_selection, svm
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import itertools

def tokenize_link(url:str):   
    url=url.replace("https://www.nhm.ac.uk/discover/dino-directory/","")
    url=re.sub("(\W|_)+"," ",url)
    return url

def extract_features(field,training_data,testing_data,type="binary"):
    """Extract features using different methods"""
    
    
    if "binary" in type:
        
        # BINARY FEATURE REPRESENTATION
        cv= CountVectorizer(binary=True, max_df=0.95)
        cv.fit_transform(training_data[field].values)
        
        train_feature_set=cv.transform(training_data[field].values)
        test_feature_set=cv.transform(testing_data[field].values)
        
        return train_feature_set,test_feature_set,cv
  
    elif "counts" in type:

        cv= CountVectorizer(binary=False, max_df=0.95)
        cv.fit_transform(training_data[field].values)
        
        train_feature_set=cv.transform(training_data[field].values)
        test_feature_set=cv.transform(testing_data[field].values)
        
        return train_feature_set,test_feature_set,cv
    
    else:    
        
        tfidf_vectorizer=TfidfVectorizer(use_idf=True, max_df=0.95)
        tfidf_vectorizer.fit_transform(training_data[field].values)
        
        train_feature_set=tfidf_vectorizer.transform(training_data[field].values)
        test_feature_set=tfidf_vectorizer.transform(testing_data[field].values)
        
        return train_feature_set,test_feature_set,tfidf_vectorizer


def get_top_k_predictions(model,X_test,k):
    
    probs = model.predict_proba(X_test)

    best_n = np.argsort(probs, axis=1)[:,-k:]

    preds=[[model.classes_[predicted_cat] for predicted_cat in prediction] for prediction in best_n]

    preds=[ item[::-1] for item in preds]
        
    return preds

df = pd.read_csv(r'/Users/hbonen/Documents/Doktora/CMP712_Machine_Learning/hw2/train.csv')
df = df.fillna(',')

df['tokenized_link']=df['link'].apply(lambda x:tokenize_link(x))

df['txt_diet_period_livedIn_length_taxonomy_namedBy_species_link'] = df['name'] + ' ' + df['diet'] + ' ' + df['period'] + \
    ' ' + df['lived_in'] + ' ' + df['length'] + ' ' + df['taxonomy'] + ' ' + df['named_by'] + \
    ' ' + df['species'] + ' ' + df['tokenized_link']


dfTest = pd.read_csv(r'/Users/hbonen/Documents/Doktora/CMP712_Machine_Learning/hw2/test.csv')
dfTest = dfTest.fillna(',')

dfTest['tokenized_link']=dfTest['link'].apply(lambda x:tokenize_link(x))

dfTest['txt_diet_period_livedIn_length_taxonomy_namedBy_species_link'] = dfTest['name'] + ' ' +dfTest['diet'] + ' ' + dfTest['period'] + \
    ' ' + dfTest['lived_in'] + ' ' + dfTest['length'] + ' ' + dfTest['taxonomy'] + ' ' + dfTest['named_by'] + \
    ' ' + dfTest['species'] + ' ' + dfTest['tokenized_link']


training_data = df
testing_data = dfTest

Q = training_data['txt_diet_period_livedIn_length_taxonomy_namedBy_species_link']
z = training_data['type']
test_data = dfTest['txt_diet_period_livedIn_length_taxonomy_namedBy_species_link']

k = 10
skf = StratifiedKFold(n_splits = k)  #the number of folds is 5
LR = LogisticRegression(verbose=1, solver='liblinear',random_state=0, C=5, penalty='l2',max_iter=10)
predicted_targets_LR = np.array([])
actual_targets_LR = np.array([])
predicted_targets_SVM = np.array([])
actual_targets_SVM = np.array([])
accuracy_scores_LR = []
accuracy_scores_SVM = []
f1_scores_LR = []
f1_scores_SVM = []
for train_index, valid_index in skf.split(Q, z): 

    Q_train_fold, Q_valid_fold = Q[train_index], Q[valid_index] 
    z_train_fold, z_valid_fold = z[train_index], z[valid_index] 
    Q_train_fold = Q_train_fold.to_frame()
    Q_valid_fold = Q_valid_fold.to_frame()
    z_train_fold = z_train_fold.to_frame()
    z_valid_fold = z_valid_fold.to_frame()
    
    # Feature extraction
    X_train,X_test,feature_transformer=extract_features('txt_diet_period_livedIn_length_taxonomy_namedBy_species_link',Q_train_fold, Q_valid_fold,type="binary")
    Y_train = z_train_fold['type'].values
    Y_test = z_valid_fold['type'].values
    
    # Logistic Regression
    model = LR.fit(X_train,Y_train)
    accuracyLR = model.score(X_test, Y_test)
    accuracy_scores_LR.append(accuracyLR)
    Y_pred_LR = get_top_k_predictions(model,X_test,1)
    fold_f1_score_LR = f1_score(Y_test, Y_pred_LR, average='weighted')
    f1_scores_LR.append(fold_f1_score_LR)
    predicted_targets_LR = np.append(predicted_targets_LR, Y_pred_LR)
    actual_targets_LR = np.append(actual_targets_LR, Y_test)
    # SVM
    SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    modelSVM = SVM.fit(X_train,Y_train)
    accuracySVM = modelSVM.score(X_test, Y_test)
    accuracy_scores_SVM.append(accuracySVM)
    Y_pred_SVM = SVM.predict(X_test)
    fold_f1_score_SVM = f1_score(Y_test, Y_pred_SVM, average='weighted')
    f1_scores_SVM.append(fold_f1_score_SVM)
    predicted_targets_SVM = np.append(predicted_targets_SVM, Y_pred_SVM)
    actual_targets_SVM = np.append(actual_targets_SVM, Y_test)

print('\n')
print('\n')    
print('\n') 

mean_accuracy_LR = np.mean(accuracy_scores_LR)
print("LR Mean accuracy:", mean_accuracy_LR)    

mean_accuracy_SVM = np.mean(accuracy_scores_SVM)
print("SVM Mean accuracy:", mean_accuracy_SVM)   
print('\n')
mean_weighted_f1_score_LR = np.mean(f1_scores_LR)
print("Mean weighted-average F1-score across for LR", k, "folds:", mean_weighted_f1_score_LR)

mean_weighted_f1_score_SVM = np.mean(f1_scores_SVM)
print("Mean weighted-average F1-score across for SVM", k, "folds:", mean_weighted_f1_score_SVM)
print('\n')

t = np.shape(X_test)[1]
test_data = test_data.to_frame()
X_test_train, test ,feature_transformer=extract_features('txt_diet_period_livedIn_length_taxonomy_namedBy_species_link',training_data, test_data ,type="binary")
from scipy.sparse import csr_matrix
test_truncated = csr_matrix((test),shape=(np.shape(test)[0],t))
preds_LR = get_top_k_predictions(model,test_truncated,1)
preds_SVM = SVM.predict(test_truncated)

dfTest['predicted_type_LR'] = preds_LR
dfTest['predicted_type_SVM'] = preds_SVM
dfTest.to_csv('/Users/hbonen/Documents/Doktora/CMP712_Machine_Learning/hw2/predicted_types.csv')

preds_SVM_series = pd.Series(preds_SVM)
preds_SVM_df = preds_SVM_series.to_frame() 
predictions = preds_SVM_df
predictions["id"] = predictions.index + 1
predictions["type"] = preds_SVM

preds = pd.DataFrame()
preds["id"] = predictions["id"]
preds["type"] = predictions["type"]
preds = preds.set_index('id')
preds.to_csv('/Users/hbonen/Documents/Doktora/CMP712_Machine_Learning/hw2/predictions.csv')


conf_matrix_LR = confusion_matrix(actual_targets_LR, predicted_targets_LR)
fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(conf_matrix_LR, cmap=plt.cm.Oranges, alpha=0.3)
for i in range(conf_matrix_LR.shape[0]):
    for j in range(conf_matrix_LR.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix_LR[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix for LR', fontsize=18)
plt.show()

conf_matrix_SVM = confusion_matrix(actual_targets_SVM, predicted_targets_SVM)
fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(conf_matrix_SVM, cmap=plt.cm.Oranges, alpha=0.3)
for i in range(conf_matrix_SVM.shape[0]):
    for j in range(conf_matrix_SVM.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix_SVM[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix for SVM', fontsize=18)
plt.show()
