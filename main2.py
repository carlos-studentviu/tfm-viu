#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.linear_model import ElasticNet
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV, KFold, StratifiedGroupKFold
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import warnings

from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings('ignore')

def plot_roc_curve(true_y, y_prob, c):
    """
    plots the roc curve based of the probabilities
    """

    fpr, tpr, thresholds = roc_curve(true_y, y_prob)
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve '+str(c))
    plt.show()

# output.write("patient,smoothp,eyeaxis,delay,dimension,apen,sampen,fuzzyen,shen,reen,specen,fgpe,fgnpe,fgcpe,mpe,mnpe,mcpe,wpe,wnpe,wcpe,aape,aanpe,aacpe,epe,enpe,ecpe,uqpe,uqnpe,uqcpe,men,d2,hfd,kfd,pfd,lzc,lle,he,dfa,emcs,emcr,ehmmsn,ehmmrn,ehmmsu,ehmmru,emrds,emrdr,parkinson\n")
df = pd.read_csv("database.csv", sep=",")
#df = pd.read_csv("Database_complexity_Mehdi.csv", sep=",")
# replace 2 in parkinson column with 0
df.loc[df['parkinson'] == 2, 'parkinson'] = 0
# print(df['parkinson'].value_counts())
# normalize nonstring columns
'''
df_num = df.select_dtypes(include='number')
# drop the target column to avoid it being normalized
df_num = df_num.drop(columns=['parkinson'])
df_norm = (df_num - df_num.mean()) / df_num.std() #(df_num.max() - df_num.min())
df[df_norm.columns] = df_norm
'''
df_input = df.drop(columns=['delay', 'dimension', 'parkinson'])
df_y = df['parkinson']
patient_ids = df['patient'].unique()

# import groupkfold
from sklearn.model_selection import GroupKFold

list_params = []
list_removed = []
list_predictions = []
true_labels = []
tp = 0
tn = 0
fp = 0
fn = 0
correct = 0
positive = 0
negative = 0
c = 55
# Lets do a kfold to create a leaveone-out
kf = KFold(n_splits=100, shuffle=False)
# Loop over each patient for the outer CV
for train_index, test_index in kf.split(patient_ids):
    # Get the patient IDs for the training set
    train_patients = patient_ids[train_index]
    # Filter instances belonging to the training patients
    train_instances = df_input[df_input['patient'].isin(train_patients)]
    train_groups = train_instances['patient']
    x_train = train_instances.drop(columns=['patient', 'smoothp', 'eyeaxis'])
    y_train = df_y[df_y.index.isin(train_instances.index)]

    # Evaluate on the test patient
    test_instances = df_input[df_input['patient'].isin(patient_ids[test_index])]
    x_test = test_instances.drop(columns=['patient', 'smoothp', 'eyeaxis'])
    y_test = df_y[df_y.index.isin(test_instances.index)]
    true_labels.append(y_test.values[1])

    # normalize with stadnardscaler
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()

    # testing: use weird normalizer instead of standardscaler
    # x_train = scaler.fit_transform(x_train)
    # x_test = scaler.transform(x_test)
    mean = x_train.mean()
    max = x_train.max()
    min = x_train.min()
    df_norm = (x_train - x_train.mean()) / (x_train.max() - x_train.min())
    x_train[df_norm.columns] = df_norm
    df_norm = (x_test - mean) / (max - min)
    x_test[df_norm.columns] = df_norm


    # Crossvalidate the SVm and ElasticNet using gridsearchcv
    # ElasticNet
    # Params to crossvalidate: alpha, l1_ratio
    params = {"l1_ratio": np.arange(0.1, 0.9, 0.1),
              "penalty": ['l1', 'elasticnet']}
    # params = {"alpha": np.logspace(-3, 1, 4),
    #             "l1_ratio": np.arange(0.1, 1, 0.1)}
    elastic = LogisticRegression(solver="saga", penalty="elasticnet", max_iter=2000, random_state=42) #l1_ratio=0.8, max_iter=2000, random_state=42)
    #elastic = ElasticNet(alpha=0.0005, l1_ratio=0.9, random_state=42)
    #best_elastic.fit(x_train, y_train)
    group_k_fold = StratifiedGroupKFold(n_splits=5)
    grid_search = GridSearchCV(elastic, params, cv=group_k_fold, scoring='roc_auc', verbose=2, n_jobs=-1)
    # # Fit grid search
    grid_search.fit(x_train, y_train.round(), groups=train_groups)
    # # Print the best model
    print(grid_search.best_params_)
    list_params.append(grid_search.best_params_)
    


    best_elastic = grid_search.best_estimator_
    removed_feats = train_instances.drop(columns=['patient', 'smoothp', 'eyeaxis']).columns[(best_elastic.coef_ == 0).ravel().tolist()]
    print(removed_feats)
    list_removed.append(removed_feats)
    x_train = train_instances.drop(columns=['patient', 'smoothp', 'eyeaxis', *removed_feats])
    x_test = test_instances.drop(columns=['patient', 'smoothp', 'eyeaxis', *removed_feats])
    # x_train = scaler.fit_transform(x_train)
    # x_test = scaler.transform(x_test)
    mean = x_train.mean()
    max = x_train.max()
    min = x_train.min()
    df_norm = (x_train - mean) / (max - min)
    x_train[df_norm.columns] = df_norm
    df_norm = (x_test - mean) / (max - min)
    x_test[df_norm.columns] = df_norm

    # SVM
    # Params to crossvalidate: C, kernel rbf, kernel linear and gamma
    # params = {"C": np.logspace(-3, 1, 4),
    #           "kernel": ['rbf', 'linear'],
    #           "gamma": ['scale', 'auto', 0.1, 0.01, 0.001]}
    # params = {"C": [0.001, 0.01, 0.1, 1, 10, 55, 100, 1000],
    #             "kernel": ['rbf'],
    #             "gamma": ['scale', 'auto', 0.1, 0.01, 0.001]}
    # svms = svm.SVC(probability=True, random_state=42)
    # kf = KFold(n_splits=5, shuffle=False)
    #
    # # Setup the inner CV with grouped k-fold
    # group_k_fold = GroupKFold(n_splits=5)
    # grid_search = GridSearchCV(svms, params, cv=group_k_fold, scoring='accuracy', verbose=2,n_jobs=-1)
    # # Fit grid search
    # grid_search.fit(x_train, y_train, groups=train_groups)
    #
    # # Print the best model
    # print(grid_search.best_params_)
    # list_params.append(grid_search.best_params_)
    # print(grid_search.best_score_)
    # # Evaluate the best model on the test set
    # best_model = grid_search.best_estimator_
    #best_model = svm.SVC(C=0.46415888336127775, kernel='rbf', gamma=0.001, probability=True, random_state=42)

    best_model = svm.SVC(C=c, kernel='rbf', probability=True, random_state=42)
    best_model.fit(x_train, y_train)
    y_pred = best_model.predict_proba(x_test)
    predicted = np.average(y_pred[:, 1])
    list_predictions.append(predicted)
    print(accuracy_score(y_test, y_pred[:, 1] > 0.5))
    answer = y_test.values[1]

    if predicted >= 0.4375:  # 0.3125:
        predicted = 1
    else:
        predicted = 0

    if predicted == 1:
        if predicted == answer:
            tp+=1
            correct+=1
        else:
            fp+=1
    else:
        if predicted == answer:
            tn+=1
            correct+=1
        else:
            fn+=1
    if answer == 1:
        positive+=1
    else:
        negative+=1
    print('predicción: ' + str(predicted) + ' respuesta: ' + str(answer))

# plot_roc_curve(true_labels, predicted_probabilities, c)
# fpr, tpr, thresholds = roc_curve(true_labels, list_predictions,c)
# best_threshold = thresholds[np.argmax(tpr-fpr)]

# print('best threshold: ' + str(best_threshold))
print('accuracy: ' + str(correct / (positive + negative)))
print('tp: ' + str(tp/positive))
print('tn: ' + str(tn/negative))
print('fp: ' + str(fp/negative))
print('fn: ' + str(fn/positive))
print('sensitivity: ' + str(tp/(tp+fn)))
print('specificity: ' + str(tn/(tn+fp)))
print('precision: ' + str(tp/(tp+fp)))
print('f1: ' + str(2*tp/(2*tp+fp+fn)))
print('MCC: ' + str((tp*tn-fp*fn)/((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))**(1/2)))
print('AUC: ' + str(roc_auc_score(true_labels, list_predictions)))

# plot the roc curve
plot_roc_curve(true_labels, list_predictions, c)
# print the best parameters in a file
# with open('best_params2.txt', 'w') as f:
#     for i in range(0, len(list_params)):
#          f.write(str(list_params[i]) + '\n')
    # for j in range(0, len(list_removed)):
    #      f.write(str(list_removed[j]) + '\n')
    # f.close()




# exit
# correct=0
# tp = 0
# tn = 0
# fp = 0
# fn = 0
# positive = 0
# negative = 0
# predicted_probabilities = []
# true_labels = []
# c = 55
# # do a train test split of leave-one-out keeping the data from same patient in the same set
# patients = df['patient'].unique()
# for patient in patients:
#     train_df = df[df['patient'] != patient]
#     test_df = df[df['patient'] == patient]
#
#     # select the rows for each type of smooth pursuit
#     train_labels = train_df['parkinson']
#
#     sp1test = test_df.loc[test_df['smoothp']=="SmoothPur_1"]
#     sp1test_y = sp1test['parkinson']
#     sp2test = test_df.loc[test_df['smoothp']=="SmoothPur_2"]
#     sp2test_y = sp2test['parkinson']
#     sp3test = test_df.loc[test_df['smoothp']=="SmoothPur_3"]
#     sp3test_y = sp3test['parkinson']
#     sp4test = test_df.loc[test_df['smoothp']=="SmoothPur_4"]
#     sp4test_y = sp4test['parkinson']
#     sp5test = test_df.loc[test_df['smoothp']=="SmoothPur_5"]
#     sp5test_y = sp5test['parkinson']
#     sp6test = test_df.loc[test_df['smoothp']=="SmoothPur_6"]
#     sp6test_y = sp6test['parkinson']
#     sp7test = test_df.loc[test_df['smoothp']=="SmoothPur_7"]
#     sp7test_y = sp7test['parkinson']
#     sp8test = test_df.loc[test_df['smoothp']=="SmoothPur_8"]
#     sp8test_y = sp8test['parkinson']
#     sp9test = test_df.loc[test_df['smoothp']=="SmoothPur_9"]
#     sp10test = sp9test.loc[sp9test['eyeaxis']=="y"]
#     sp9test = sp9test.loc[sp9test['eyeaxis']=="x"]
#     sp9test_y = sp9test['parkinson']
#     sp10test_y = sp10test['parkinson']
#     sp11test = test_df.loc[test_df['smoothp']=="SmoothPur_10"]
#     sp12test = sp11test.loc[sp11test['eyeaxis']=="y"]
#     sp11test = sp11test.loc[sp11test['eyeaxis']=="x"]
#     sp11test_y = sp11test['parkinson']
#     sp12test_y = sp12test['parkinson']
#     sp13test = test_df.loc[test_df['smoothp']=="SmoothPur_11"]
#     sp14test = sp13test.loc[sp13test['eyeaxis']=="y"]
#     sp13test = sp13test.loc[sp13test['eyeaxis']=="x"]
#     sp13test_y = sp13test['parkinson']
#     sp14test_y = sp14test['parkinson']
#     sp15test = test_df.loc[test_df['smoothp']=="SmoothPur_12"]
#     sp16test = sp15test.loc[sp15test['eyeaxis']=="y"]
#     sp15test = sp15test.loc[sp15test['eyeaxis']=="x"]
#     sp15test_y = sp15test['parkinson']
#     sp16test_y = sp16test['parkinson']
#
#     # we drop the columns that are not needed and the target column
#     train_df = train_df.drop(columns=['patient', 'smoothp', 'eyeaxis', 'parkinson'])
#     sp1test = sp1test.drop(columns=['patient', 'smoothp', 'eyeaxis', 'parkinson'])
#     sp2test = sp2test.drop(columns=['patient', 'smoothp', 'eyeaxis', 'parkinson'])
#     sp3test = sp3test.drop(columns=['patient', 'smoothp', 'eyeaxis', 'parkinson'])
#     sp4test = sp4test.drop(columns=['patient', 'smoothp', 'eyeaxis', 'parkinson'])
#     sp5test = sp5test.drop(columns=['patient', 'smoothp', 'eyeaxis', 'parkinson'])
#     sp6test = sp6test.drop(columns=['patient', 'smoothp', 'eyeaxis', 'parkinson'])
#     sp7test = sp7test.drop(columns=['patient', 'smoothp', 'eyeaxis', 'parkinson'])
#     sp8test = sp8test.drop(columns=['patient', 'smoothp', 'eyeaxis', 'parkinson'])
#     sp9test = sp9test.drop(columns=['patient', 'smoothp', 'eyeaxis', 'parkinson'])
#     sp10test = sp10test.drop(columns=['patient', 'smoothp', 'eyeaxis', 'parkinson'])
#     sp11test = sp11test.drop(columns=['patient', 'smoothp', 'eyeaxis', 'parkinson'])
#     sp12test = sp12test.drop(columns=['patient', 'smoothp', 'eyeaxis', 'parkinson'])
#     sp13test = sp13test.drop(columns=['patient', 'smoothp', 'eyeaxis', 'parkinson'])
#     sp14test = sp14test.drop(columns=['patient', 'smoothp', 'eyeaxis', 'parkinson'])
#     sp15test = sp15test.drop(columns=['patient', 'smoothp', 'eyeaxis', 'parkinson'])
#     sp16test = sp16test.drop(columns=['patient', 'smoothp', 'eyeaxis', 'parkinson'])
#
#     # parameters to be tested on GridSearchCV
#     #params = {"alpha":np.arange(0.00001, 10, 500)}
#     # Number of Folds and adding the random state for replication
#     #kf=KFold(n_splits=5,shuffle=True, random_state=42)
#     # Initializing the Model
#     #lasso = Lasso()
#     # GridSearchCV with model, params and folds.
#     #lasso_cv=GridSearchCV(lasso, param_grid=params, cv=kf)
#     #lasso_cv.fit(train_df, train_labels)
#     #print("Best Params {}".format(lasso_cv.best_params_))
#
#     #names=train_df.columns
#     #print("Column Names: {}".format(names.values))
#
#     # calling the model with the best parameter
#     #lasso1 = Lasso(alpha=0.00001)
#     #lasso1.fit(train_df, train_labels)
#     # Using np.abs() to make coefficients positive.
#     #lasso1_coef = np.abs(lasso1.coef_)
#
#     # Subsetting the features which has more than 0.001 importance.
#     #feature_subset=np.array(names)[lasso1_coef>0.001]
#     #print("Selected Feature Columns: {}".format(feature_subset))
#
#     #train_df = train_df.drop(columns=['reen', 'wpe', 'ehmmrn'])
#     #sp1test = sp1test.drop(columns=['reen', 'wpe', 'ehmmrn'])
#     #sp2test = sp2test.drop(columns=['reen', 'wpe', 'ehmmrn'])
#     #sp3test = sp3test.drop(columns=['reen', 'wpe', 'ehmmrn'])
#     #sp4test = sp4test.drop(columns=['reen', 'wpe', 'ehmmrn'])
#     #sp5test = sp5test.drop(columns=['reen', 'wpe', 'ehmmrn'])
#     #sp6test = sp6test.drop(columns=['reen', 'wpe', 'ehmmrn'])
#     #sp7test = sp7test.drop(columns=['reen', 'wpe', 'ehmmrn'])
#     #sp8test = sp8test.drop(columns=['reen', 'wpe', 'ehmmrn'])
#     #sp9test = sp9test.drop(columns=['reen', 'wpe', 'ehmmrn'])
#     #sp10test = sp10test.drop(columns=['reen', 'wpe', 'ehmmrn'])
#     #sp11test = sp11test.drop(columns=['reen', 'wpe', 'ehmmrn'])
#     #sp12test = sp12test.drop(columns=['reen', 'wpe', 'ehmmrn'])
#     #sp13test = sp13test.drop(columns=['reen', 'wpe', 'ehmmrn'])
#     #sp14test = sp14test.drop(columns=['reen', 'wpe', 'ehmmrn'])
#     #sp15test = sp15test.drop(columns=['reen', 'wpe', 'ehmmrn'])
#     #sp16test = sp16test.drop(columns=['reen', 'wpe', 'ehmmrn'])
#
#     """
#     sel_ = SelectFromModel(
#         LogisticRegression(C=0.5, penalty='l1', solver='liblinear', random_state=10))
#     sel_.fit(train_df, train_labels)
#     removed_feats = train_df.columns[(sel_.estimator_.coef_ == 0).ravel().tolist()]
#     print(removed_feats)
#     train_df = train_df.drop(columns=removed_feats)
#     sp1test = sp1test.drop(columns=removed_feats)
#     sp2test = sp2test.drop(columns=removed_feats)
#     sp3test = sp3test.drop(columns=removed_feats)
#     sp4test = sp4test.drop(columns=removed_feats)
#     sp5test = sp5test.drop(columns=removed_feats)
#     sp6test = sp6test.drop(columns=removed_feats)
#     sp7test = sp7test.drop(columns=removed_feats)
#     sp8test = sp8test.drop(columns=removed_feats)
#     sp9test = sp9test.drop(columns=removed_feats)
#     sp10test = sp10test.drop(columns=removed_feats)
#     sp11test = sp11test.drop(columns=removed_feats)
#     sp12test = sp12test.drop(columns=removed_feats)
#     sp13test = sp13test.drop(columns=removed_feats)
#     sp14test = sp14test.drop(columns=removed_feats)
#     sp15test = sp15test.drop(columns=removed_feats)
#     sp16test = sp16test.drop(columns=removed_feats)
#     """
#
#     elastic = ElasticNet(alpha=0.0005, l1_ratio=0.9)
#     elastic.fit(train_df, train_labels)
#     removed_feats = train_df.columns[(elastic.coef_ == 0).ravel().tolist()]
#     #print(removed_feats)
#     train_df = train_df.drop(columns=removed_feats)
#     sp1test = sp1test.drop(columns=removed_feats)
#     ... #repeat for all smooth pursuits
#     sp2test = sp2test.drop(columns=removed_feats)
#     sp3test = sp3test.drop(columns=removed_feats)
#     sp4test = sp4test.drop(columns=removed_feats)
#     sp5test = sp5test.drop(columns=removed_feats)
#     sp6test = sp6test.drop(columns=removed_feats)
#     sp7test = sp7test.drop(columns=removed_feats)
#     sp8test = sp8test.drop(columns=removed_feats)
#     sp9test = sp9test.drop(columns=removed_feats)
#     sp10test = sp10test.drop(columns=removed_feats)
#     sp11test = sp11test.drop(columns=removed_feats)
#     sp12test = sp12test.drop(columns=removed_feats)
#     sp13test = sp13test.drop(columns=removed_feats)
#     sp14test = sp14test.drop(columns=removed_feats)
#     sp15test = sp15test.drop(columns=removed_feats)
#     sp16test = sp16test.drop(columns=removed_feats)
#
#
#     # Create SVM object
#     svms = svm.SVC(kernel='rbf', C=c, random_state=42) # svm.LinearSVC(C=75, random_state=20) # svm.SVC(kernel='rbf', C=1,gamma=0)
#     # svms = KNeighborsClassifier(n_neighbors=13)
#     # svms = RandomForestClassifier(random_state=42)
#     # Train the model using the training sets and check score on test dataset
#     svms.fit(train_df, train_labels)
#
#     for i in range(0,len(sp1test)):
#         #double brackets to get a dataframe instead of series
#         test1 = sp1test.iloc[[i]]
#         test2 = sp2test.iloc[[i]]
#         test3 = sp3test.iloc[[i]]
#         test4 = sp4test.iloc[[i]]
#         test5 = sp5test.iloc[[i]]
#         test6 = sp6test.iloc[[i]]
#         test7 = sp7test.iloc[[i]]
#         test8 = sp8test.iloc[[i]]
#         test9 = sp9test.iloc[[i]]
#         test10 = sp10test.iloc[[i]]
#         test11 = sp11test.iloc[[i]]
#         test12 = sp12test.iloc[[i]]
#         test13 = sp13test.iloc[[i]]
#         test14 = sp14test.iloc[[i]]
#         test15 = sp15test.iloc[[i]]
#         test16 = sp16test.iloc[[i]]
#         #answer is the same for all the models
#         answer = sp1test_y.iloc[i]
#         true_labels.append(answer)
#
#         predicted1= svms.predict(test1)
#         predicted2= svms.predict(test2)
#         predicted3= svms.predict(test3)
#         predicted4= svms.predict(test4)
#         predicted5= svms.predict(test5)
#         predicted6= svms.predict(test6)
#         predicted7= svms.predict(test7)
#         predicted8= svms.predict(test8)
#         predicted9= svms.predict(test9)
#         predicted10= svms.predict(test10)
#         predicted11= svms.predict(test11)
#         predicted12= svms.predict(test12)
#         predicted13= svms.predict(test13)
#         predicted14= svms.predict(test14)
#         predicted15= svms.predict(test15)
#         predicted16= svms.predict(test16)
#
#         predictedParkinson = (predicted1 + predicted2 + predicted3 + predicted4 + predicted5 + predicted6 + predicted7 + predicted8
#                  + predicted9 + predicted10 + predicted11 + predicted12 + predicted13 + predicted14 + predicted15 + predicted16)
#
#         predicted = predictedParkinson/16
#         predicted_probabilities.append(predicted)
#         print('predicted: ' + str(predicted) + ' answer: ' + str(answer))
#         #use consensus of all models to predict if patient has parkinson
#         if predicted >= 0.4375: #0.3125:
#             predicted = 1
#         else:
#             predicted = 0
#
#         if predicted == 1:
#             if predicted == answer:
#                 tp+=1
#                 correct+=1
#             else:
#                 fp+=1
#         else:
#             if predicted == answer:
#                 tn+=1
#                 correct+=1
#             else:
#                 fn+=1
#         if answer == 1:
#             positive+=1
#         else:
#             negative+=1
#         print('predicción: ' + str(predicted) + ' respuesta: ' + str(answer))
#
# plot_roc_curve(true_labels, predicted_probabilities, c)
# fpr, tpr, thresholds = roc_curve(true_labels, predicted_probabilities)
# best_threshold = thresholds[np.argmax(tpr-fpr)]
#
# print('best threshold: ' + str(best_threshold))
# print('accuracy: ' + str(correct / (positive + negative)))
# print('tp: ' + str(tp/positive))
# print('tn: ' + str(tn/negative))
# print('fp: ' + str(fp/negative))
# print('fn: ' + str(fn/positive))
# print('sensitivity: ' + str(tp/(tp+fn)))
# print('specificity: ' + str(tn/(tn+fp)))
# print('precision: ' + str(tp/(tp+fp)))
# print('f1: ' + str(2*tp/(2*tp+fp+fn)))
# print('MCC: ' + str((tp*tn-fp*fn)/((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))**(1/2)))
# print('AUC: ' + str(roc_auc_score(true_labels, predicted_probabilities)))
