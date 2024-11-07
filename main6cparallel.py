#!/usr/bin/env python
# -*- coding: utf-8 -*-
# este main utiliza dos leave-one-out, uno para elegir la mejor selección de características y otro para evaluar el modelo
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.linear_model import ElasticNet
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import GridSearchCV, KFold, StratifiedGroupKFold
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.metrics import PrecisionRecallDisplay, precision_recall_curve
from sklearn.metrics import det_curve, DetCurveDisplay
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import warnings
import rocch as rocch
import concurrent.futures

from sklearn.neighbors import KNeighborsClassifier
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

database = "database6c.csv" #"database6c.csv"

def plot_roc_curve(true_y, y_prob, c, filename="roc_curve.png"):
    """
    plots the roc curve based of the probabilities
    """

    fpr, tpr, thresholds = roc_curve(true_y, y_prob)
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], "r--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve " + str(c) +" features")

    # Save the figure to a file
    plt.savefig(filename)

    plt.show()


# Define preprocessing steps
def build_preprocessor(x_train, normalizer="standard"):
    if normalizer == "standard":
        scaler = StandardScaler()
    elif normalizer == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError("Unsupported normalizer. Choose 'standard' or 'minmax'.")

    return (
        ColumnTransformer(
            [
                (
                    "numeric",
                    scaler,
                    slice(0, len(x_train.columns)),
                )  # Apply scaler to numeric columns
            ]
        ),
        scaler,
    )

def build_final_pipeline(X, normalizer="standard", model="random_forest"):
    preprocessor, scaler = build_preprocessor(X,normalizer)
    model_obj, param_grid = models[model]

    return (
        Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    GridSearchCV(
                        model_obj,
                        param_grid,
                        cv=StratifiedGroupKFold(n_splits=5),
                        scoring="roc_auc",
                        verbose=0,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        scaler,
    )

def run_feature_selection(num_features):

    list_params = []
    list_removed = []
    list_predictions = []
    list_predict_binary = []
    list_pos_class_index = []
    list_neg_class_index = []
    list_best_thresholds = []
    list_features = []
    true_labels = []
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    correct = 0
    positive = 0
    negative = 0

    c = num_features #c used to be a hyper-parameter of the svm classifier, now it's used to name the roc output files by the number of features
    # Lets do a kfold to create a leaveone-out for the feature selection
    kf = KFold(n_splits=100, shuffle=False)
    # Loop over each patient for the outer CV
    for train_index, test_index in kf.split(patient_ids): #tqdm(kf.split(patient_ids)):

        # Get the patient IDs for the training set
        train_patients = patient_ids[train_index]
        # Filter instances belonging to the training patients
        train_instances = df_input[df_input["patient"].isin(train_patients)]
        train_groups = train_instances["patient"]
        x_train = train_instances.drop(columns=["patient", "smoothp", "eyeaxis"])
        y_train = df_y[df_y.index.isin(train_instances.index)]

        # Evaluate on the test patient
        #test_instances = df_input[df_input["patient"].isin(patient_ids[test_index])]
        #x_test = test_instances.drop(columns=["patient", "smoothp", "eyeaxis"])
        #y_test = df_y[df_y.index.isin(test_instances.index)]
        #true_labels.append(y_test.values[1])

        # Elasticnet Define and fit the pipeline
        # pipeline, scaler = build_pipeline(normalizer='minmax', model='logistic_regression')
        # pipeline.fit(x_train, y_train, classifier__groups=train_groups)
        #
        # # Access best parameters and estimator from the pipeline
        # print("Best score:", pipeline.named_steps['classifier'].best_score_)
        # print("Best Parameters:", pipeline.named_steps['classifier'].best_params_)
        # best_elastic = pipeline.named_steps['classifier'].best_estimator_
        #
        # # # Print the best elasticnet model
        # list_params.append(pipeline.named_steps['classifier'].best_params_)
        #
        # # if the classifier is logistic regression, we can remove the features with 0 coefficients
        # if best_elastic.__class__.__name__ == 'LogisticRegression':
        #     removed_feats = train_instances.drop(columns=['patient', 'smoothp', 'eyeaxis']).columns[
        #         (best_elastic.coef_ == 0).ravel().tolist()]
        #     print(removed_feats)
        #     list_removed.append(removed_feats)
        #     x_train = train_instances.drop(columns=['patient', 'smoothp', 'eyeaxis', *removed_feats])
        #     x_test = test_instances.drop(columns=['patient', 'smoothp', 'eyeaxis', *removed_feats])

        # make stratifiedgroupkfold for mutual_info_classif
        scores_mutual_info = np.zeros(x_train.shape[1])
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=False)

        # we do cross-validation of 5 splits to get the mutual information scores for feature selection
        for (train_index, test_index) in sgkf.split(x_train, y_train, train_groups):
            fold_x_train = x_train.iloc[train_index]
            fold_y_train = y_train[y_train.index.isin(fold_x_train.index)]
            # the test is not used because we are not validating the model, just getting the mutual information scores
            fold_x_test = x_train.iloc[test_index]
            fold_y_test = y_train[y_train.index.isin(fold_x_test.index)]

            # apply minmaxscaler to numeric columns of x_train_cv and x_test_cv
            myscaler = MinMaxScaler()
            # Select numeric columns of x_train and x_test
            numeric_columns = fold_x_train.select_dtypes(include=['float64', 'int64']).columns
            norm_fold_x_train = fold_x_train.copy()
            norm_fold_x_test = fold_x_test.copy()
            # fit and transform the scaler with training data
            norm_fold_x_train[numeric_columns] = myscaler.fit_transform(fold_x_train[numeric_columns])
            # transform test data with the same scaler (without fitting again)
            norm_fold_x_test[numeric_columns] = myscaler.transform(fold_x_test[numeric_columns])

            # Use mutual_info_classif to get the mutual information scores
            # Compute mutual information for the current fold
            mi = mutual_info_classif(norm_fold_x_train, fold_y_train, n_neighbors=5, random_state=42, n_jobs=-1)

            # Aggregate the results
            scores_mutual_info += mi

        # Average the mutual information scores over all folds
        scores_mutual_info /= 5

        # Initialize a binary vector to store the selected features as zeros or ones
        feature_map = np.zeros_like(scores_mutual_info)

        # Select the top num_features features
        top_indices = np.argsort(scores_mutual_info)[-num_features:]

        # mark the selected features with 1
        feature_map[top_indices] = 1

        list_features.append(feature_map)


    indexes_filename = 'mi_best_indexes'+database.split('.')[0]+'_'+str(num_features)+'features.txt'
    with open(indexes_filename, 'w') as f:
        for i in range(0, len(list_features)):
            f.write(str(list_features[i]) + '\n')
    f.close()


    # Convertir list_features a un array de numpy para facilitar el manejo
    features_array = np.array(list_features)
    # Contar cuántas veces cada índice tiene un valor de 1
    feature_counts = np.sum(features_array, axis=0)

    # ordenar los indices de mayor número de unos a menor
    ordered_feature_indices = np.argsort(feature_counts)[::-1]
    with open(indexes_filename, 'a') as f:
        #f.write("Feature counts from higher to lower: " + '\n')
        #change the vectors to one-line strings
        feature_counts_str = " ".join(map(str, np.sort(feature_counts)[::-1]))
        ordered_feature_indices_str = " ".join(map(str, ordered_feature_indices))
        f.write(feature_counts_str + '\n')
        #f.write("Ordered feature indices from higher to lower: " + '\n')
        f.write(ordered_feature_indices_str + '\n')
    f.close()

    # Seleccionar num_features índices con mayor número de unos
    consensus_features = np.argsort(feature_counts)[::-1][:num_features]
    # Seleccionar los índices que tienen al menos 30 unos
    #consensus_features = np.where(feature_counts >= 30)[0]
    with open(indexes_filename, 'a') as f:
        consensus_features_str = " ".join(map(str, np.sort(consensus_features)))
        f.write(consensus_features_str + '\n')
    f.close()

    # Lets do a kfold to create a leaveone-out for the evaluation of the classifier
    kf = KFold(n_splits=100, shuffle=False)
    # Loop over each patient for the outer CV
    for train_index, test_index in kf.split(patient_ids): #tqdm(kf.split(patient_ids)):

        # Get the patient IDs for the training set
        train_patients = patient_ids[train_index]
        # Filter instances belonging to the training patients
        train_instances = df_input[df_input["patient"].isin(train_patients)]
        train_groups = train_instances["patient"]
        x_train = train_instances.drop(columns=["patient", "smoothp", "eyeaxis"])
        y_train = df_y[df_y.index.isin(train_instances.index)]

        # Evaluate on the test patient
        test_instances = df_input[df_input["patient"].isin(patient_ids[test_index])]
        x_test = test_instances.drop(columns=["patient", "smoothp", "eyeaxis"])
        y_test = df_y[df_y.index.isin(test_instances.index)]
        true_labels.append(y_test.values[1])
        list_train_predictions = []
        true_labels_patient = []

        # Filter x_train and x_test so they only contain the selected features
        x_train = x_train.iloc[:, consensus_features]
        x_test = x_test.iloc[:, consensus_features]


        # Elasticnet Define and fit the pipeline
        # pipeline, scaler = build_pipeline(normalizer='minmax', model='logistic_regression')
        # pipeline.fit(x_train, y_train, classifier__groups=train_groups)
        #
        # # Access best parameters and estimator from the pipeline
        # print("Best score:", pipeline.named_steps['classifier'].best_score_)
        # print("Best Parameters:", pipeline.named_steps['classifier'].best_params_)
        # best_elastic = pipeline.named_steps['classifier'].best_estimator_
        #
        # # # Print the best elasticnet model
        # list_params.append(pipeline.named_steps['classifier'].best_params_)
        #
        # # if the classifier is logistic regression, we can remove the features with 0 coefficients
        # if best_elastic.__class__.__name__ == 'LogisticRegression':
        #     removed_feats = train_instances.drop(columns=['patient', 'smoothp', 'eyeaxis']).columns[
        #         (best_elastic.coef_ == 0).ravel().tolist()]
        #     print(removed_feats)
        #     list_removed.append(removed_feats)
        #     x_train = train_instances.drop(columns=['patient', 'smoothp', 'eyeaxis', *removed_feats])
        #     x_test = test_instances.drop(columns=['patient', 'smoothp', 'eyeaxis', *removed_feats])

        # # Use now a SVM with the best parameters
        fpipeline, scaler = build_final_pipeline(x_train, normalizer="minmax", model="svm")  # model='svm')
        fpipeline.fit(x_train, y_train, classifier__groups=train_groups)

        # best_model = pipeline.named_steps['classifier'].best_estimator_

        y_train_pred = fpipeline.predict_log_proba(x_train)

        for patient in train_patients:
            patient_idx = train_instances["patient"].isin([patient])
            y_train_pred_idx = y_train_pred[patient_idx]
            y_train_idx = y_train[patient_idx]
            avg_1 = np.average(y_train_pred_idx[:, 1])
            avg_0 = np.average(y_train_pred_idx[:, 0])

            true_labels_patient.append(y_train_idx.array[0])
            result = avg_1 - avg_0
            if np.isposinf(result):
                result = 100
            if np.isneginf(result):
                result = -100
            list_train_predictions.append(result)  # predicted)

        fpr, tpr, thresholds = roc_curve(true_labels_patient, list_train_predictions)
        best_threshold = thresholds[np.argmax(tpr - fpr)]
        list_best_thresholds.append(best_threshold)

        y_pred = fpipeline.predict_log_proba(x_test)
        avg_1 = np.average(y_pred[:, 1])
        avg_0 = np.average(y_pred[:, 0])
        list_predictions.append(avg_1 - avg_0)
        if avg_1 - avg_0 > best_threshold:  # avg_0:
            predicted = 1
            list_predict_binary.append(1)
        else:
            predicted = 0
            list_predict_binary.append(0)

        answer = y_test.values[1]
        if answer == 1:
            list_pos_class_index.append(avg_1 - avg_0)
        else:
            list_neg_class_index.append(avg_1 - avg_0)

        # threshold = 0.4375
        if predicted == 1:
            if predicted == answer:
                tp += 1
            else:
                fp += 1
        else:
            if predicted == answer:
                tn += 1
            else:
                fn += 1

        print("predicción: " + str(predicted) + " respuesta: " + str(answer))

    # save the scores to a file
    npz_filename = 'mi_classificationscores' + database.split('.')[0] + 'svmminmax' + str(num_features) + 'features.npz'
    np.savez(npz_filename, list_predictions=list_predictions, list_best_thresholds=list_best_thresholds,
                 list_pos_class_index=list_pos_class_index, list_neg_class_index=list_neg_class_index,
                 true_labels=true_labels, list_predict_binary=list_predict_binary)

    # plot_roc_curve(true_labels, predicted_probabilities, c)
    # fpr, tpr, thresholds = roc_curve(true_labels, list_predictions,c)
    # best_threshold = thresholds[np.argmax(tpr-fpr)]
    correct = tp + tn
    positive = tp + fn
    negative = tn + fp

    print("Consensus features:", np.sort(consensus_features))
    print("best threshold: " + str(best_threshold))
    print("accuracy: " + str(correct / (positive + negative)))
    print("tp: " + str(tp / positive))
    print("tn: " + str(tn / negative))
    print("fp: " + str(fp / negative))
    print("fn: " + str(fn / positive))
    print("sensitivity: " + str(0) if (tp + fn) == 0 else "sensitivity: " + str(tp / (tp + fn)))
    print("specificity: " + str(0) if (tn + fp) == 0 else "specificity: " + str(tn / (tn + fp)))
    print("precision: " + str(0) if (tp + fp) == 0 else "precision: " + str(tp / (tp + fp)))
    print("f1: " + str(0) if (tp + fp) == 0 else "f1: " + str(2 * tp / (2 * tp + fp + fn)))
    print(
        "MCC: " + str(0)
        if (tp + fp) == 0 or (tp + fn) == 0 or (tn + fp) == 0 or (tn + fn) == 0
        else "MCC: " + str(
            (tp * tn - fp * fn) / ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** (1 / 2)
        )
    )
    print("AUC: " + str(roc_auc_score(true_labels, list_predictions)))
    results_filename = 'mi_results'+database.split('.')[0]+'svmminmax'+str(num_features)+'features.txt'
    with open(results_filename, "a") as f:
        f.write("=====================================================================\n")
        f.write("Database: " + database + "+svm minmax\n")
        f.write("Num features: " + str(num_features) + " backwards\n")
        f.write("Consensus features: " + str(np.sort(consensus_features)) + "\n")
        f.write("best threshold: " + str(best_threshold) + "\n")
        f.write("accuracy: " + str(correct / (positive + negative)) + "\n")
        f.write("tp: " + str(tp / positive) + "\n")
        f.write("tn: " + str(tn / negative) + "\n")
        f.write("fp: " + str(fp / negative) + "\n")
        f.write("fn: " + str(fn / positive) + "\n")
        f.write("sensitivity: " + (str(0) if (tp + fn) == 0 else str(tp / (tp + fn))) + "\n")
        f.write("specificity: " + (str(0) if (tn + fp) == 0 else str(tn / (tn + fp))) + "\n")
        f.write("precision: " + (str(0) if (tp + fp) == 0 else str(tp / (tp + fp))) + "\n")
        f.write("f1: " + (str(0) if (tp + fp) == 0 else str(2 * tp / (2 * tp + fp + fn))) + "\n")
        f.write(
            "MCC: " + (
                str(0) if (tp + fp) == 0 or (tp + fn) == 0 or (tn + fp) == 0 or (tn + fn) == 0
                else str((tp * tn - fp * fn) / ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** (1 / 2))
            ) + "\n"
        )
        f.write("AUC: " + str(roc_auc_score(true_labels, list_predictions)) + "\n")



    roc_filename = 'mi_roc'+database.split('.')[0]+'svmminmax'+str(num_features)+'features.png'
    rocch_filename = 'mi_rocch'+database.split('.')[0]+'svmminmax'+str(num_features)+'features.png'
    pr_filename = 'mi_pr'+database.split('.')[0]+'svmminmax'+str(num_features)+'features.png'
    det_filename = 'mi_det'+database.split('.')[0]+'svmminmax'+str(num_features)+'features.png'
    detch_filename = 'mi_detch'+database.split('.')[0]+'svmminmax'+str(num_features)+'features.png'
    score_filename = 'mi_score'+database.split('.')[0]+'svmminmax'+str(num_features)+'features.png'
    # plot the roc curve
    plot_roc_curve(true_labels, list_predictions, c, roc_filename)

    # plot the ROCCH curve
    fpr, tpr, _ = roc_curve(true_labels, list_predictions)
    F, T, auc = rocch.rocch(fpr, tpr)
    print("AUC ROCCH: " + str(auc))

    plt.plot(F, T)
    plt.plot([0, 1], [0, 1], "r--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROCCH Curve "+str(c)+" features")
    plt.savefig(rocch_filename)
    plt.show()

    # plot the score chart
    _ = plt.hist(list_pos_class_index, bins=30, rwidth=0.9, alpha=0.5, label="P_scores")
    _ = plt.hist(list_neg_class_index, bins=30, rwidth=0.9, alpha=0.5, label="N_scores")
    plt.legend()
    plt.title("Score "+str(c)+" features")
    plt.savefig(score_filename)
    plt.show()

    # plot the precision-recall curve
    precision, recall, thresholds = precision_recall_curve(true_labels, list_predictions)
    pr_display = PrecisionRecallDisplay(precision=precision, recall=recall).plot()
    plt.title("PR Curve "+str(c)+" features")
    plt.savefig(pr_filename)
    plt.show()

    # plot the DET curve
    fpr, fnr, thresholds = det_curve(true_labels, list_predictions)
    fig,ax = plt.subplots()
    plt.plot(fpr, fnr)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=2)
    #plt.yscale('log')
    #plt.xscale('log')
    ticks_to_use = [0.00390625,0.0078125,0.015625,0.03125,0.0625,0.125,0.25,0.5,1,2,4,8,16] #[0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10,20,50]
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.get_yaxis().set_major_formatter(ScalarFormatter())
    ax.set_xticks(ticks_to_use)
    ax.set_yticks(ticks_to_use)
    plt.axis([0.00390625,16,0.00390625,16]) #[0.001,50,0.001,50])
    # display = DetCurveDisplay(fpr=fpr, fnr=fnr, estimator_name="SVM")
    # display.plot()
    plt.xlabel("False Positive Rate")
    plt.ylabel("False Negative Rate")
    plt.title("Detection Error Tradeoff (DET) Curve "+str(c)+" features")
    plt.savefig(det_filename)
    plt.show()

    # plot the DETCH curve
    fpr, fnr, thresholds = det_curve(true_labels, list_predictions)
    #fpr = 1-fpr
    fnr = 1-fnr
    F, T, auc = rocch.rocch(fpr, fnr)
    Fcomplement = []
    for i in F:
        Fcomplement.append(i)
    Tcomplement = []
    for i in T:
        Tcomplement.append(1-i)

    fig,ax = plt.subplots()
    plt.plot(Fcomplement, Tcomplement)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=2)
    #plt.yscale('log')
    #plt.xscale('log')
    ticks_to_use = [0.00390625,0.0078125,0.015625,0.03125,0.0625,0.125,0.25,0.5,1,2,4,8,16] #[0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10,20,50]
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.get_yaxis().set_major_formatter(ScalarFormatter())
    ax.set_xticks(ticks_to_use)
    ax.set_yticks(ticks_to_use)
    plt.axis([0.00390625,16,0.00390625,16]) #[0.001,50,0.001,50])
    # display = DetCurveDisplay(fpr=Fcomplement, fnr=Tcomplement, estimator_name="SVM")
    # display.plot()
    plt.xlabel("False Positive Rate")
    plt.ylabel("False Negative Rate")
    plt.title("Detection Error Tradeoff Convex Hull (DETCH) Curve "+str(c)+" features")
    plt.savefig(detch_filename)
    plt.show()
    print("AUC DETCH: " + str(auc))
    # print the best parameters in a file
    # with open('best_params2.txt', 'w') as f:
    #     for i in range(0, len(list_params)):
    #          f.write(str(list_params[i]) + '\n')
    # for j in range(0, len(list_removed)):
    #      f.write(str(list_removed[j]) + '\n')
    # f.close()
    return

# output.write("patient,smoothp,eyeaxis,delay,dimension,apen,sampen,fuzzyen,shen,reen,specen,fgpe,fgnpe,fgcpe,mpe,mnpe,mcpe,wpe,wnpe,wcpe,aape,aanpe,aacpe,epe,enpe,ecpe,uqpe,uqnpe,uqcpe,men,d2,hfd,kfd,pfd,lzc,lle,he,dfa,emcs,emcr,ehmmsn,ehmmrn,ehmmsu,ehmmru,emrds,emrdr,parkinson\n")
df = pd.read_csv(
    database,# "/Users/julian/Documents/UPM/Oculography/DataBases/Database_complexity_Carlos.csv",
    sep=",",
)
# df = pd.read_csv("Database_complexity_Mehdi.csv", sep=",")
# replace 2 in young control parkinson column with 0, making it equivalent to control subjects
df.loc[df["parkinson"] == 2, "parkinson"] = 0
# print(df['parkinson'].value_counts())
# normalize nonstring columns
"""
df_num = df.select_dtypes(include='number')
# drop the target column to avoid it being normalized
df_num = df_num.drop(columns=['parkinson'])
df_norm = (df_num - df_num.mean()) / df_num.std() #(df_num.max() - df_num.min())
df[df_norm.columns] = df_norm
"""
'''
# remove rows with axis without signal
xonly = ["SmoothPur_1", "SmoothPur_2", "SmoothPur_3", "SmoothPur_4"]
yonly = ["SmoothPur_5", "SmoothPur_6", "SmoothPur_7", "SmoothPur_8"]
# Remove rows where 'eyeaxis' is 'x' and 'smoothp' is in the list task
df = df[~((df['eyeaxis'] == 'y') & (df['smoothp'].isin(xonly)))]
df = df[~((df['eyeaxis'] == 'x') & (df['smoothp'].isin(yonly)))]
'''

#
# database2.csv
#columns_to_maybe_drop = ['mpe','mnpe', 'aape', 'aanpe', 'wcpe', 'fgcpe'] #70% 70% 70% 71% 71% 71%
#df = df.drop(columns=columns_to_maybe_drop)
# columns_to_keep = ['wpe', 'mcpe', 'epe', 'enpe', 'uqpe', 'ecpe', 'wnpe', 'd2', 'shen', 'specen', 'uqpe', 'uqnpe', 'uqcpe', 'men', 'fgpe', 'fuzzyen', 'fgnpe', 'dfa', 'apen', 'sampen', 'reen', 'hfd', 'kfd', 'pfd', 'lle', 'he'] #68% 68% 70% 70% 70% 67% 68% 69% 70% 65% 70% 68% 70% 70% 69% 68% 70% 64% 69% 66% 69% 70% 65% 67% 66% 61%
#columns_to_drop = ['aacpe', 'lzc'] #71% 73%
#recrear los campos que faltan en la base de datos database6c.csv
if database == "database6c.csv":
    df["eyeaxis"] = "x"
    df["delay"] = 0
    df["dimension"] = 0
# columns_to_maybe_drop = ['ylzc', 'aacpe', 'yaacpe', 'mpe', 'ympe', 'mnpe', 'ymnpe', 'aape', 'yaape', 'aanpe', 'yaanpe', 'wcpe', 'wpe', 'ymcpe', 'yecpe', 'yuqcpe', 'fgnpe', 'yfgnpe']
# columns_to_keep = ['ywcpe', 'fgcpe', 'ywpe', 'mcpe', 'epe', 'yepe', 'enpe', 'yenpe', 'uqpe', 'yuqpe', 'ecpe', 'wnpe', 'ywnpe', 'd2', 'yd2', 'shen', 'yshen', 'specen', 'yspecen', 'uqpe', 'yuqpe', 'uqnpe', 'yuqnpe', 'uqcpe', 'men', 'ymen', 'fgpe', 'yfgpe', 'fuzzyen', 'yfuzzyen', 'dfa', 'ydfa', 'apen', 'yapen', 'sampen', 'ysampen', 'reen', 'yreen', 'hfd', 'yhfd', 'kfd', 'ykfd', 'pfd', 'ypfd', 'lle', 'ylle', 'he', 'yhe'] #70% 72% 73% 73% 73% 72% 72% 72% 73% 72% 71% 69% 73% 73% 73% 68% 67% 72% 72% 71% 73% 72% 72% 70% 70% 71% 73% 69% 71% 71% 72% 72% 70% 70% 73% 73% 72% 73% 69% 71% 71% 70% 69% 66% 65% 68%
# columns_to_drop = ['lzc', 'yfgcpe'] #73% 74%
# df = df.drop(columns=columns_to_drop)

df_input = df.drop(columns=["delay", "dimension", "parkinson"])
df_y = df["parkinson"]
patient_ids = df["patient"].unique()

# Define models and their respective parameter grids for GridSearchCV
models = {
    "random_forest": (
        RandomForestClassifier(random_state=42),
        {
            "n_estimators": [100, 200],
            "max_depth": [None, 10, 20, 40],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
        },
    ),
    "svm": (
        SVC(probability=True, random_state=42),
        {
            "C": [0.001, 0.01, 0.1, 1, 10, 55],  # "C": [10, 20, 30, 40, 55],  #'C': [0.001, 0.01, 0.1, 1, 10, 55],
            "kernel": ["linear", "rbf"],
            "gamma": ["scale", "auto", 0.01, 0.001, 0.1, 1],
        },
    ),
    "logistic_regression": (
        LogisticRegression(solver="saga", max_iter=2000, random_state=42),
        {
            "penalty": ["elasticnet"],  # "l1", "l2", "elasticnet"],
            "C": [0.01, 0.1, 1, 10],
            "l1_ratio": [0.1, 0.5, 0.9],
        },
    ),
    "knn": (
        KNeighborsClassifier(),
        {
            "n_neighbors": [3, 5, 7],
            "weights": ["uniform", "distance"],
            "p": [
                1,
                2,
            ],  # p=1 for Manhattan distance (L1), p=2 for Euclidean distance (L2)
        },
    ),
}
#num_features = 10
if __name__ == '__main__':
    # Paralelizar el bucle principal
    num_features_range = range(65, 66)

    # Usar ThreadPoolExecutor para multithreading o ProcessPoolExecutor para multiprocessing
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_feature_selection, num_features) for num_features in num_features_range]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()  # Puedes manejar los resultados aquí, si es necesario

# exit
