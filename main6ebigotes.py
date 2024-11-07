#!/usr/bin/env python
# -*- coding: utf-8 -*-
# este main utiliza dos leave-one-out, uno para elegir la mejor selección de características y otro para evaluar el modelo
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
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
from sklearn.linear_model import enet_path
from sklearn.linear_model import ElasticNet
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, StratifiedGroupKFold, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.metrics import PrecisionRecallDisplay, precision_recall_curve
from sklearn.metrics import det_curve, DetCurveDisplay
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import warnings
import rocch as rocch
from tqdm import tqdm

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


def build_final_pipeline(X, num_features, normalizer="standard", model="random_forest"):
    preprocessor, scaler = build_preprocessor(X,normalizer)
    model_obj, param_grid = models[model]

    return (
        Pipeline(
            [
                ("preprocessor", preprocessor),
                ("reducer", PCA(n_components=num_features, random_state=42)), #n_components=65
                (
                    "classifier",
                    GridSearchCV(
                        model_obj,
                        param_grid,
                        cv=StratifiedGroupKFold(n_splits=5, shuffle=False),
                        scoring="roc_auc",
                        verbose=0,
                        n_jobs=-1,
                    ),
                ),
                #("classifier", SVC(probability=True, random_state=42, C=1, gamma=0.001, kernel='rbf')),
            ]
        ),
        scaler,
    )

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
            "C": [0.001,0.01, 0.1, 1, 10],
            "l1_ratio": [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
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

# Diccionario para almacenar los accuracy por número de componentes
accuracy_scores = {n: [] for n in range(1, 65)}
#num_features = 10
for num_features in range(1, 20):
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

    # Configuración de grupos para el StratifedGroupKFold
    train_groups = df_input["patient"]
    # Configuración de cross-validation
    cv = StratifiedGroupKFold(n_splits=5, shuffle=False)

    df_x = df_input.drop(columns=["patient", "smoothp", "eyeaxis"])
    scores = []

    for train_idx, test_idx in cv.split(df_x, df_y, groups=train_groups):
        x_train, x_test = df_x.iloc[train_idx], df_x.iloc[test_idx]
        y_train, y_test = df_y.iloc[train_idx], df_y.iloc[test_idx]

        fpipeline, scaler = build_final_pipeline(x_train, num_features, normalizer="minmax", model="svm")
        fpipeline.fit(x_train, y_train, classifier__groups=train_groups[train_idx])

        # Evaluar en el conjunto de prueba
        score = fpipeline.score(x_test, y_test)
        scores.append(score)

    print(f"Accuracy scores for {num_features} features: {scores}")

    # fpipeline, scaler = build_final_pipeline(df_x, num_features, normalizer="minmax", model="svm")  # model='svm')
    # fpipeline.fit(df_x, df_y) #, classifier__groups=train_groups)
    #
    # scores = cross_val_score(fpipeline, df_x, df_y, cv=cv, groups=train_groups, scoring='accuracy')
    # accuracy_scores[num_features] = scores  # Guardar los accuracy de cada fold

    #print(f"Accuracy scores for {num_features} features: {scores}")

    # Convertir a DataFrame para facilitar el plotting
    #accuracy_df = pd.DataFrame.from_dict(accuracy_scores, orient='index')

    continue

    # Lets do a kfold to create a leaveone-out for the evaluation of the classifier
    kf = KFold(n_splits=100, shuffle=False)
    # Loop over each patient for the outer CV
    for train_index, test_index in tqdm(kf.split(patient_ids)):

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

        # # Use now a SVM with the best parameters
        fpipeline, scaler = build_final_pipeline(x_train, num_features, normalizer="minmax",model="svm")  # model='svm')
        fpipeline.fit(x_train, y_train, classifier__groups=train_groups)

        # best_model = pipeline.named_steps['classifier'].best_estimator_

        # We get the predictions for the training set, to be used as the input to get the best threshold
        y_train_pred = fpipeline.predict_log_proba(x_train)

        for patient in train_patients:
            patient_idx = train_instances["patient"].isin([patient])
            y_train_pred_idx = y_train_pred[patient_idx]
            y_train_idx = y_train[patient_idx]
            avg_1 = np.average(y_train_pred_idx[:, 1])
            avg_0 = np.average(y_train_pred_idx[:, 0])

            true_labels_patient.append(y_train_idx.array[0])
            prediction = avg_1 - avg_0
            if np.isposinf(prediction):
                result = 100
            if np.isneginf(prediction):
                result = -100
            list_train_predictions.append(prediction)  # predicted)

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
    npz_filename = 'classificationscores'+database.split('.')[0] + 'svmminmax' + str(num_features) + 'features.npz'
    np.savez(npz_filename, list_predictions=list_predictions, list_best_thresholds=list_best_thresholds,
             list_pos_class_index=list_pos_class_index, list_neg_class_index=list_neg_class_index,
            true_labels=true_labels, list_predict_binary=list_predict_binary)

    # plot_roc_curve(true_labels, predicted_probabilities, c)
    # fpr, tpr, thresholds = roc_curve(true_labels, list_predictions,c)
    # best_threshold = thresholds[np.argmax(tpr-fpr)]
    correct = tp + tn
    positive = tp + fn
    negative = tn + fp

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
    results_filename = 'results'+database.split('.')[0]+'svmminmax'+str(num_features)+'features.txt'
    with open(results_filename, "a") as f:
        f.write("=====================================================================\n")
        f.write("Database: " + database + "+svm minmax\n")
        f.write("Num features: " + str(num_features) + " backwards\n")
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



    roc_filename = 'roc'+database.split('.')[0]+'svmminmax'+str(num_features)+'features.png'
    rocch_filename = 'rocch'+database.split('.')[0]+'svmminmax'+str(num_features)+'features.png'
    pr_filename = 'pr'+database.split('.')[0]+'svmminmax'+str(num_features)+'features.png'
    det_filename = 'det'+database.split('.')[0]+'svmminmax'+str(num_features)+'features.png'
    detch_filename = 'detch'+database.split('.')[0]+'svmminmax'+str(num_features)+'features.png'
    score_filename = 'score'+database.split('.')[0]+'svmminmax'+str(num_features)+'features.png'
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


# exit
