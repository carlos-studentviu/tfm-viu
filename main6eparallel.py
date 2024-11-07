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
from sklearn.linear_model import ElasticNet
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import GridSearchCV, KFold, StratifiedGroupKFold
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.metrics import PrecisionRecallDisplay, precision_recall_curve
from sklearn.metrics import det_curve, DetCurveDisplay
from statistics import mode
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


def build_final_pipeline(X, num_features, normalizer="standard", model="random_forest"):
    preprocessor, scaler = build_preprocessor(X,normalizer)
    model_obj, param_grid = models[model]

    return (
        Pipeline(
            [
                ("preprocessor", preprocessor),
                ("reducer", PCA(n_components=num_features, random_state=42)), #n_components=65
                #("classifier", SVC(probability=True, random_state=42, C=1, gamma="scale", kernel='linear')),
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
        fpipeline, scaler = build_final_pipeline(x_train, num_features, normalizer="minmax", model="svm")  # model='svm')
        #fpipeline.fit(x_train, y_train)
        fpipeline.fit(x_train, y_train, classifier__groups=train_groups)

        # store in a file the best parameters
        # with open("best_params.txt", "a") as f:
        #     f.write(str(fpipeline.named_steps['classifier'].best_params_) + '\n')

        # Crear un pipeline solo con preprocesador y PCA de 65 componentes
        # preprocessor_pca_pipeline = Pipeline([
        #     ('preprocessor', fpipeline.named_steps['preprocessor']),
        #     ('pca', PCA(n_components=65, random_state=42))
        # ])
        #
        # # Ajustar solo el preprocesador y el PCA en x_train
        # preprocessor_pca_pipeline.fit(x_train)
        #
        # # Transformar x_train y x_test usando el pipeline de preprocesamiento y PCA
        # x_train_pca_65 = preprocessor_pca_pipeline.transform(x_train)
        # x_test_pca_65 = preprocessor_pca_pipeline.transform(x_test)
        #
        # # Seleccionar solo los primeros 15 componentes de x_train y x_test
        # x_train_pca_num = x_train_pca_65[:, :num_features]
        # x_test_pca_num = x_test_pca_65[:, :num_features]
        #
        # model_obj, param_grid = models["svm"]
        # # Crear un nuevo clasificador SVC y ajustarlo solo con los primeros 15 componentes
        # classifier = GridSearchCV(
        #                 model_obj,
        #                 param_grid,
        #                 cv=StratifiedGroupKFold(n_splits=5),
        #                 scoring="roc_auc",
        #                 verbose=0,
        #                 n_jobs=-1,
        #             )  # Usamos probability=True para obtener predict_log_proba
        # classifier.fit(x_train_pca_num, y_train, groups=train_groups)
        #
        # # Realizar predicciones con los primeros 15 componentes
        #
        # # best_model = pipeline.named_steps['classifier'].best_estimator_
        # # Realiza la predicción en base a los primeros num_features componentes
        # y_train_pred = classifier.predict_log_proba(x_train_pca_num)

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
                prediction = 100
            if np.isneginf(prediction):
                prediction = -100
            list_train_predictions.append(prediction)  # predicted)

        fpr, tpr, thresholds = roc_curve(true_labels_patient, list_train_predictions)
        best_threshold = thresholds[np.argmax(tpr - fpr)]
        list_best_thresholds.append(best_threshold)


        # y_pred = classifier.predict_log_proba(x_test_pca_num)
        y_pred = fpipeline.predict_log_proba(x_test)
        avg_1 = np.average(y_pred[:, 1])
        avg_0 = np.average(y_pred[:, 0])
        list_predictions.append(avg_1 - avg_0)
        if avg_1 - avg_0 >= best_threshold:  # avg_0:
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
    npz_filename = 'pcatest_classificationscores' + database.split('.')[0] + 'svmminmax' + str(num_features) + 'features.npz'
    np.savez(npz_filename, list_predictions=list_predictions, list_best_thresholds=list_best_thresholds,
                 list_pos_class_index=list_pos_class_index, list_neg_class_index=list_neg_class_index,
                 true_labels=true_labels, list_predict_binary=list_predict_binary)

    # plot_roc_curve(true_labels, predicted_probabilities, c)
    # fpr, tpr, thresholds = roc_curve(true_labels, list_predictions,c)
    # best_threshold = thresholds[np.argmax(tpr-fpr)]
    correct = tp + tn
    positive = tp + fn
    negative = tn + fp

    best_threshold_overall = mode(list_best_thresholds)

    print("best threshold: " + str(best_threshold_overall))
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
    results_filename = 'pcatest_results'+database.split('.')[0]+'svmminmax'+str(num_features)+'features.txt'
    with open(results_filename, "a") as f:
        f.write("=====================================================================\n")
        f.write("Database: " + database + "+svm minmax\n")
        f.write("Num features: " + str(num_features) + " backwards\n")
        f.write("best threshold: " + str(best_threshold_overall) + "\n")
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



    roc_filename = 'pcatest_roc'+database.split('.')[0]+'svmminmax'+str(num_features)+'features.png'
    rocch_filename = 'pcatest_rocch'+database.split('.')[0]+'svmminmax'+str(num_features)+'features.png'
    pr_filename = 'pcatest_pr'+database.split('.')[0]+'svmminmax'+str(num_features)+'features.png'
    det_filename = 'pcatest_det'+database.split('.')[0]+'svmminmax'+str(num_features)+'features.png'
    detch_filename = 'pcatest_detch'+database.split('.')[0]+'svmminmax'+str(num_features)+'features.png'
    score_filename = 'pcatest_score'+database.split('.')[0]+'svmminmax'+str(num_features)+'features.png'
    # plot the roc curve
    plot_roc_curve(true_labels, list_predictions, c, roc_filename)

    # plot the ROCCH curve
    fpr, tpr, _ = roc_curve(true_labels, list_predictions)
    F, T, auc = rocch.rocch(fpr, tpr)
    print("AUC ROCCH: " + str(auc))
    with open(results_filename, "a") as f:
        f.write("AUC ROCCH: " + str(auc) + "\n")


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

# df_pca = df_input.drop(columns=["patient", "smoothp", "eyeaxis"])
# preprocessor, scaler = build_preprocessor(df_pca, normalizer="minmax")
#
# # Aplicar el preprocesador (MinMaxScaler) a los datos de df_pca
# df_pca_scaled = preprocessor.fit_transform(df_pca)
#
# # Aplicar PCA con 23 componentes
# pca = PCA(n_components=23, random_state=42) #PCA(random_state=42) # PCA(n_components=23, random_state=42)
# df_pca_reduced = pca.fit_transform(df_pca_scaled)
#
# # Multiplicar los loadings por la varianza explicada para ponderar la importancia de cada componente
# weighted_components = np.abs(pca.components_) * pca.explained_variance_ratio_[:, np.newaxis]
#
# # Sumar la importancia ponderada de cada feature a lo largo de todos los componentes
# feature_importance = weighted_components.sum(axis=0)
#
# # Crear un DataFrame para mostrar la importancia de cada feature original
# feature_importance_df = pd.DataFrame({
#     'Feature': df_pca.columns,
#     'Importance': feature_importance
# })
#
# # Ordenar las características por importancia en orden descendente
# feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)
# print(feature_importance_df)
#
# # Guardar el DataFrame de importancia en un archivo de texto
# feature_importance_df.to_csv("feature_importance2.txt", sep='\t')
#
# # Opcional: Gráfica de la importancia de cada feature original
# plt.figure(figsize=(10, 6))
# plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='blue')
# plt.xlabel("Importancia")
# plt.ylabel("Feature Original")
# plt.title("Importancia de Cada Feature Original en el PCA")
# plt.gca().invert_yaxis()  # Invertir el eje Y para que la característica más importante esté arriba
# plt.show()

# # Ajustar PCA en los datos
# pca = PCA()
# pca.fit(df_pca_scaled)
#
# # Calcular la varianza explicada acumulada
# explained_variance = np.cumsum(pca.explained_variance_ratio_)
#
# # Graficar la varianza explicada acumulada para encontrar el punto de corte
# plt.figure(figsize=(10, 6))
# plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='-')
# plt.xlabel("Número de componentes principales")
# plt.ylabel("Varianza explicada acumulada")
# plt.title("Curva de varianza explicada acumulada para PCA")
# plt.grid(True)
# plt.show()
#
# for i in range(1, len(explained_variance) + 1):
#     print(f"Varianza acumulada con {i} componentes: {explained_variance[i - 1]}")
#
# # Seleccionar el punto de corte donde la varianza acumulada alcanza un umbral deseado, por ejemplo, el 95%
# n_components = np.argmax(explained_variance >= 0.95) + 1
# print(f"Número óptimo de componentes: {n_components}")

# exit(0)

#num_features = 10
if __name__ == '__main__':
    # Paralelizar el bucle principal
    num_features_range = range(23, 24)

    # Usar ThreadPoolExecutor para multithreading o ProcessPoolExecutor para multiprocessing
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_feature_selection, num_features) for num_features in num_features_range]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()  # Puedes manejar los resultados aquí, si es necesario

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
