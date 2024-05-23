import os
import pickle
from datetime import datetime
from sklearn.metrics import accuracy_score, roc_curve, auc
from scipy.stats import chi2
import pandas as pd
import numpy as np

# loads a pickled file using a relative path from the ../data/ directory.
def load_file(data_path):
    with open('../pickled_data/' + data_path, "rb") as input_file:
        return pickle.load(input_file)

# saves a Python object to a pickled file using a relative path from the ../data/ directory.
def pickle_file(data_path, python_obj):
    with open('../pickled_data/' + data_path, 'wb') as pickle_file:
        pickle.dump(python_obj, pickle_file, protocol=pickle.DEFAULT_PROTOCOL)

# prints to file tracking dropped patients.
def print_to_drop(print_string):
    with open(f"../tables/dropped_patients.txt", 'a') as f:
        f.write(print_string + " \n")
        
# prints operation start dialogue
def starting_run(operation_name=''):
    print('Starting ' + operation_name + ' ' + str(datetime.now().time()))

# prints operation completion dialogue
def finished_run(operation_name=''):
    print('Finished ' + operation_name + ' ' + str(datetime.now().time()))

# reports model accuracy, AUC, and HL p-value
def evaluate_model(target, model_predictions, n_groups):
    return [
        f"Area under the curve (AUC): {auc_calc(target, model_predictions)}",
        f"Model Accuracy: {model_accuracy(target, model_predictions)}",
        f"Hosmer-Lemeshow Statistics: {hl_test(target, model_predictions, n_groups)}"
    ]

def auc_calc(target, model_predictions):
    fpr, tpr, thresholds = roc_curve(target,model_predictions)
    return round(auc(fpr, tpr), 2)

def model_accuracy(target, model_predictions):
    return round(accuracy_score(model_predictions.apply(round),target), 2)

def hl_test(target, model_predictions, n_groups):
    # groups rows into evenly divided percentiles
    groups = pd.cut(
        model_predictions,
        np.percentile(model_predictions,[100/n_groups*i for i in range(n_groups+1)]),
        labels=False,
        include_lowest=True
    )
    # sums and means for model predictions and real values
    hl_data = pd.DataFrame({
        "pihat": model_predictions,
        "pihat_inv": 1-model_predictions,
        "group": groups,
        "target": target,
        "target_inv": 1 - target
        }).groupby("group").agg(["sum", "mean"])
    hl_data = pd.DataFrame({
        "expected events": hl_data[("pihat", "mean")] * hl_data[("pihat", "sum")],
        "expected non-events": hl_data[("pihat_inv", "mean")] * hl_data[("pihat_inv", "sum")],
        "observed events": hl_data[("target", "sum")],
        "observed non-events": hl_data[("target_inv", "sum")]
    })
    # calculated using sum((obs - expected)^2/expected + (non_obs - non_expected)^2/non_expected)
    hl_statistic = (
        (hl_data["observed events"]-hl_data["expected events"])**2/hl_data["expected events"] + \
        (hl_data["observed non-events"]-hl_data["expected non-events"])**2/hl_data["expected non-events"]
    ).sum()
    chi2_stat = chi2.cdf(hl_statistic,n_groups-2)
    return f"chi2 statistic: {round(chi2_stat,2)}, p: {round(1-chi2_stat,2)}"