# %%
import warnings
warnings.filterwarnings('ignore') #filters out annoying warnings in PCA output

# %%
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from statsmodels.discrete.discrete_model import Logit
from sklearn.metrics import accuracy_score
from scipy.stats import ttest_ind, f_oneway, norm, sem, chisquare
from utility_functions import load_file, pickle_file, starting_run, finished_run
from analysis_variables import logreg_targets, de_col_keys, de_col_values, outcome_cols, code_category_dict

# %%
data = load_file("summary_costs_enhanced.pickle")
category_status = load_file("comorbidities.pickle")
pca = PCA(n_components=10)
enc = OneHotEncoder(sparse=False)
scaler = StandardScaler()
mScaler = MinMaxScaler()
if not os.path.isdir(f"../tables/logreg"):
    os.mkdir(f"../tables/logreg")
if not os.path.isdir(f"../tables/ttest"):
    os.mkdir(f"../tables/ttest")

# %%
demographic_cols = ['marital_status', 'initial_discharge_quarter', 'gender', 'race', 'payer']
numerical_demographic_cols = ['age', 'median_zip_income', 'CMDF CCI']
def encode_dataset(dataset):
    encoded_dataset = pd.DataFrame(
        enc.fit_transform(dataset), dataset.index
    )
    encoded_dataset.columns = enc.get_feature_names(dataset.columns)
    return encoded_dataset
def preprocess_dataset(dataset):
    #encode dataset demographics
    dem_dataset = dataset.loc[:, demographic_cols].dropna()
    encoded_dataset = encode_dataset(dem_dataset).join(dataset[numerical_demographic_cols], how='inner')\
        .join(category_status.loc[:, list(code_category_dict.keys())[20:]], how="inner")
    outer_encoded_dataset = encode_dataset(dem_dataset).join(dataset[numerical_demographic_cols], how='outer')\
        .join(category_status.loc[:, list(code_category_dict.keys())[20:]], how="outer") # only computed to find dropped row number.
    print(f"Dropped {outer_encoded_dataset.shape[0] - encoded_dataset.shape[0]} rows for 3.2 analysis due to missing demographics.")
    #scale columns
    scaled_data = pd.DataFrame(
        scaler.fit_transform(encoded_dataset),
        index = encoded_dataset.index,
        columns = encoded_dataset.columns
    )
    pca_data = pd.DataFrame(
        mScaler.fit_transform(encoded_dataset),
        index = encoded_dataset.index,
        columns = encoded_dataset.columns
    )
    return scaled_data, pca_data, encoded_dataset

# %%
def run_logreg(dataset, target, print_output_file):
    logit_model = Logit(target, dataset).fit_regularized(maxiter=1000, disp=False)
    accuracy = round(accuracy_score(
        logit_model.predict(dataset).apply(round),
        target
    ), 2)
    summary = logit_model.summary2().tables[1].sort_values(['P>|z|', 'Coef.'])
    summary["OR"] = summary['Coef.'].transform(np.abs).transform(np.exp)
    summary["OR CI"] = [norm.interval(alpha=0.95, loc=0, scale=item[1])[1] for item in summary['Std.Err.'].iteritems()]
    print_output_file.write(f"Logistic Regression Accuracy: {accuracy}")
    return summary

# %%
def run_PCA(dataset):
    print(dataset.shape)
    fitted_model = pca.fit(dataset)
    return pd.DataFrame(
        scaler.fit_transform(fitted_model.transform(dataset)), index=dataset.index
    ), fitted_model.explained_variance_ratio_, pd.DataFrame(fitted_model.components_.T, index=dataset.columns)

# %%
def ttest(target):
    def ci(col, is_true, dataset):
        if col["type"] == "string":
            dataset[col["name"]] = dataset[col["name"]].map({col["positive_class"]: 1}).fillna(0)
        mean, sem = dataset.loc[dataset["target"] == is_true][col["name"]].agg(["mean", "sem"])
        return f"{round(mean, 4)} ± {round(norm.interval(alpha=0.95,loc=0,scale=sem)[1], 4)}"
    dataset = data.join(target.rename("target"), how="inner")
    stats = pd.DataFrame({
        'Metric': [col["name"] for col in outcome_cols],
        'Mean True': [ci(col, True, dataset) for col in outcome_cols],
        'Mean False': [ci(col, False, dataset) for col in outcome_cols],
        'P value': [
            ttest_ind(
                dataset.loc[dataset["target"] == True][col["name"]],
                dataset.loc[dataset["target"] == False][col["name"]],
                equal_var=False
            ).pvalue*2 for col in outcome_cols
        ]
    })
    return stats

# %%
starting_run("LogReg")
scaled_data, pca_data, encoded_dataset = preprocess_dataset(data)
pca_dataset, component_importance, component_eigenvalues = run_PCA(pca_data)
data = data.loc[scaled_data.index]
for target_name, target_function in logreg_targets.items():
    starting_run(target_name)
    with open(f"../tables/ttest/{target_name}.txt", 'w') as f:
        target_data = target_function(data)
        f.write(target_name)
        f.write("\n")
        f.write(target_data.value_counts().to_string())
        f.write("\n")
        f.write("Logreg Results:")
        run_logreg(scaled_data.loc[target_data.index], target_data, f).to_csv(f"../tables/logreg/{target_name} Feature Scores.csv")
        f.write("\n")
        f.write("PCA Results:")
        run_logreg(pca_dataset.loc[target_data.index], target_data, f).to_csv(f"../tables/logreg/{target_name} PCA Component Scores.csv")
        f.write("\n")
        f.write(ttest(target_data).to_string())
component_eigenvalues.to_csv(f"../tables/logreg/PCA eigenvalues.csv")
pd.DataFrame(component_importance).to_csv(f"../tables/logreg/PCA explained variance.csv")

# %%
# for val in de_col_values[de_col_keys[1]]:
#     full_data = data.join(category_status).query(f"`{de_col_keys[1]}` == '{val}'")
#     print(val, chisquare(
#         full_data.groupby(['Obesity', 'Mood Disorders'])['initial_record_id'].count(),
#         list(full_data.groupby('Obesity')['initial_record_id'].count()/2)*2
#     )[1])


