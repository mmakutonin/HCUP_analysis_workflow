# # %%
# import warnings
# warnings.filterwarnings('ignore') #filters out annoying warnings in PCA output
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from statsmodels.discrete.discrete_model import Logit
from statsmodels.regression.linear_model import OLS
from scipy.stats import ttest_ind, norm, chisquare
from utility_functions import load_file, starting_run, print_to_drop, evaluate_model

demographic_cols = ['marital_status', 'initial_discharge_quarter', 'gender', 'race', 'payer', "Pediatric", "Geriatric"]
numerical_demographic_cols = ['median_zip_income', 'CMDF CCI']
pca = PCA(n_components=10)
enc = OneHotEncoder(sparse_output=False)
scaler = StandardScaler()
mScaler = MinMaxScaler()

def encode_dataset(dataset:pd.DataFrame):
    encoded_dataset = pd.DataFrame(
        enc.fit_transform(dataset), dataset.index
    )
    encoded_dataset.columns = enc.get_feature_names_out(dataset.columns)
    return encoded_dataset
def preprocess_dataset(analysis_name:str, dataset:pd.DataFrame, category_status:pd.DataFrame, code_category_dict:dict[str, list[str]], include_pca:bool):
    #encode dataset demographics
    dem_dataset = dataset.loc[:, demographic_cols].dropna()
    encoded_dataset = encode_dataset(dem_dataset).join(dataset[numerical_demographic_cols], how='inner')\
        .join(category_status.loc[:, list(code_category_dict.keys())[20:]], how="inner").dropna()
    print_to_drop(f"Dropped {dem_dataset.shape[0] - encoded_dataset.dropna().shape[0]} rows for 3.2 analysis due to missing demographics.", analysis_name)
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
    ) if include_pca else None
    return scaled_data, pca_data

def run_logreg(dataset:pd.DataFrame, target:pd.Series, results_output_file):
    logit_model = Logit(target, dataset).fit_regularized(maxiter=1000, disp=False)
    summary = logit_model.summary2().tables[1].sort_values(['P>|z|', 'Coef.'])
    summary["OR"] = summary['Coef.'].transform(np.exp)
    summary["OR CI"] = [norm.interval(confidence=0.95, loc=0, scale=item[1])[1] for item in summary['Std.Err.'].items()]
    summary["OR Formatted"] = summary.apply(
        lambda row: f"{round(row['OR'], 2)} ({round(row['OR']-row['OR CI'], 2)}, {round(row['OR']+row['OR CI'], 2)})",
        axis=1
    )
    # Cleaning up summary output cols:
    summary["Odds Ratio of ____ (95% CI)"] = summary["OR Formatted"]
    summary["P-value"] = summary["P>|z|"]
    output_summary = summary[["Odds Ratio of ____ (95% CI)", "P-value"]].round(5).map(
        lambda cell: "< 0.00001" if cell == 0.0 else cell
    )
    # Model eval printing:
    for line in evaluate_model(target, logit_model.predict(dataset), len(dataset.columns)):
        results_output_file.write(line + " \n")
    results_output_file.write(f"Numer of Groups: {len(dataset.columns)} \n")
    return output_summary, pd.DataFrame({
        "predicted": logit_model.predict(dataset),
        "expected": target.astype("int")
    })

def run_PCA(dataset):
    print(dataset.shape)
    fitted_model = pca.fit(dataset)
    return pd.DataFrame(
        scaler.fit_transform(fitted_model.transform(dataset)), index=dataset.index
    ), fitted_model.explained_variance_ratio_, pd.DataFrame(fitted_model.components_.T, index=dataset.columns)

def ttest(target, outcome_cols, features):
    def ci(col, is_true, dataset):
        if col["type"] == "string":
            dataset[col["name"]] = dataset[col["name"]].map({col["positive_class"]: 1}).fillna(0)
        mean, sem = dataset.loc[dataset["target"] == is_true][col["name"]].agg(["mean", "sem"])
        return f"{round(mean, 4)} ± {round(norm.interval(confidence=0.95,loc=0,scale=sem)[1], 4)}"
    dataset = features.join(target.rename("target"), how="inner")
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

def run_multivariate_analyses(
        outcome_cols: list[dict[str, str]],
        code_category_dict: dict[str, list[str]],
        logreg_targets: dict[str, callable],
        linreg_targets: dict[str, callable],
        analysis_name:str,
        include_pca:bool=True
    ):
    data = load_file("summary_costs_enhanced.pickle", analysis_name)
    category_status = load_file("comorbidities.pickle", analysis_name)
    if not os.path.isdir(f"../results/{analysis_name}/tables/logreg/"):
        os.mkdir(f"../results/{analysis_name}/tables/logreg/")
    if not os.path.isdir(f"../results/{analysis_name}/tables/linreg/"):
        os.mkdir(f"../results/{analysis_name}/tables/linreg/")
    if not os.path.isdir(f"../results/{analysis_name}/tables/ttest/"):
        os.mkdir(f"../results/{analysis_name}/tables/ttest/")
    
    starting_run("LogReg")
    scaled_data, pca_data = preprocess_dataset(analysis_name, data, category_status, code_category_dict, include_pca)
    if include_pca:
        pca_dataset, component_importance, component_eigenvalues = run_PCA(pca_data)
    data = data.loc[scaled_data.index]
    for target_name, target_function in logreg_targets.items():
        starting_run(target_name)
        with open(f"../results/{analysis_name}/tables/ttest/{target_name}.txt", 'w') as f:
            target_data = target_function(data)
            f.write(target_name + " \n")
            f.write(target_data.value_counts().to_string() + " \n")
            f.write("Logreg Results: \n")
            try:
                summary, model_eval = run_logreg(scaled_data.loc[target_data.index], target_data, f)
                summary.to_csv(f"../results/{analysis_name}/tables/logreg/{target_name} Feature Scores.csv")
                model_eval.to_csv(f"../results/{analysis_name}/tables/logreg/Model Eval {target_name} Feature Scores.csv")
                if(include_pca):
                    f.write("PCA Results: \n")
                    run_logreg(pca_dataset.loc[target_data.index], target_data, f)[0].to_csv(f"../results/{analysis_name}/tables/logreg/{target_name} PCA Component Scores.csv")
                    component_eigenvalues.to_csv(f"../results/{analysis_name}/tables/logreg/PCA eigenvalues.csv")
                    pd.DataFrame(component_importance).to_csv(f"../results/{analysis_name}/tables/logreg/PCA explained variance.csv")
                f.write(ttest(target_data, outcome_cols, data).to_string())
            except Exception as e:
                f.write(f"Error: could not run Logreg Stats: {e}")

    starting_run("Linreg")
    for target_name, target_function in linreg_targets.items():
        starting_run(target_name)
        with open(f"../results/{analysis_name}/tables/ttest/{target_name}.txt", 'w') as f:
            target_data = target_function(data)
            f.write(target_name + " \n")
            f.write(target_data.describe().to_string() + " \n")
            f.write("Linreg Results: \n")
            try:
                results = OLS(target_data, scaled_data.loc[target_data.index]).fit_regularized()
                summary = results.summary().tables[1].sort_values(['P>|z|', 'Coef.'])
                summary["Coef. CI"] = [norm.interval(confidence=0.95, loc=0, scale=item[1])[1] for item in summary['Std.Err.'].items()]
                summary["Odds Ratio of ____ (95% CI)"] = summary.apply(
                    lambda row: f"{round(row['Coef.'], 2)} ({round(row['Coef.']-row['Coef. CI'], 2)}, {round(row['Coef.']+row['Coef. CI'], 2)})",
                    axis=1
                )
                # Cleaning up summary output cols:
                summary["P-value"] = summary["P>|z|"]
                output_summary = summary[["Odds Ratio of ____ (95% CI)", "P-value"]].round(5).map(
                    lambda cell: "< 0.00001" if cell == 0.0 else cell
                )
                output_summary.to_csv(f"../results/{analysis_name}/tables/linreg/{target_name} Feature Scores.csv")
                f.write(f"R^2: {results.rsquared}")
            except Exception as e:
                f.write(f"Error: could not run LinReg Stats: {e}")

    # TODO: Very rough chi-square test. Decide whether to add to workflow in a better format.
    # for val in de_col_values[de_col_keys[1]]:
    #     full_data = data.join(category_status).query(f"`{de_col_keys[1]}` == '{val}'")
    #     print(val, chisquare(
    #         full_data.groupby(['Obesity', 'Mood Disorders'])['initial_record_id'].count(),
    #         list(full_data.groupby('Obesity')['initial_record_id'].count()/2)*2
    #     )[1])