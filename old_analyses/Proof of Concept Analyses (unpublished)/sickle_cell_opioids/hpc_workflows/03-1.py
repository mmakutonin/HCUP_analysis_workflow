# %% [markdown]
# ### Imports

# %%
import pandas as pd
import numpy as np
from utility_functions import load_file, pickle_file, starting_run, finished_run
from analysis_variables import de_col_keys, de_col_values, demographic_tables, code_category_dict
from scipy.stats import f_oneway, sem, norm, t
from statsmodels.stats.api import DescrStatsW, CompareMeans

full_dataset = load_file("summary_costs_enhanced.pickle")
filtered_dataset_codes = load_file("fully_filtered_codes.pickle")
comorbidities = load_file('comorbidities.pickle')

# %% [markdown]
# ### Demographic Column Classification

# %%
full_dataset["Married"] = full_dataset["marital_status"].eq('Married')
full_dataset["Uninsured"] = full_dataset["payer"].isin(['No charge', 'Self-pay'])
full_dataset["Medicare"] = full_dataset["payer"].eq('Medicare')
full_dataset["Medicaid"] = full_dataset["payer"].eq('Medicaid')
full_dataset["Private Insurance"] = full_dataset["payer"].eq('Private insurance')
full_dataset["White"] = full_dataset["race"].eq("White")
full_dataset["African American or Hispanic"] = full_dataset["race"].isin(['African-American', "Hispanic"])
full_dataset["Female"] = full_dataset["gender"].eq("Female")
full_dataset["Died"] = full_dataset["Died"].eq(1)

#rename some columns for cosmetic reasons:
full_dataset["Median Zip Code Income Quartile"] = full_dataset["median_zip_income"]
full_dataset["Age"] = full_dataset["age"]
full_dataset["Recurrent ED Visits (number per 1000 patients)"] = full_dataset["ED Readmissions"].mul(1000)
full_dataset["Repeat Hospitalizations (number per 1000 patients)"] = full_dataset["Inpatient Readmissions"].mul(1000)
full_dataset["Cost (USD)"] = full_dataset["Cost"]

del full_dataset["Admitted"] #step needed in this analysis due to "Admitted" column in de_col_values
for key in de_col_keys:
    full_dataset = full_dataset.join([full_dataset[key].eq(val).rename(val).loc[full_dataset.index] for val in de_col_values[key]])
# full_dataset = full_dataset.join([full_dataset[de_col_name].eq(val).rename(val) for val in de_col_values])

dem_dataset = full_dataset[[
    'Median Zip Code Income Quartile', 
    'Age',
    'Cost',
    "Married", 
    "Uninsured", 
    "Medicare",
    "Medicaid",
    "Private Insurance",
    "White",
    "African American or Hispanic",
    "Female",
    'Recurrent ED Visits (number per 1000 patients)',
    'Repeat Hospitalizations (number per 1000 patients)',
    'Died',
    'CMDF CCI',
    'LOS',
    *pd.core.common.flatten(de_col_values.values())
]].copy()

category_dict = {
    'Totals': 'Demographic',
    'Age': 'Demographic',
    'African American or Hispanic': 'Demographic',
    'Female': 'Demographic',
    'Married': 'Demographic',
    'Medicaid': 'Insurance Status',
    'Medicare': 'Insurance Status',
    'Private Insurance': 'Insurance Status',
    'Uninsured': 'Insurance Status',
    'White': 'Demographic',
    'Median Zip Code Income Quartile': 'Demographic',
    'CMDF CCI': 'Comorbidity',
    **{key: 'Comorbidity' for key in pd.core.common.flatten(list(code_category_dict.keys())[20:])}, #first 20 used to calculate CCI
    'Recurrent ED Visits (number per 1000 patients)': 'Outcome',
    'Repeat Hospitalizations (number per 1000 patients)': 'Outcome',
    'Cost': 'Outcome',
    'Died': 'Outcome',
    'LOS': 'Outcome',
    'Admitted': 'Clinical Pathway',
    **{value: 'Clinical Pathway' for value in pd.core.common.flatten(de_col_values.values())}
}

summary_table_sum_cols = [ #these are the columns that are not aggregates of proportions of patients
    "Cost", "CMDF CCI", "Age",
    "Median Zip Code Income Quartile",
    "Recurrent ED Visits (number per 1000 patients)",
    "Repeat Hospitalizations (number per 1000 patients)",'LOS'
]

# %% [markdown]
# ### Create Summary Table

# %%
def create_summary(groupby_col, filter_criteria="Cost >= 0"): #cost should always be positive, making this a universal filter
    num_full_dataset = dem_dataset.query(filter_criteria).join(comorbidities, how="left")\
        .fillna(0).astype("int").join(full_dataset[groupby_col]).groupby(groupby_col)
    agg_table = num_full_dataset.apply(lambda x: pd.Series(
        [DescrStatsW(x[column]) for column in x.columns],
        index=x.columns)).T.drop(groupby_col)
    summary_table = agg_table.transform(
        lambda row: [f"{round(val.mean, 2)} ({round(val.tconfint_mean(0.05)[0],2)}, {round(val.tconfint_mean(0.05)[1],2)})" for val in row] \
                    if row.name in summary_table_sum_cols else \
                    [f"{round(val.mean*100,0)}% ({round(val.tconfint_mean(0.05)[0]*100,0)}, {round(val.tconfint_mean(0.05)[1]*100,0)}), {round(val.sum, 0)}" for val in row],
        axis=1
    )
    summary_table.columns = [f"{col} (%, 95% CI, N)" for col in summary_table.columns]
    if(len(summary_table.columns) == 2):
        summary_table["Difference (95% CI)"] = agg_table.agg(
            lambda row: ("({} - {})" if row.name in summary_table_sum_cols else "({}% - {}%)").format(
                *[
                    round(val * (1 if row.name in summary_table_sum_cols else 100), 2)\
                    for val in CompareMeans(row[0], row[1]).tconfint_diff(0.05, usevar='unequal')
                ]
            ),
            axis=1
        )
        summary_table.loc["Totals"] = [*num_full_dataset.count().T.iloc[0], ' ']
    elif len(summary_table.columns) > 2:
        summary_table["ANOVA P"] = ["p < 0.01" if f_oneway(*[x for _, x in num_full_dataset[col]]).pvalue < 0.01 else "p > 0.01" for col in agg_table.index]
        summary_table.loc["Totals"] = [*num_full_dataset.count().T.iloc[0], ' ']
    else:
        summary_table.loc["Totals"] = [*num_full_dataset.count().T.iloc[0]]
    return summary_table.reindex(category_dict.keys())

# %%
for tb in demographic_tables:
    create_summary(tb["key"], tb["query_string"]).to_csv(tb["save_filepath"])

# %%



