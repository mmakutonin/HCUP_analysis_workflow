# %%
import pandas as pd
import numpy as np
from utility_functions import load_file, pickle_file, starting_run, finished_run
from data_reading_functions import code_lengths
from analysis_variables import procedure_codes, data_enrichment_function, code_category_dict

# %% [markdown]
# ### Imports & File Loading

# %%
sedd = load_file("sedd_core_filtered.pickle")
sasd = load_file("sasd_core_filtered.pickle")
sid = load_file("sid_core_filtered.pickle")
sid_ed = load_file("sedd_appendix.pickle")
null_codes = {name: '                       '[:length] for name, length in code_lengths.items()}

# %% [markdown]
# ### Utility Functions

# %%
def create_linker_table(sedd, sid_ed, sid):
    def create_linker_table(dataset, sid_flag):
        join_dataset = sid if sid_flag else dataset
        dataset = dataset.reset_index().groupby("visit_link")[["record_id", "year"]].min().join(
            join_dataset[["age", "female", "homeless", "race", "married", "median_zip_income", "payer", "discharge_quarter"]],
            on="record_id"
        ).rename(columns={
            "record_id": "initial_record_id",
            "year": "initial_year",
            "discharge_quarter": "initial_discharge_quarter"
        })
        dataset["Admitted"] = sid_flag
        return dataset
    #init linker_table with year and record_id of initial ED visit
    linker_table = create_linker_table(sedd, False).append(
        create_linker_table(sid_ed, True)
    ).sort_values(["initial_year", "initial_discharge_quarter"])\
    .reset_index().drop_duplicates("visit_link", keep="first")\
    .set_index("visit_link")
    print(f"Dropped {(linker_table['initial_year'] >= 2018).sum()} patients because initial_visit year = 2018")
    linker_table = linker_table[linker_table["initial_year"] < 2018]
    
    #add max_year based on initial_year (assume following for 1 year)
    linker_table["max_year"] = (linker_table["initial_year"] + 1)
    
    print(f"Dropped {(linker_table.index <= 0).sum()} patients because index was non-positive.")
    return linker_table.loc[linker_table.index > 0, :]

# %%
def censor_first_6_mos(linker_table):
    min_year = linker_table["initial_year"].min()
    censored_table = linker_table.query(
        f"initial_year > {min_year} or initial_discharge_quarter > 2"
    )
    print(f"Dropped {linker_table.shape[0] - censored_table.shape[0]} patients by censoring first 6 months.")
    return censored_table

# %%
def filter_data_on_year(sedd, sasd, sid, sid_ed, linker_table):
    return (
        dataset.loc[dataset[["visit_link", "year", "discharge_quarter"]].join(
            linker_table[["initial_year", "max_year", "initial_discharge_quarter"]], on="visit_link"
        ).query(
            "(initial_year == year and initial_discharge_quarter <= discharge_quarter) or (max_year == year and initial_discharge_quarter >= discharge_quarter)"
        ).index] for dataset in [sedd, sasd, sid, sid_ed]
    )

# %%
def count_admits(sedd, sasd, sid, sid_ed, linker_table):
    def count_visits(dataset, col_name):
        return dataset.join(linker_table, on="visit_link", rsuffix="_x")\
        .query('initial_record_id != record_id').groupby("visit_link")\
        .count()["year"].rename(col_name)
    return linker_table.join(
        count_visits(sedd, "ED Readmissions")\
        .add(count_visits(sid_ed, "ED Readmissions"), fill_value=0)
    ).join(
        count_visits(sasd, "Surgery Visits")).join(
        count_visits(sid, "Inpatient Readmissions")).fillna(0)

# %%
def create_code_lookup_table(sedd, sasd, sid, linker_table):
    def preprocess_dataset_on_init_chart(dataset):
        return linker_table.join(
            dataset, on="initial_record_id", how="inner", rsuffix="_x"
        ).reset_index(drop=True).groupby("visit_link")
    def preprocess_dataset(dataset):
        return dataset.join(
            linker_table, on="visit_link", how="inner", rsuffix="_x"
        ).query("initial_record_id != record_id").groupby("visit_link")
    def postprocess_dataset(dataset, code_type):
        return pd.DataFrame(
            dataset[code_type].agg(np.sum).explode()\
            .replace(null_codes[code_type], np.nan).dropna()\
            .rename("codes").astype("str")
        )
    
    dataset_list = [
        {
            "dataset": sedd,
            "pcs_type": "cpt_codes",
            "pcs_flag": "cpt_flag",
            "ED_chart": True
        },
        {
            "dataset": sasd,
            "pcs_type": "cpt_codes",
            "pcs_flag": "cpt_flag",
            "ED_chart": False
        },
        {
            "dataset": sid.loc[sid["ed_admission"] <= 0],
            "pcs_type": "ICD-10-procedures",
            "pcs_flag": "icd_proc_flag",
            "ED_chart": False
        },
        {
            "dataset": sid.loc[sid["ed_admission"] > 0],
            "pcs_type": "ICD-10-procedures",
            "pcs_flag": "icd_proc_flag",
            "ED_chart": True
        },
        
    ]
    data_list = []
    for dataset in dataset_list:
        for flag in [{"pcs_type":"ICD-10","pcs_flag":"icd_flag"},dataset]:
            data = postprocess_dataset(preprocess_dataset(dataset["dataset"]), flag["pcs_type"])
            data[flag["pcs_flag"]] = True
            if dataset["ED_chart"] == True:
                data['ed_flag'] = True
                #Only ED charts can be initial visits
                data_init = postprocess_dataset(preprocess_dataset_on_init_chart(dataset["dataset"]), flag["pcs_type"])
                data_init[flag["pcs_flag"]] = True
                data_init['init_chart'] = True
                data_init['ed_flag'] = True
                data_list.append(data_init)
            data_list.append(data)
    return pd.concat(data_list).fillna(False)

# %%
def enrich_comorbidities(codes):
    category_list = []
    visit_codes = codes.reset_index().\
        groupby("visit_link")["codes"].unique().apply(lambda x: [st.strip() for st in x])
    for key, values in code_category_dict.items():
        category_list.append(visit_codes.transform(
            lambda x: any([any([code.startswith(value) for value in values]) for code in x])
        ).rename(key))
    comorbidities = pd.DataFrame(category_list).astype("int").T
    comorbidities.index = comorbidities.index
    return comorbidities

# %%
def calculate_cci_score(linker_table, comorbidities): # based on https://www.mdcalc.com/charlson-comorbidity-index-cci#evidence
    cci = comorbidities.agg(
        lambda x: 1 if x["Myocardial Infarction History"] else 0 + \
            1 if x['Heart Failure'] else 0 + \
            1 if x['Peripheral Vascular Disease'] else 0 + \
            2 if x['Hemiplegia'] else 1 if x['CVA/TIA'] else 0 + \
            1 if x['Dementia'] else 0 + \
            1 if x['COPD'] else 0 + \
            1 if x['Rheumatic Disease'] else 0 + \
            1 if x['Peptic Ulcer Disease'] else 0 + \
            3 if x['Severe Liver Disease'] else 1 if x['Mild Liver Disease'] else 0 + \
            2 if x['Complicated Diabetes'] else 1 if x['Uncomplicated Diabetes'] else 0 + \
            3 if x['Severe Renal Disease'] else 1 if x['Uncomplicated Renal Disease'] else 0 + \
            6 if x['Metastatic Tumor'] else 0 if x['Invalid Malignancy'] else 1 if x['Malignancy'] else 0 + \
            6 if x['HIV'] and x['AIDS Opportunistic Infection'] else 3 if x['HIV'] else 0
        , axis=1)
    linker_table['CMDF CCI'] = cci
    return linker_table

# %%
def calc_charges(sedd, sid, linker_table):
    def charges_for_dataset(dataset):
        return dataset.set_index("visit_link")["total_charges"].reset_index().groupby("visit_link")["total_charges"].sum()
    linker_table = linker_table.join(
        charges_for_dataset(sedd).rename("SEDD Charges"), how="left").join(
        charges_for_dataset(sid).rename("SID Charges"), how="left")
    linker_table["SEDD Charges"].fillna(0, inplace=True)
    linker_table["SID Charges"].fillna(0, inplace=True)
    return linker_table

# %%
def calc_LOS(linker_table, sedd, sid, sasd):
    def calc_dataset_los(dataset):
        return dataset.query("length_of_stay >= 0").groupby("visit_link")["length_of_stay"].sum()
    linker_table = linker_table.join(
        calc_dataset_los(sedd).rename("ED LOS"), how="left")\
        .join(calc_dataset_los(sid).rename("Inpatient LOS"), how="left")\
        .join(calc_dataset_los(sasd).rename("Outpatient LOS"), how="left")
    linker_table["LOS"] = linker_table["ED LOS"].fillna(0) \
    + linker_table["Inpatient LOS"].fillna(0) + linker_table["Outpatient LOS"].fillna(0)
    return linker_table

# %% [markdown]
# ### Main Code

# %%
starting_run("process full datasets")
linker_table = create_linker_table(sedd, sid_ed, sid)
linker_table = censor_first_6_mos(linker_table)
sedd, sasd, sid, sid_ed = filter_data_on_year(sedd, sasd, sid, sid_ed, linker_table)
linker_table = count_admits(sedd, sasd, sid, sid_ed, linker_table)
codes = create_code_lookup_table(sedd, sasd, sid, linker_table)
comorbidities = enrich_comorbidities(codes)
linker_table = calculate_cci_score(linker_table, comorbidities)
linker_table = calc_charges(sedd, sid, linker_table)
linker_table = calc_LOS(linker_table, sedd, sid, sasd)
linker_table = data_enrichment_function(sedd, sasd, sid, sid_ed, codes, linker_table)
starting_run("store datasets")
pickle_file("filtered_dataset.pickle", linker_table)
pickle_file("filtered_dataset_codes.pickle", codes)
pickle_file("filtered_sid_data.pickle", sid)
pickle_file("filtered_sedd_data.pickle", sedd)
pickle_file("filtered_sid_ed_data.pickle", sid_ed)
pickle_file("filtered_sasd_data.pickle", sasd)
pickle_file("comorbidities.pickle", comorbidities)
finished_run()


