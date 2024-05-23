# %%
import pandas as pd
import numpy as np
import os
from data_reading_functions import data_dir, read_data, core_reference, code_lengths
from analysis_variables import procedure_codes, diagnosis_codes, dataset_filtering_function
from utility_functions import pickle_file, starting_run, finished_run

# %%
if not os.path.isdir(f"../pickled_data/"):
    os.mkdir(f"../pickled_data/")

# %% [markdown]
# ### Data Reading

# %%
split_codes = lambda val, col_name: [val[i:i+code_lengths[col_name]] for i in range(0, len(val), code_lengths[col_name])]

def process_dataset(dataset, proc_code_type):
    dataset_core = read_data(core_reference[dataset]["2018"], f"MD_{dataset.upper()}_2018_CORE.asc").append(
        read_data(core_reference[dataset]["2017"], f"MD_{dataset.upper()}_2017_CORE.asc")
    ).append(
        read_data(core_reference[dataset]["2016"], f"MD_{dataset.upper()}_2016_CORE.asc"), ignore_index=True
    )
    
    dataset_core = dataset_filtering_function(dataset, dataset_core, proc_code_type)
    dataset_core = dataset_core.astype(core_reference[dataset]["dtypes"]).set_index("record_id")
    dataset_core["ICD-10"] = dataset_core["ICD-10"].transform(split_codes, col_name="ICD-10")
    dataset_core[proc_code_type] = dataset_core[proc_code_type].transform(split_codes, col_name=proc_code_type)
    pickle_file(f"{dataset}_core_filtered.pickle", dataset_core)
    del dataset_core


# %%
process_dataset("sedd", "cpt_codes")
process_dataset("sasd", "cpt_codes")
process_dataset("sid", "ICD-10-procedures")


