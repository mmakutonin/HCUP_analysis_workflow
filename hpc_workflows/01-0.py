import pandas as pd
import os
from data_reading_functions import read_data, core_reference, code_lengths
from utility_functions import pickle_file

def import_data(analysis_name: str, dataset_filtering_function: callable):
    if not os.path.isdir(f"../pickled_data/"):
        os.mkdir(f"../pickled_data/")
    if not os.path.isdir(f"../results/"):
        os.mkdir(f"../results/")
    if not os.path.isdir(f"../pickled_data/{analysis_name}/"):
        os.mkdir(f"../pickled_data/{analysis_name}/")
    if not os.path.isdir(f"../results/{analysis_name}/"):
        os.mkdir(f"../results/{analysis_name}/")
        os.mkdir(f"../results/{analysis_name}/figures/")
        os.mkdir(f"../results/{analysis_name}/tables/")
        
    split_codes = lambda val, col_name: [val[i:i+code_lengths[col_name]] for i in range(0, len(val), code_lengths[col_name])]

    def process_dataset(dataset, proc_code_type):
        dataset_core = pd.concat([
            read_data(core_reference[dataset]["2021"], f"MD_{dataset.upper()}_2021_CORE.asc"),
            read_data(core_reference[dataset]["2020"], f"MD_{dataset.upper()}_2020_CORE.asc"),
            read_data(core_reference[dataset]["2019"], f"MD_{dataset.upper()}_2019_CORE.asc"),
            read_data(core_reference[dataset]["2018"], f"MD_{dataset.upper()}_2018_CORE.asc"),
            read_data(core_reference[dataset]["2017"], f"MD_{dataset.upper()}_2017_CORE.asc"),
            read_data(core_reference[dataset]["2016"], f"MD_{dataset.upper()}_2016_CORE.asc")
        ], ignore_index=True)
        dataset_core = dataset_filtering_function(dataset, dataset_core, proc_code_type)
        dataset_core = dataset_core.astype(core_reference[dataset]["dtypes"]).set_index("record_id")
        dataset_core["ICD-10"] = dataset_core["ICD-10"].transform(split_codes, col_name="ICD-10")
        dataset_core[proc_code_type] = dataset_core[proc_code_type].transform(split_codes, col_name=proc_code_type)
        pickle_file(f"{dataset}_core_filtered.pickle", analysis_name, dataset_core)
        del dataset_core

    process_dataset("sedd", "cpt_codes")
    process_dataset("sasd", "cpt_codes")
    process_dataset("sid", "ICD-10-procedures")