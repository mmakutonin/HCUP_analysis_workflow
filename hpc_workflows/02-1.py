from utility_functions import load_file, pickle_file, starting_run, print_to_drop
from data_reading_functions import read_data, died_reference
import pandas as pd
import numpy as np

def note_deaths_in_study_period(analysis_name:str):
    summary_table = load_file("filtered_dataset.pickle", analysis_name)
    codes_table = load_file("filtered_dataset_codes.pickle", analysis_name)
    index = summary_table["max_year"].reset_index().astype('str').set_index('visit_link') #to prevent corrupt data from erroring an astype()
    death_records = []
    for dataset in ["sedd", "sid", "sasd"]:
        for year in [
            "2016", "2017", "2018",
            "2019", "2020", "2021"
        ]:
            starting_run(dataset + year)
            data_file = read_data(died_reference[dataset][year], f"MD_{dataset.upper()}_{year}_CORE.asc")
            data_file['visit_link'] = data_file['visit_link'].str.strip() #to prevent corrupt data from erroring an astype()
            deaths = data_file.join(index, how="inner", on="visit_link").astype('int').query(f"max_year >={year} and Died == 1")
            death_records.append(deaths)
            del data_file
            del deaths
    #Dropping duplicates due to some HCUP data vagaries
    deaths = pd.concat(death_records).sort_values("record_id", ascending=False).drop_duplicates('visit_link').set_index('visit_link')
    invalid_deaths = deaths.join(summary_table, how="right", lsuffix="_died").query("Died == 1 and record_id == initial_record_id")
    fully_filtered_summary = summary_table.join(deaths["Died"], how="left").drop(index=invalid_deaths.index)
    fully_filtered_summary["Died"] = fully_filtered_summary["Died"].fillna(0)
    print_to_drop(f"dropped {invalid_deaths.shape[0]} invalid patient(s) due to death on initial record.", analysis_name)
    print_to_drop(f"dropped {np.logical_not(deaths.index.isin(summary_table.index)).sum()} invalid patient(s) due to death from unrelated cause.", analysis_name)
    pickle_file("fully_filtered_summary.pickle", analysis_name, fully_filtered_summary)
    pickle_file("fully_filtered_codes.pickle", analysis_name, codes_table.loc[fully_filtered_summary.index])