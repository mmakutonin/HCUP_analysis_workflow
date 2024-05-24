import pandas as pd
import numpy as np
from utility_functions import load_file, pickle_file
data_conversion_dict = {
    "initial_year": lambda col: col.astype("int"),
    "female": lambda col: col.map({0: "Male", 1:"Female"}),
    "homeless": lambda col: col.map({0: "Not Homeless", 1:"Homeless"}),
    "married": lambda col: col.map({
        "I": "Single",
        "M": "Married",
        "A": "Common Law",
        "B": "Registered Domestic Partner",
        "S": "Separated",
        "X": "Legally Separated",
        "D": "Divorced",
        "W": "Widowed",
        "U": "Unmarried"
    }),
    "payer": lambda col: col.map({
        1: "Medicare",
        2: "Medicaid",
        3: "Private insurance",
        4: "Self-pay",
        5: "No charge",
        6: "Other"
    }),
    "race": lambda col: col.map({
        1: "White",
        2: "African-American",
        3: "Hispanic",
        4: "Asian",
        5: "Native American",
        6: "Other"
    }),
    "median_zip_income": lambda col: col.replace(-99, np.nan)
}
col_renaming_dict = {
    "female" : "gender",
    "married": "marital_status"
}
additional_columns_dict = {
    "Pediatric": lambda dataset: dataset["age"] <18,
    "Adult": lambda dataset: (dataset["age"] >=18) & (dataset["age"] < 65),
    "Geriatric": lambda dataset: dataset["age"] >=65,
    "age_groups": lambda dataset: pd.cut(dataset["age"], [0,30,40,60,80,200], labels=["30-", "30-40", "40-60", "60-80", "80+"])
}

def convert_data_values_to_readable_format(analysis_name:str):
    dataset = load_file("fully_filtered_summary.pickle", analysis_name)
    output_dataset = dataset.copy()
    for key, func in data_conversion_dict.items():
        output_dataset[key] = func(dataset[key])
    output_dataset.rename(columns=col_renaming_dict, inplace=True)
    for key, func in additional_columns_dict.items():
        output_dataset[key] = func(dataset)
    pickle_file(f"summary_enhanced.pickle", analysis_name, output_dataset)