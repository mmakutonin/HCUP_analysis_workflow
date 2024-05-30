# %%
import pandas as pd
from utility_functions import load_file, pickle_file, starting_run, finished_run

# %%
data_conversion_dict = {
    "initial_year": [lambda col: col.astype("int")],
    "female": [lambda col: col.map({0: "Male", 1:"Female"}), "gender"],
    "homeless": [lambda col: col.map({0: "Not Homeless", 1:"Homeless"})],
    "married": [lambda col: col.map({
        "I": "Single",
        "M": "Married",
        "A": "Common Law",
        "B": "Registered Domestic Partner",
        "S": "Separated",
        "X": "Legally Separated",
        "D": "Divorced",
        "W": "Widowed",
        "U": "Unmarried"
    }), "marital_status"],
    "payer": [lambda col: col.map({
        1: "Medicare",
        2: "Medicaid",
        3: "Private insurance",
        4: "Self-pay",
        5: "No charge",
        6: "Other"
    })],
    "race": [lambda col: col.map({
        1: "White",
        2: "African-American",
        3: "Hispanic",
        4: "Asian",
        5: "Native American",
        6: "Other"
    })],
    "median_zip_income": [lambda col: col.replace(-99, None)]
}

additional_columns_dict = {
    #Age Groups
    "Pediatric": lambda dataset: dataset["age"] <18,
    "Adult": lambda dataset: (dataset["age"] >=18) & (dataset["age"] < 65),
    "Geriatric": lambda dataset: dataset["age"] >=65,
    "age_groups": lambda dataset: pd.cut(dataset["age"], [0,30,40,60,80,200], labels=["30-", "30-40", "40-60", "60-80", "80+"])
}

# %%
dataset = load_file("fully_filtered_summary.pickle")
for key, func in data_conversion_dict.items():
    store_key = key if len(func) == 1 else func[1]
    func = func[0]
    dataset[store_key] = func(dataset[key])
    if key != store_key:
        del dataset[key]
for key, func in additional_columns_dict.items():
    dataset[key] = func(dataset)
pickle_file(f"summary_enhanced.pickle", dataset)

# %%



