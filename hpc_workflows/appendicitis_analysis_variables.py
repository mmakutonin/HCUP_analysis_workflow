# This file contains variables that specify analysis constants needed to make the analysis pipeline applicable
# to multiple projects and research questions.
import pandas as pd
import numpy as np
import re

procedure_codes = { #currently appendectomy procedures, change for new procedure
    "cpt_codes": ["44950", "44955", "44960", "44970"],
    "ICD-10-procedures": ["0DTJ"]
}
diagnosis_codes = [ #currently for acute appendicitis
    "K35"
]
linker_table_diagnosis_codes = [ #currently for acute appendicitis
    "K35"
]

init_visit_datasets = {
    "sedd": True,
    "sasd": False,
    "sid": False,
    "sid_ed": True #rows in SID that represent patients admitted from the ED.
}

code_category_dict = {
    #Charlson Comorbidity Index variables (needed for 2.0), ICD-10 code definitions from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6684052
    "Myocardial Infarction History": ["I21", "I22", "I252"],
    "Heart Failure": ["I110", "I130", "I132", "I255", "I420", "I425", "I426", "I427", "I428", "I429", "I43", "I50", "P290"],
    "Peripheral Vascular Disease": ["I70", "I71", "I731", "I738", "I739", "I771", "I790", "I791", "I798", "K551", "K558", "K559", "Z958", "Z959"],
    "CVA/TIA": ["G45", "G46", "H340", "H341", "H342", "I60", "I61", "I62", "I63", "I64", "I65", "I66", "I67", "I68"],
    "Dementia": ["F01", "F02", "F03", "F04", "F05", "F061", "F068", "G132", "G138", "G30", "G310", "G311", "G312", "G914", "G94", "R4181", "R54"],
    "COPD": ["J4", "J60", "J61", "J62", "J63", "J64", "J65", "J66", "J67", "J684", "J701", "J703"],
    "Rheumatic Disease": ["M05", "M06", "M315", "M32", "M33", "M34", "M351", "M353", "M360"],
    "Peptic Ulcer Disease": ["K25", "K26", "K27", "K28"],
    "Mild Liver Disease": ["B18", "K700", "K701", "K702", "K703", "K709", "K713", "K714", "K715", "K717", "K73", "K74", "K760", "K762", "K763", "K764", "K768", "K769", "Z944"],
    "Severe Liver Disease": ["I850", "I864", "K704", "K711", "K721", "K729", "K765", "K766", "K767"],
    "Uncomplicated Diabetes": ["E080", "E081", "E086", "E088", "E089", "E090", "E091", "E096", "E098", "E099", "E100", "E101", "E106", "E108", "E109", "E110", "E111", "E116", "E118", "E119", "E130", "E131", "E136", "E138", "E139"],
    "Complicated Diabetes": ["E082", "E083", "E084", "E085", "E092", "E093", "E094", "E095", "E102", "E103", "E104", "E105", "E112", "E113", "E114", "E115", "E132", "E133", "E134", "E135"],
    "Hemiplegia": ["G041", "G114", "G800", "G801", "G802", "G81", "G82", "G83"],
    "Uncomplicated Renal Disease": ["I129", "I130", "I1310", "N03", "N05", "N181", "N182", "N183", "N184", "N189", "Z940"],
    "Severe Renal Disease": ["I120", "I1311", "N185", "N186", "N19", "N250", "Z49", "Z992"],
    "Malignancy": ["C"], #some will be superceded by invalid malignancies to save processing time
    "Metastatic Tumor": ["C77", "C78", "C79", "C800", "C802"],
    "Invalid Malignancy": ["C44", "C4A", "C64", "C65", "C66", "C67", "C68", "C69", "C70", "C71", "C72", "C73", "C74", "C75", "C7A", "C7B", "C86"],
    "HIV": ["B20"],
    "AIDS Opportunistic Infection": ["A021", "A072", "A073", "A1", "A31", "A812", "B00", "B25", "B37", "B38", "B39", "B45", "B58", "B59", "C53", "C46", "C8", "C9", "G934", "R64", "Z8701"],
    
    #Additional comorbidities
    "Diabetes Mellitus": ["E08", "E09", "E10", "E11", "E13", "O24", "Z8632"],
    "Hyperlipidemia": ["E780", "E781", "E782", "E783", "E784", "E785"],
    "Obesity": ["O9921", "O66", "Z683", "Z684"],
    "Hypertension":["I10", "I11", "I12", "I13", "I14", "I15", "I16", "O10", "O11", "O13", "O14", "O15", "O16"],
    "Ischemic Heart Disease": ["I20", "I21", "I22", "I23", "I24", "I25"],
    "Mood Disorders": ["F2", "F3", "F4"],
    "Aspirin": ["Z7982"],
    "Nicotine Dependence": ["F17", "T652", "Z87891", "Z720"],
    "Alcohol-Related Disorders": ["F10"],
    
    #Analysis-specific comorbidities
    "Antibiotic Treatment": [
        "J2543", "J0690", "J0696", "J0697", "J0698", "J0744", "J1956", "S0030", #UpToDate 2022
        "J1335", "J0694", "J0696", "J0690", "J1956", #CODA
        "J1335" #APPAC
    ],
    
    "pregnant": ["O", "Z33", "Z34", "Z3A"],
    "appendecolith": ["K381"],
    "bowel obstruction": ["K56"]
}

# TODO the current approach makes it so that the linker_table charts are **necessarily** a subset of revisit-eligible charts.
# If we need a different use case eventually, will need to be re-engineered.
def dataset_filtering_function(dataset_name, dataset_core, proc_code_type):
    return dataset_core[dataset_core["ICD-10"].transform(
        lambda x: any([x.startswith(code) for code in diagnosis_codes])
    )].copy()

def linker_table_filtering_function(dataset):
    return dataset[pd.concat([
        dataset["ICD-10"].astype('str').str.slice(2,9).str.contains(f"^{code}") for code in linker_table_diagnosis_codes
    ], axis=1).any(axis=1)].copy()

# Data Enrichment
# Separates records/patients into subgroups for statistical analysis. Currently _________
de_col_keys = [
    "Appendicitis Type",
    "Management Type",
    "Morphine Administration",
    "Opioid Administration",
    "Readmit with Complication",
    "Obtained Immediated Appendectomy",
    "IV Antibiotics on Index Visit",
    "Dataset Years"
]
de_col_values = {
    de_col_keys[0]: ["Complicated", "Uncomplicated"],
    de_col_keys[1]: [
        "Non-Surgical Management",
        "Immediate Appendectomy",
        "Delayed Appendectomy",
        "Return Emergency Appendectomy",
    ],
    de_col_keys[2]: ["Morphine on Initial Visit", "- Morphine"],
    de_col_keys[3]: ["Opioid on Initial Visit", "- Opioid"],
    de_col_keys[4]: ["New Complicated Appendicitis", "No New Complicated Appendicitis"],
    de_col_keys[5]: ["Immediate Appendectomy Obtained", "Initial Management Without Appendectomy"],
    de_col_keys[6]: ["IV Antibiotics", "No IV Antibiotics"],
    de_col_keys[7]: ["2016-18", "2019-21"],
}
def data_enrichment_function(sedd, sasd, sid, sid_ed, codes, linker_table):
    code_types = {
        de_col_keys[0]: ["K352", "K3520", "K3521", "K3531", "K3532", "K3533", "K35891"],
        de_col_keys[1]: [*procedure_codes["cpt_codes"], *procedure_codes["ICD-10-procedures"]],
        de_col_keys[2]: ["J2270", "J2274"],
        de_col_keys[3]: ["J0595", "J1170", "J1810", "J1960", "J2175", "J2270", "J2274", "J3010", "J3070"],
        de_col_keys[6]: code_category_dict["Antibiotic Treatment"]
    }
    def code_linker(code_type_key, init_visit="any", ed_visit="any",codes=codes):
        if init_visit != "any":
            codes = codes.query(f"init_chart == {init_visit}")
        if ed_visit != "any":
            codes = codes.query(f"ed_flag == {ed_visit}")
        return codes.reset_index().groupby("visit_link")["codes"].unique().transform(lambda ls: pd.Series(ls)).agg(
            lambda row: any([row.str.contains(code).any() for code in code_types[code_type_key]]),
            axis=1
        )
    
    #Label complicated/uncomplicated appendicitis
    linker_table[de_col_keys[0]] = code_linker(de_col_keys[0], True).map({
        True: de_col_values[de_col_keys[0]][0],
        False: de_col_values[de_col_keys[0]][1]
    })

    #Label surgery type
    linker_table[de_col_keys[1]] = code_linker(de_col_keys[1], True).map({
        True: de_col_values[de_col_keys[1]][1],
        False: de_col_values[de_col_keys[1]][0]
    })
    linker_table[de_col_keys[1]] = code_linker(de_col_keys[1], False, False).map({
        True: de_col_values[de_col_keys[1]][2],
        False: np.nan #this way the previous array won't update
    }).combine_first(linker_table[de_col_keys[1]])
    linker_table[de_col_keys[1]] = code_linker(de_col_keys[1], False, True).map({
        True: de_col_values[de_col_keys[1]][3],
        False: np.nan #this way the previous array won't update
    }).combine_first(linker_table[de_col_keys[1]])
    
    #Add admission indicator if emergency surgery obtained
    linker_table[de_col_keys[1]].eq(de_col_values[de_col_keys[1]][1]).map({
        True: True,
        False: np.nan
    }).combine_first(linker_table["Admitted"])
    #Label morphine administration
    linker_table[de_col_keys[2]] = code_linker(de_col_keys[2], True).map({
        True: de_col_values[de_col_keys[2]][0],
        False: de_col_values[de_col_keys[2]][1]
    })

    #Label opioid administration
    linker_table[de_col_keys[3]] = code_linker(de_col_keys[3], True).map({
        True: de_col_values[de_col_keys[3]][0],
        False: de_col_values[de_col_keys[3]][1]
    })

    #label whether a new-onset complication arose
    linker_table[de_col_keys[4]] = code_linker(de_col_keys[0], False)
    linker_table[de_col_keys[4]] = linker_table[de_col_keys[4]].fillna(False).map({
        True: de_col_values[de_col_keys[4]][0],
        False: de_col_values[de_col_keys[4]][1]
    })
    
    #Make management type binary
    linker_table[de_col_keys[5]] = linker_table[de_col_keys[1]].eq(de_col_values[de_col_keys[1]][1]).map({
        True: de_col_values[de_col_keys[5]][0],
        False: de_col_values[de_col_keys[5]][1]
    })

    #IV Abx on indev visit
    linker_table[de_col_keys[6]] = code_linker(de_col_keys[6], True).map({
        True: de_col_values[de_col_keys[6]][0],
        False: de_col_values[de_col_keys[6]][1]
    })

    #Label analysis years
    linker_table[de_col_keys[7]] = linker_table["initial_year"].apply(
        lambda year: de_col_values[de_col_keys[7]][0] if year <= 2018 else de_col_values[de_col_keys[7]][1]
    )
    
    return linker_table

demographic_table_configurations = [
    # {
    #     "key": "Admitted",
    #     "query_string": "Complicated == True",
    #     "save_filepath": f"../tables/Table 1 Admission Status Complicated.csv",
    #     "has_outcome_crosscomparison": True,
    #     "outcome_crosscomparison": [
    #         {
    #             "save_filepath": "../tables/Complicated - Cost of Admission vs Cost of Surgery.csv",
    #             "outcome_variable": "Cost (USD)",
    #             "groupby_row": de_col_values[de_col_keys[1]] #groupby_col is always the "key" attribute above
    #         }
    #     ]
    # },
    # {
    #     "key": de_col_keys[1],
    #     "query_string": "Complicated == True",
    #     "save_filepath": f"../tables/Table 1 Management Type Complicated.csv",
    #     "has_outcome_crosscomparison": False,
    # },
    # {
    #     "key": de_col_keys[1],
    #     "query_string": "Complicated == False",
    #     "save_filepath": f"../tables/Table 1 Management Type Uncomplicated.csv",
    #     "has_outcome_crosscomparison": False,
    # },
    # {
    #     "key": de_col_keys[5],
    #     "query_string": "Complicated == False",
    #     "save_filepath": f"../tables/Table 1 Surgery vs None Uncomplicated.csv",
    #     "has_outcome_crosscomparison": False,
    # },
    # {
    #     "key": "Admitted",
    #     "query_string": "Complicated == False",
    #     "save_filepath": f"../tables/Table 1 Admission Status Uncomplicated.csv",
    #     "has_outcome_crosscomparison": True,
    #     "outcome_crosscomparison": [
    #         {
    #             "save_filepath": "../tables/Uncomplicated - Cost of Admission vs Cost of Surgery.csv",
    #             "outcome_variable": "Cost (USD)",
    #             "groupby_row": de_col_values[de_col_keys[1]] #groupby_col is always the "key" attribute above
    #         }
    #     ]
    # },
    {
        "key": de_col_keys[1],
        "query_string": "(Complicated == False) and (`Pediatric (<18)` == False)",
        "save_name": f"Table 1 Management Type Uncomplicated Adult",
        "has_outcome_crosscomparison": False,
    },
    {
        "key": de_col_keys[5],
        "query_string": "(Complicated == False) and (`Pediatric (<18)` == False)",
        "save_name": f"Table 1 Surgery vs None Uncomplicated Adult",
        "has_outcome_crosscomparison": False,
    },
    {
        "key": de_col_keys[6],
        "query_string": "(Complicated == False) and (`Pediatric (<18)` == False)",
        "save_name": f"Table 1 IV Abx vs None Uncomplicated Adult",
        "has_outcome_crosscomparison": False,
    },
    {
        "key": de_col_keys[6],
        "query_string": f"(Complicated == False) and (`Pediatric (<18)` == False) and (`{de_col_values[de_col_keys[5]][1]}` == True)",
        "save_name": f"Table 1 IV Abx vs None in Medical Management Uncomplicated Adult",
        "has_outcome_crosscomparison": False,
    },
    # {
    #     "key": de_col_keys[1],
    #     "query_string": "(Complicated == False) and (`Pediatric (<18)` == True)",
    #     "save_filepath": f"../tables/Table 1 Management Type Uncomplicated Pediatric.csv",
    #     "has_outcome_crosscomparison": False,
    # },
    # {
    #     "key": de_col_keys[5],
    #     "query_string": "(Complicated == False) and (`Pediatric (<18)` == True)",
    #     "save_filepath": f"../tables/Table 1 Surgery vs None Uncomplicated Pediatric.csv",
    #     "has_outcome_crosscomparison": False,
    # }
    {
        "key": de_col_keys[7],
        "query_string": f"(Complicated == False) and (`Pediatric (<18)` == False)",
        "save_name": f"Table 1 2016-18 vs 2019-21 Uncomplicated Adult",
        "has_outcome_crosscomparison": False,
    },
]

linreg_targets = {
    "Predictors of Antibiotic Management Cost": \
        lambda dataset: dataset.loc[
            dataset["age"] >= 18,
            "Cost"
        ]
}

logreg_targets = {
    'Given Uncomplicated Appendicitis - Surgical vs Non-Surgical Management':\
        lambda dataset: dataset.loc[
            dataset[de_col_keys[0]] == de_col_values[de_col_keys[0]][1],
            de_col_keys[1]
        ].eq(de_col_values[de_col_keys[1]][1]),
    'Given Uncomplicated Appendicitis - Immediate Appendectomy vs Fully Non-Surgical Management':\
        lambda dataset: dataset.loc[
            (dataset[de_col_keys[0]] == de_col_values[de_col_keys[0]][1]) &
            (
                (dataset[de_col_keys[1]] == de_col_values[de_col_keys[1]][1]) |
                (dataset[de_col_keys[1]] == de_col_values[de_col_keys[1]][0])
            ),
            de_col_keys[1]
        ].eq(de_col_values[de_col_keys[1]][1]),
    'Given Uncomplicated Appendicitis and Initial Non-Surgical Management - Risks of Surgical Conversion':\
        lambda dataset: dataset.loc[
            (dataset[de_col_keys[0]] == de_col_values[de_col_keys[0]][1]) &
            (dataset[de_col_keys[1]] != de_col_values[de_col_keys[1]][1]),
            de_col_keys[1]
        ].ne(de_col_values[de_col_keys[1]][0]),
    'Given Uncomplicated Appendicitis and Immediate Appendectomy- Opioid vs None': \
        lambda dataset: dataset.loc[
            (dataset[de_col_keys[0]] == de_col_values[de_col_keys[0]][1])\
            & (dataset[de_col_keys[1]] == de_col_values[de_col_keys[1]][1]),
            de_col_keys[3]
        ].eq(de_col_values[de_col_keys[3]][0]),
    'Given Uncomplicated Appendicitis and Non-Surgical Management - Opioid vs None': \
        lambda dataset: dataset.loc[
            (dataset[de_col_keys[0]] == de_col_values[de_col_keys[0]][1])\
            & (dataset[de_col_keys[1]] != de_col_values[de_col_keys[1]][1]),
            de_col_keys[3]
        ].eq(de_col_values[de_col_keys[3]][0]),
    'Given Complicated Appendicitis - Surgical vs Non-Surgical Management':\
        lambda dataset: dataset.loc[
            dataset[de_col_keys[0]] == de_col_values[de_col_keys[0]][0],
            de_col_keys[1]
        ].eq(de_col_values[de_col_keys[1]][1]),
    'Given Complicated Appendicitis and Immediate Appendectomy- Opioid vs None': \
        lambda dataset: dataset.loc[
            (dataset[de_col_keys[0]] == de_col_values[de_col_keys[0]][0])\
            & (dataset[de_col_keys[1]] == de_col_values[de_col_keys[1]][1]),
            de_col_keys[3]
        ].eq(de_col_values[de_col_keys[3]][0]),
    'Given Complicated Appendicitis and Non-Surgical Management - Opioid vs None': \
        lambda dataset: dataset.loc[
            (dataset[de_col_keys[0]] == de_col_values[de_col_keys[0]][0])\
            & (dataset[de_col_keys[1]] != de_col_values[de_col_keys[1]][1]),
            de_col_keys[3]
        ].eq(de_col_values[de_col_keys[3]][0]),
}

outcome_cols = [
    # for outcome variable in de_col_keys, format is:
    # {"name": "de_col_key", "type": "string | number", "positive_class": "de_col_value"}
    {"name": "Cost", "type": "number"},
    {"name": "Inpatient Readmissions", "type": "number"},
    {"name": "ED Readmissions", "type": "number"},
    {"name": "Died", "type": "number"},
    {"name": "LOS", "type": "number"},
    {"name": "Readmit with Complication", "type": "string", "positive_class": "New Complication"}
]

chart_plotting_configurations = [
    {
        'metric':'Cost',
        'y_axis_label': 'Cost (USD)',
        'title': 'One-Year Cost of Immediate Appendectomy vs Non-Surgical Management on Initial Visit',
        'true_label': 'Appendectomy',
        'false_label': 'Non-Surgical Management',
        'logreg_key': 'Given Uncomplicated Appendicitis - Surgical vs Non-Surgical Management',
        'forecast_length': 0
    },
    {
        'metric':'Cost',
        'y_axis_label': 'Cost (USD)',
        'title': 'One-Year Cost of Immediate Appendectomy vs Non-Surgical Management Without Complications',
        'true_label': 'Immediate Appendectomy',
        'false_label': 'Non-Surgical Management',
        'logreg_key': 'Given Uncomplicated Appendicitis - Immediate Appendectomy vs Fully Non-Surgical Management',
        'forecast_length': 0
    },
    {
        'metric':'ED Readmissions',
        'y_axis_label': 'Repeat ED Visits per Patient',
        'title': 'One-Year Repeat ED Visit Rates for Immediate Appendectomy vs Non-Surgical Management on Initial Visit',
        'true_label': 'Appendectomy',
        'false_label': 'Non-Surgical Management',
        'logreg_key': 'Given Uncomplicated Appendicitis - Surgical vs Non-Surgical Management',
    },
    {
        'metric':'Inpatient Readmissions',
        'y_axis_label': 'Repeat Hospitalizations per Patient',
        'title': 'One-Year Repeat Hospitalization Rates for Immediate Appendectomy vs Non-Surgical Management on Initial Visit',
        'true_label': 'Appendectomy',
        'false_label': 'Non-Surgical Management',
        'logreg_key': 'Given Uncomplicated Appendicitis - Surgical vs Non-Surgical Management',
    },
    {
        'metric':'Cost',
        'y_axis_label': 'Cost (USD)',
        'title': 'One-Year Cost of Immediate Appendectomy vs Non-Surgical Management on Initial Visit for Complicated Appendicitis',
        'true_label': 'Appendectomy',
        'false_label': 'Non-Surgical Management',
        'logreg_key': 'Given Complicated Appendicitis - Surgical vs Non-Surgical Management',
    },
    {
        'metric':'ED Readmissions',
        'y_axis_label': 'Repeat ED Visits per Patient',
        'title': 'One-Year Repeat ED Visit Rates for Immediate Appendectomy vs Non-Surgical Management on Initial Visit for Complicated Appendicitis',
        'true_label': 'Appendectomy',
        'false_label': 'Non-Surgical Management',
        'logreg_key': 'Given Complicated Appendicitis - Surgical vs Non-Surgical Management',
    },
    {
        'metric':'Inpatient Readmissions',
        'y_axis_label': 'Repeat Hospitalizations per Patient',
        'title': 'One-Year Repeat Hospitalization Rates for Immediate Appendectomy vs Non-Surgical Management on Initial Visit for Complicated Appendicitis',
        'true_label': 'Appendectomy',
        'false_label': 'Non-Surgical Management',
        'logreg_key': 'Given Complicated Appendicitis - Surgical vs Non-Surgical Management',
    },
]