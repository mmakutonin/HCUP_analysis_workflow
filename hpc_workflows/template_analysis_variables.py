# This file contains variables that specify analysis constants needed to make the analysis pipeline applicable
# to multiple projects and research questions.
import pandas as pd
import numpy as np

procedure_codes = { #currently ____ procedures, change for new procedure
    "cpt_codes": [],
    "ICD-10-procedures": []
}
diagnosis_codes = [ #currently for _______
    # "ICDCODE", # No periods, regex allowed
]
linker_table_diagnosis_codes = [ #currently for _______
    # "ICDCODE", # No periods, regex allowed
]

init_visit_datasets = {
    "sedd": True,
    "sasd": True,
    "sid": True,
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
    # "ICDCODE", # No periods, regex allowed
}

# TODO the current approach makes it so that the linker_table charts are **necessarily** a subset of revisit-eligible charts.
# If we need a different use case eventually, will need to be re-engineered.
def dataset_filtering_function(dataset_name, dataset_core, proc_code_type):
    icd_code_matches = [dataset_core["ICD-10"].str.contains(f"^{code}") for code in diagnosis_codes]
    chief_complaint_matches = [dataset_core["chief_complaint"].str.contains(f"^{code}") for code in diagnosis_codes] if "sid" not in dataset_name else []
    return dataset_core[pd.concat([*icd_code_matches, *chief_complaint_matches], axis=1).any(axis=1)].copy()

def linker_table_filtering_function(dataset):
    return dataset[pd.concat([
        dataset["ICD-10"].astype('str').str.slice(2,9).str.contains(f"^{code}") for code in linker_table_diagnosis_codes
    ], axis=1).any(axis=1)].copy()


# Data Enrichment
# Separates records/patients into subgroups for statistical analysis. Currently _________
    
de_col_keys = [
    # "Key Name", ...
]
de_col_numerical_cols = [
    # de_col_key[X], ...
]
de_col_values = {
    # de_col_keys[0]: ["Key Value", ...],
}
def data_enrichment_function(sedd, sasd, sid, sid_ed, codes, linker_table):
    code_types = {
        # de_col_keys[0]: ["CPT or ICD Codes, no regex"], ...
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
    
    # linker_table[de_col_keys[0]] = code_linker(de_col_keys[0], True).map({
    #     True: de_col_values[de_col_keys[0]][0],
    #     False: de_col_values[de_col_keys[0]][1]
    # })

    # linker_table[de_col_keys[1]] = code_linker(de_col_keys[1], True).map({
    #     True: de_col_values[de_col_keys[1]][1],
    #     False: de_col_values[de_col_keys[1]][0]
    # })
    # linker_table[de_col_keys[1]] = code_linker(de_col_keys[1], False, False).map({
    #     True: de_col_values[de_col_keys[1]][2],
    #     False: np.nan #this way the previous array won't update
    # }).combine_first(linker_table[de_col_keys[1]])
    
    return linker_table

demographic_table_configurations = [
    # {
    #     "key": de_col_keys[0],
    #     "query_string": "`Cost (USD)` >= 0", #Universal filter
    #     "save_name": f"table name",
    #     "has_outcome_crosscomparison": True,
    #     "outcome_crosscomparison": [
    #         {
    #             "save_name": "table name",
    #             "outcome_variable": "Cost (USD)",
    #             "groupby_row": de_col_values[de_col_keys[1]] #groupby_col is always the "key" attribute above
    #         }
    #     ]
    # },
]

linreg_targets = {
    # "Analysis Name": \
    #     lambda dataset: dataset.loc[
    #         Filtering Rule,
    #         "Outcome Column"
    #     ]
}

logreg_targets = {
    # 'Analysis Name':\
    #     lambda dataset: dataset.loc[
    #         Filtering Rule,
    #         Outcome Column
    #     ].eq(Target Value),
}

'''Define columns to be tested via univariate T-test in 03-2.py'''
outcome_cols = [
    # for outcome variable in de_col_keys, format is:
    # {"name": "de_col_key", "type": "string | number", "positive_class": "de_col_value"}
    {"name": "Cost", "type": "number"},
    {"name": "Inpatient Readmissions", "type": "number"},
    {"name": "ED Readmissions", "type": "number"},
    {"name": "Died", "type": "number"},
    {"name": "LOS", "type": "number"},
]

chart_plotting_configurations = [
    # {
    #     'metric':'Cost',
    #     'y_axis_label': 'Cost (USD)',
    #     'title': 'Analysis Title',
    #     'true_label': 'Label for True from logreg values',
    #     'false_label': 'Label for False from logreg values',
    #     'logreg_key': 'Analysis Name in logreg targets',
    #     'forecast_length': 0 # Number of quarters to forecast based on slope of Q4 segment
    # },
]