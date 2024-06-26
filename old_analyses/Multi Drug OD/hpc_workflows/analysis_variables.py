# This file contains variables that specify analysis constants needed to make the analysis pipeline applicable
# to multiple projects and research questions.
import pandas as pd
import numpy as np
import re

procedure_codes = { #currently ____ procedures, change for new procedure
    "cpt_codes": [],
    "ICD-10-procedures": []
}
diagnosis_codes = [ #currently for overdoses/poisonings of psychoactive drugs
    "T40", #Opioid poisoning
    "T51",  #alcohol poisoning
    "F1[01].2",   #0 alcohol, 1 opioid, 2 cannabis, 3 sedative-hypnotics and anxiolytics, 4 cocaine, 5 other stimulant, 6 hallucinogen, 7 nicotine, 8 inhalants, 9 other
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
    "Alcohol-Related Disorders": ["F10.[013-9]"],
    
    #Analysis-specific comorbidities
    "Suicidal ideation": ["R45851"],
    "Depression": ["F32", "F33"],
    # "Alcohol Abuse": [],
    # "Alcohol Dependence": [],
}

def dataset_filtering_function(dataset_name, dataset_core, proc_code_type):
    return dataset_core[pd.concat([
        dataset_core["ICD-10"].str.contains(f"^{code}") for code in diagnosis_codes
    ], axis=1).any(axis=1)].copy()


# Data Enrichment
# Separates records/patients into subgroups for statistical analysis. Currently _________
de_col_keys = [
   "Alcohol Intoxication",
   "Opioid Intoxication",
   "Cannabis Intoxication",
   "Sedative-hypnotic or Anxiolytic Intoxication",
   "Sedative Intoxication",
   "Cocaine Intoxication",
   "Stimulant Intoxication",
   "Hallucinogen Intoxication",
   "Inhalant Intoxication",

   "Concurrent Non-Alcohol Drug Use with Alcohol",
   "Concurrent Alcohol Use with Non-Alcohol Drugs",

   "Ventilator on Initial Visit",
]
de_col_values = {
    de_col_keys[0]: ["Concurrent Alcohol Intoxication", "No Alcohol Intoxication"],
    de_col_keys[1]: ["Concurrent Opioid Intoxication", "No Opioid Intoxication"],
    de_col_keys[2]: ["Concurrent Cannabis Intoxication", "No Cannabis Intoxication"],
    de_col_keys[3]: ["Concurrent Sedative-hypnotic Intoxication", "No Sedative-hypnotic Intoxication"],
    de_col_keys[4]: ["Concurrent Sedative Intoxication", "No Sedative Intoxication"],
    de_col_keys[5]: ["Concurrent Cocaine Intoxication", "No Cocaine Intoxication"],
    de_col_keys[6]: ["Concurrent Stimulant Intoxication", "No Stimulant Intoxication"],
    de_col_keys[7]: ["Concurrent Hallucinogen Intoxication", "No Hallucinogen Intoxication"],
    de_col_keys[8]: ["Concurrent Inhalant Intoxication", "No Inhalant Intoxication"],

    de_col_keys[9]:["Concurrent Drug Use", "Pure Alcohol Poisoning"],
    de_col_keys[10]: ["Concurrent Alcohol Use", "Isolated Drug Use"],

    de_col_keys[11]: ["Mechanical Ventilator Use", "No Mechanical Ventilator Use"]
    
}
def data_enrichment_function(sedd, sasd, sid, sid_ed, codes, linker_table):
    code_types = {
        de_col_keys[0]: ["F10.2", "T51"],
        de_col_keys[1]: ["F11.2", "T40[0-46]"],
        de_col_keys[2]: ["F12.2", "T407"],
        de_col_keys[3]: ["F13.2", "T4[1346]"],
        de_col_keys[4]: ["F1[013]", "T41", "T40[0-4]" "T42[34678]", "T43[0-5]", "T510"], #TODO
        de_col_keys[5]: ["F14.2", "T405"],
        de_col_keys[6]: ["F1[4-5].2", "T40[58]", "T436"], #TODO
        de_col_keys[7]: ["F16.2", "T40[89]"],
        de_col_keys[8]: ["F18.2"],

        de_col_keys[9]: ["F1[1-9].2", "T4"],
        de_col_keys[10]: ["F10.2", "T51"],

        de_col_keys[11]: ["94669", "5A1945Z"]
    }
    def code_linker(code_type_key, init_visit="any", ed_visit="any",codes=codes):
        if init_visit != "any":
            codes = codes.query(f"init_chart == {init_visit}")
        if ed_visit != "any":
            codes = codes.query(f"ed_flag == {ed_visit}")
        return codes.reset_index().groupby("visit_link")["codes"].unique()\
            .transform(lambda x: pd.Series(x)).agg(
                lambda row: any([row.str.contains(code).any() for code in code_types[code_type_key]]),
                axis = 1
            )
    
    for key in code_types.keys():
        linker_table[key] = code_linker(key, init_visit=True).map({
            True: de_col_values[key][0],
            False: de_col_values[key][1]
        })
    
    return linker_table

sid_inflation_adjustment = pd.DataFrame([
    #needed for 2.3, based on annual inflation data from Bureau of Labor Statistics
    #https://data.bls.gov/pdq/SurveyOutputServlet
    {'year':2016, 'discharge_quarter': 1, 'inflation_adjustment': 237.111/251.107},
    {'year':2016, 'discharge_quarter': 2, 'inflation_adjustment': 240.229/251.107},
    {'year':2016, 'discharge_quarter': 3, 'inflation_adjustment': 240.849/251.107},
    {'year':2016, 'discharge_quarter': 4, 'inflation_adjustment': 241.353/251.107},
    {'year':2017, 'discharge_quarter': 1, 'inflation_adjustment': 243.603/251.107},
    {'year':2017, 'discharge_quarter': 2, 'inflation_adjustment': 244.733/251.107},
    {'year':2017, 'discharge_quarter': 3, 'inflation_adjustment': 245.519/251.107},
    {'year':2017, 'discharge_quarter': 4, 'inflation_adjustment': 246.669/251.107},
    {'year':2018, 'discharge_quarter': 1, 'inflation_adjustment': 248.991/251.107},
    {'year':2018, 'discharge_quarter': 2, 'inflation_adjustment': 251.588/251.107},
    {'year':2018, 'discharge_quarter': 3, 'inflation_adjustment': 252.146/251.107},
    {'year':2018, 'discharge_quarter': 4, 'inflation_adjustment': 252.038/251.107},
]).set_index(['year', 'discharge_quarter'])

demographic_tables = [
    #Concurrent Alcohol Intoxication
    {
        "key": de_col_keys[9],
        "query_string": f"`{de_col_values[de_col_keys[0]][0]}` == True",
        "save_filepath": f"../tables/Alcohol Poisoning.csv",
        "has_outcome_crosscomparison": False,
        # "outcome_crosscomparison": [
        #     {
        #         "save_filepath": "../tables/table name.csv",
        #         "outcome_variable": "Cost (USD)",
        #         "groupby_row": de_col_values[de_col_keys[1]] #groupby_col is always the "key" attribute above
        #     }
        # ]
    },
    {
        "key": de_col_keys[10],
        "query_string": f"`{de_col_values[de_col_keys[9]][0]}` == True",
        "save_filepath": f"../tables/Drug Poisoning.csv",
        "has_outcome_crosscomparison": True,
        "outcome_crosscomparison": [
            {
                "save_filepath": "../tables/Alcohol co-intoxcation vs psychiatric dx.csv",
                "outcome_variable": "Cost (USD)",
                "groupby_row": ["Mood Disorders"] #groupby_col is always the "key" attribute above
            },
            {
                "save_filepath": "../tables/Alcohol co-intoxcation vs depression history.csv",
                "outcome_variable": "Cost (USD)",
                "groupby_row": ["Depression"] #groupby_col is always the "key" attribute above
            }
        ]
    },
    {
        "key": de_col_keys[1],
        "query_string": f"`Cost (USD)` >= 0",
        "save_filepath": f"../tables/Opioid Use Description.csv",
        "has_outcome_crosscomparison": False
    }
]

linreg_targets = {
    # "Analysis Name": \
    #     lambda dataset: dataset.loc[
    #         Filtering Rule,
    #         "Outcome Column"
    #     ]
}

logreg_targets = {
    # Need logreg for combined drugs and each drug class with alcohol
    'Concurrent Alcohol Poisoning vs Pure Drug Poisoning':\
        lambda dataset: dataset.loc[
            dataset[de_col_keys[9]] == de_col_values[de_col_keys[9]][0],
            de_col_keys[10]
        ].eq(de_col_values[de_col_keys[10]][0]),
    'Given Opioid Use - Concurrent Alcohol Poisoning vs Pure Drug Poisoning':\
        lambda dataset: dataset.loc[
            dataset[de_col_keys[1]] == de_col_values[de_col_keys[1]][0],
            de_col_keys[10]
        ].eq(de_col_values[de_col_keys[10]][0]),
    'Given Cannabis Use - Concurrent Alcohol Poisoning vs Pure Drug Poisoning':\
        lambda dataset: dataset.loc[
            dataset[de_col_keys[2]] == de_col_values[de_col_keys[2]][0],
            de_col_keys[10]
        ].eq(de_col_values[de_col_keys[10]][0]),
    'Given Sedative-Hypnotic Use - Concurrent Alcohol Poisoning vs Pure Drug Poisoning':\
        lambda dataset: dataset.loc[
            dataset[de_col_keys[3]] == de_col_values[de_col_keys[3]][0],
            de_col_keys[10]
        ].eq(de_col_values[de_col_keys[10]][0]),
    'Given Sedative Use - Concurrent Alcohol Poisoning vs Pure Drug Poisoning':\
        lambda dataset: dataset.loc[
            dataset[de_col_keys[4]] == de_col_values[de_col_keys[4]][0],
            de_col_keys[10]
        ].eq(de_col_values[de_col_keys[10]][0]),
    'Given Cocaine Use - Concurrent Alcohol Poisoning vs Pure Drug Poisoning':\
        lambda dataset: dataset.loc[
            dataset[de_col_keys[5]] == de_col_values[de_col_keys[5]][0],
            de_col_keys[10]
        ].eq(de_col_values[de_col_keys[10]][0]),
    'Given Stimulant Use - Concurrent Alcohol Poisoning vs Pure Drug Poisoning':\
        lambda dataset: dataset.loc[
            dataset[de_col_keys[6]] == de_col_values[de_col_keys[6]][0],
            de_col_keys[10]
        ].eq(de_col_values[de_col_keys[10]][0]),
    'Given Hallucinogen Use - Concurrent Alcohol Poisoning vs Pure Drug Poisoning':\
        lambda dataset: dataset.loc[
            dataset[de_col_keys[7]] == de_col_values[de_col_keys[7]][0],
            de_col_keys[10]
        ].eq(de_col_values[de_col_keys[10]][0]),
    
}

outcome_cols = [
    # for outcome variable in de_col_keys, format is:
    # {"name": "de_col_key", "type": "string | number", "positive_class": "de_col_value"}
    {"name": "Cost", "type": "number"},
    {"name": "Inpatient Readmissions", "type": "number"},
    {"name": "ED Readmissions", "type": "number"},
    {"name": "Died", "type": "number"},
    {"name": "LOS", "type": "number"},
    {"name": de_col_keys[11], "type": "string", "positive_class": de_col_values[de_col_keys[11]][0]},
]

chart_plotting = [
    {
        'metric':'ED Readmissions',
        'y_axis_label': 'Cost (USD)',
        'title': 'Cost Per Quarter for OUD',
        'true_label': 'Alcohol Plus Opioids',
        'false_label': 'Just Opioids',
        'logreg_key': 'Given Opioid Use - Concurrent Alcohol Poisoning vs Pure Drug Poisoning',
        'forecast_length': 0 # Number of quarters to forecast based on slope of Q4 segment
    },
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