# This file contains variables that specify analysis constants needed to make the analysis pipeline applicable
# to multiple projects and research questions.
import pandas as pd
import numpy as np

procedure_codes = { #currently optho procedures, change for new procedure
    "cpt_codes": [],
    "ICD-10-procedures": []
}
diagnosis_codes = [ #currently for eye injury
    "S0500XA", #corneal abrasion,
    "S0501XA", #corneal abrasion,
    "S0502XA", #corneal abrasion,
]

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
    # "S05", #ocular injury
    # "S011", #wound of eyelid +/- eye
    # "S001", #eyelid contusion or other injury
    # "S021", "S023", "S028", #ocular skull fracture
    # "H[0-6]" #conjunctivitis
}

def dataset_filtering_function(dataset_name, dataset_core, proc_code_type):
    return dataset_core[dataset_core["ICD-10"].transform(
        lambda x: any([x.startswith(code) for code in diagnosis_codes])
    )].copy()


# Data Enrichment
# Separates records/patients into subgroups for statistical analysis. Currently surgery_type
de_col_keys = [
    "Optho Exam on Index Visit",
    "Optho Consult on Index Visit"
]
de_col_values = {
    de_col_keys[0]: ["Optho Exam", "No Optho Exam",],
    de_col_keys[1]: ["Optho Consult", "No Optho Consult"],
}
def data_enrichment_function(sedd, sasd, sid, sid_ed, codes, linker_table):
    code_types = {
        de_col_keys[0]: [
            "08J0", "08J1", "08JJ", "08JK", #ICD-10-PCS Eye Inspection
            "920", "921", "9220", "9221", "9222", "9223", "9224", "9225", "92260" #CPT codes opthalmologic/opthalmoscopic exams
        ],
        de_col_keys[1]: [
            "99241", "99242", "99243", "99244", "92245", #CPT codes
        ]
    }
    def code_linker(code_type_key, init_visit="any", ed_visit="any",codes=codes):
        if init_visit != "any":
            codes = codes.query(f"init_chart == {init_visit}")
        if ed_visit != "any":
            codes = codes.query(f"ed_flag == {ed_visit}")
        return codes.reset_index().\
            groupby("visit_link")["codes"].unique().apply(lambda x: [st.strip() for st in x]).\
            transform(lambda x: any([code in str(x) for code in code_types[code_type_key]]))
    
    linker_table[de_col_keys[0]] = code_linker(de_col_keys[0], init_visit=True).map({
        True: de_col_values[de_col_keys[0]][0],
        False: de_col_values[de_col_keys[0]][1]
    })

    linker_table[de_col_keys[1]] = code_linker(de_col_keys[1], init_visit=True).map({
        True: de_col_values[de_col_keys[1]][0],
        False: de_col_values[de_col_keys[1]][1]
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
    {
        "key": de_col_keys[0],
        "query_string": "`Cost (USD)` >= 0",
        "save_filepath": f"../tables/Table 1 Optho Exam.csv",
        "has_outcome_crosscomparison": True,
        "outcome_crosscomparison": [
            {
                "save_filepath": "../tables/Cost of Eye Exam vs Optho Consult.csv",
                "outcome_variable": "Cost (USD)",
                "groupby_row": de_col_values[de_col_keys[1]] #groupby_col is always the "key" attribute above
            }
        ]
    },
    {
        "key": de_col_keys[1],
        "query_string": "`Cost (USD)` >= 0",
        "save_filepath": f"../tables/Table 1 Optho Consult.csv",
        "has_outcome_crosscomparison": True,
        "outcome_crosscomparison": [
            {
                "save_filepath": "../tables/Cost of Consult vs Admission.csv",
                "outcome_variable": "Cost (USD)",
                "groupby_row": de_col_values[de_col_keys[1]] #TODO: vs admission
            }
        ]
    },
    {
        "key": "payer",
        "query_string": "`Cost (USD)` >= 0",
        "save_filepath": f"../tables/Table 1 Insurance Status.csv",
        "has_outcome_crosscomparison": False,
    }

]

linreg_targets = {
    # "Predictors of Antibiotic Management Cost": \
    #     lambda dataset: dataset.loc[
    #         dataset["Pediatric (<18)"] == False,
    #         "Cost"
    #     ]
}

outcome_cols = [
    # for outcome variable in de_col_keys, format is:
    # {"name": "de_col_key", "type": "string", "positive_class": "de_col_value"}
    {"name": "Cost", "type": "number"},
    {"name": "Inpatient Readmissions", "type": "number"},
    {"name": "ED Readmissions", "type": "number"},
    {"name": "Died", "type": "number"},
    {"name": "LOS", "type": "number"},
    {"name": "Readmit with Complication", "type": "string", "positive_class": "New Complication"}
]

chart_plotting = [ #TODO: charting
    # {
    #     'metric':'Cost',
    #     'y_axis_label': 'Cost (USD)',
    #     'title': 'One-Year Cost of Immediate Appendectomy vs Non-Surgical Management on Initial Visit',
    #     'true_label': 'Appendectomy',
    #     'false_label': 'Non-Surgical Management',
    #     'logreg_key': 'Given Uncomplicated Appendicitis - Surgical vs Non-Surgical Management',
    #     'forecast_length': 0
    # },
    # {
    #     'metric':'Cost',
    #     'y_axis_label': 'Cost (USD)',
    #     'title': 'One-Year Cost of Immediate Appendectomy vs Non-Surgical Management Without Complications',
    #     'true_label': 'Immediate Appendectomy',
    #     'false_label': 'Non-Surgical Management',
    #     'logreg_key': 'Given Uncomplicated Appendicitis - Immediate Appendectomy vs Fully Non-Surgical Management',
    #     'forecast_length': 0
    # },
    # {
    #     'metric':'ED Readmissions',
    #     'y_axis_label': 'Repeat ED Visits per Patient',
    #     'title': 'One-Year Repeat ED Visit Rates for Immediate Appendectomy vs Non-Surgical Management on Initial Visit',
    #     'true_label': 'Appendectomy',
    #     'false_label': 'Non-Surgical Management',
    #     'logreg_key': 'Given Uncomplicated Appendicitis - Surgical vs Non-Surgical Management',
    # },
    # {
    #     'metric':'Inpatient Readmissions',
    #     'y_axis_label': 'Repeat Hospitalizations per Patient',
    #     'title': 'One-Year Repeat Hospitalization Rates for Immediate Appendectomy vs Non-Surgical Management on Initial Visit',
    #     'true_label': 'Appendectomy',
    #     'false_label': 'Non-Surgical Management',
    #     'logreg_key': 'Given Uncomplicated Appendicitis - Surgical vs Non-Surgical Management',
    # },
    # {
    #     'metric':'Cost',
    #     'y_axis_label': 'Cost (USD)',
    #     'title': 'One-Year Cost of Immediate Appendectomy vs Non-Surgical Management on Initial Visit for Complicated Appendicitis',
    #     'true_label': 'Appendectomy',
    #     'false_label': 'Non-Surgical Management',
    #     'logreg_key': 'Given Complicated Appendicitis - Surgical vs Non-Surgical Management',
    # },
    # {
    #     'metric':'ED Readmissions',
    #     'y_axis_label': 'Repeat ED Visits per Patient',
    #     'title': 'One-Year Repeat ED Visit Rates for Immediate Appendectomy vs Non-Surgical Management on Initial Visit for Complicated Appendicitis',
    #     'true_label': 'Appendectomy',
    #     'false_label': 'Non-Surgical Management',
    #     'logreg_key': 'Given Complicated Appendicitis - Surgical vs Non-Surgical Management',
    # },
    # {
    #     'metric':'Inpatient Readmissions',
    #     'y_axis_label': 'Repeat Hospitalizations per Patient',
    #     'title': 'One-Year Repeat Hospitalization Rates for Immediate Appendectomy vs Non-Surgical Management on Initial Visit for Complicated Appendicitis',
    #     'true_label': 'Appendectomy',
    #     'false_label': 'Non-Surgical Management',
    #     'logreg_key': 'Given Complicated Appendicitis - Surgical vs Non-Surgical Management',
    # },
]